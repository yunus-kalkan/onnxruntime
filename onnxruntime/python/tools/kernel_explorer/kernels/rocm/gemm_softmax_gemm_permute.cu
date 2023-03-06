// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/rocm/gemm_softmax_gemm_permute.h"

#include "pybind11/stl.h"

#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

#include <vector>

namespace py = pybind11;

using namespace onnxruntime::contrib::rocm;

namespace onnxruntime {

template <typename T>
class IGemmSoftmaxGemmPermuteKernelExplorer : public IKernelExplorer {
 public:
  IGemmSoftmaxGemmPermuteKernelExplorer(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t mask_dim,
      double scale,
      DeviceArray& Q,
      DeviceArray& K,
      DeviceArray& V,
      std::optional<DeviceArray>& attn_mask,
      DeviceArray& out) {
    attn_.batch_size = batch;
    attn_.sequence_length = seqlen;
    attn_.kv_sequence_length = -1;             // NOTE: not used
    attn_.past_sequence_length = -1;           // NOTE: not used
    attn_.original_past_sequence_length = -1;  // NOTE: not used
    attn_.total_sequence_length = total_seqlen;
    attn_.max_sequence_length = -1;  // TODO: set
    attn_.hidden_size = num_heads * head_size;
    attn_.head_size = head_size;
    attn_.v_hidden_size = attn_.hidden_size;  // Q,K,V hidden size must agree now
    attn_.v_head_size = attn_.head_size;      // Q,K,V hidden size must agree now
    attn_.num_heads = num_heads;
    attn_.is_unidirectional = false;
    attn_.past_present_share_buffer = false;
    attn_.do_rotary = false;
    attn_.mask_filter_value = -10000.0f;
    attn_.scale = scale;
    if (mask_dim == 0) {
      attn_.mask_type = contrib::MASK_NONE;
    } else if (mask_dim == 2) {
      attn_.mask_type = contrib::MASK_2D_KEY_PADDING;
    } else if (mask_dim == 3) {
      attn_.mask_type = contrib::MASK_3D_ATTENTION;
    } else if (mask_dim == 2) {
      attn_.mask_type = contrib::MASK_4D_MEGATRON;
    }

    auto device_prop = GetEp()->GetDeviceProp();
    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    params_.handle = rocblas_handle_;
    params_.attention = &attn_;
    params_.device_prop = &device_prop;
    params_.scale = scale;

    params_.q_buffer = reinterpret_cast<T*>(Q.ptr());
    params_.k_buffer = reinterpret_cast<T*>(K.ptr());
    params_.v_buffer = reinterpret_cast<T*>(V.ptr());
    if (attn_mask.has_value()) {
      params_.mask_index_buffer = reinterpret_cast<int*>(attn_mask->ptr());
      if (mask_dim == 2) {
        params_.mask_index_dims = {batch, total_seqlen};
      } else if (mask_dim == 3) {
        params_.mask_index_dims = {batch, seqlen, total_seqlen};
      } else {
        int max_seqlen = 1024;  // FIXME:
        params_.mask_index_dims = {batch, 1, max_seqlen, max_seqlen};
      }
    }
    params_.out_buffer = reinterpret_cast<T*>(out.ptr());
  }

  void SetWorkspace(size_t num_bytes) {
    void* ptr;
    HIP_CALL_THROW(hipMalloc(&ptr, num_bytes));
    workspace_.reset(ptr, [](void* ptr) { HIP_CALL_THROW(hipFree(ptr)); });
    params_.workspace_buffer = reinterpret_cast<T*>(workspace_.get());
  }

 protected:
  using ParamsT = contrib::rocm::GemmSoftmaxGemmPermuteParams<T>;
  rocblas_handle rocblas_handle_;
  contrib::AttentionParameters attn_;
  ParamsT params_;
  std::shared_ptr<void> workspace_;
};

template <typename T>
class CKGemmSoftmaxGemmPermute : public IGemmSoftmaxGemmPermuteKernelExplorer<T> {
 public:
  CKGemmSoftmaxGemmPermute(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t mask_dim,
      double scale,
      DeviceArray& Q,
      DeviceArray& K,
      DeviceArray& V,
      std::optional<DeviceArray>& attn_mask,
      DeviceArray& out)
      : IGemmSoftmaxGemmPermuteKernelExplorer<T>(batch, seqlen, total_seqlen, num_heads, head_size, mask_dim, scale,
                                                 Q, K, V, attn_mask, out) {
    this->SetWorkspace(GemmSoftmaxGemmPermuteTunableOp<T>::GetWorkspaceNumBytes(&this->attn_));

    // for (auto&& [ts, op] : GetCKGemmSoftmaxGemmPermuteTypeStringAndOps<T, /*USE_MASK=*/false, /*USE_BIAS=*/false>() ) {
    //   type_strings_.emplace_back(std::move(ts));
    //   ops_.emplace_back(std::move(op));
    // }

    for (auto&& [ts, op] : GetCKGemmSoftmaxGemmPermuteTypeStringAndOps<T, /*USE_MASK=*/true, /*USE_BIAS=*/false>()) {
      type_strings_.emplace_back(std::move(ts));
      ops_.emplace_back(std::move(op));
    }

    for (auto&& [ts, op] : GetCKGemmSoftmaxGemmPermuteTypeStringAndOps<T, /*USE_MASK=*/false, /*USE_BIAS=*/true>()) {
      type_strings_.emplace_back(std::move(ts));
      ops_.emplace_back(std::move(op));
    }

    // for (auto&& [ts, op] : GetCKGemmSoftmaxGemmPermuteTypeStringAndOps<T, /*USE_MASK=*/true, /*USE_BIAS=*/true>() ) {
    //   type_strings_.emplace_back(std::move(ts));
    //   ops_.emplace_back(std::move(op));
    // }
  }

  std::vector<std::string> ListOps() const {
    return type_strings_;
  }

  bool SelectOp(const std::string& name) {
    for (size_t i = 0; i < ops_.size(); i++) {
      if (type_strings_[i] == name) {
        selected_op_ = i;
        Status status = ops_[i](&this->params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&this->params_));
  }

 private:
  using ParamsT = typename IGemmSoftmaxGemmPermuteKernelExplorer<T>::ParamsT;
  using OpT = Op<ParamsT>;

  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};

#define REGISTER_OP(type)                                                          \
  py::class_<CKGemmSoftmaxGemmPermute<type>>(m, "CKGemmSoftmaxGemmPermute_" #type) \
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,          \
                    float,                                                         \
                    DeviceArray&,                                                  \
                    DeviceArray&,                                                  \
                    DeviceArray&,                                                  \
                    std::optional<DeviceArray>&,                                   \
                    DeviceArray&>())                                               \
      .def("SetRepeats", &CKGemmSoftmaxGemmPermute<type>::SetRepeats)              \
      .def("Run", &CKGemmSoftmaxGemmPermute<type>::Run)                            \
      .def("Profile", &CKGemmSoftmaxGemmPermute<type>::Profile)                    \
      .def("ListOps", &CKGemmSoftmaxGemmPermute<type>::ListOps)                    \
      .def("SelectOp", &CKGemmSoftmaxGemmPermute<type>::SelectOp);

void InitGemmSoftmaxGemmPermute(py::module m) {
  REGISTER_OP(float)
  REGISTER_OP(half)
}

}  // namespace onnxruntime
