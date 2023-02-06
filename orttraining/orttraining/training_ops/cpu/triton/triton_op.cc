// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/triton/triton_op.h"

#include "core/framework/op_kernel_context_internal.h"
#include "orttraining/core/framework/torch/dlpack_python.h"
#include "orttraining/core/framework/torch/torch_proxy.h"

namespace onnxruntime {
namespace contrib {

using PythonObjectPtr = language_interop_ops::torch::PythonObjectPtr;
constexpr auto PythonObjectDeleter = language_interop_ops::torch::PythonObjectDeleter;
constexpr auto ToDlpack = training::framework::torch::ToDlpack;
constexpr auto FromDlpack = training::framework::torch::FromDlpack;

ONNX_OPERATOR_KERNEL_EX(TritonOp, kMSDomain, 1, kCpuExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        TritonOp);

Status TritonOp::Compute(OpKernelContext* context) const {
  auto* p_ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  size_t input_size = static_cast<size_t>(p_ctx_internal->InputCount());
  size_t output_size = static_cast<size_t>(p_ctx_internal->OutputCount());

  // Support only device tensor for now.
  PythonObjectPtr triton_module_name(PyUnicode_FromString("onnxruntime.training.ortmodule.triton"),
                                     PythonObjectDeleter);
  PythonObjectPtr triton_module(PyImport_Import(triton_module_name.get()), PythonObjectDeleter);
  ORT_ENFORCE(triton_module, "ORTModule Triton module is failed to import.");
  PythonObjectPtr op_func(PyObject_GetAttrString(triton_module.get(), func_name_.c_str()), PythonObjectDeleter);
  ORT_ENFORCE(op_func, "Function ", func_name_, " is not found in ORTModule Triton module.");

  PythonObjectPtr args(PyTuple_New(input_size), PythonObjectDeleter);
  ORT_ENFORCE(args, "Failed to create input tuple with size ", input_size, ".");
  for (size_t i = 0; i < input_size; ++i) {
    PyObject* p_value = ToDlpack(*p_ctx_internal->GetInputMLValue(i));
    // p_value reference stolen here.
    PyTuple_SetItem(args.get(), i, p_value);
  }

  // TODO: bool tensor not supported as output for now.
  PythonObjectPtr ret(PyObject_CallObject(op_func.get(), args.get()), PythonObjectDeleter);
  if (PyTuple_Check(ret.get())) {
    ORT_ENFORCE(static_cast<size_t>(PyTuple_Size(ret.get())) == output_size, "Output size mismatch.");
    for (size_t i = 0; i < output_size; ++i) {
      ORT_THROW_IF_ERROR(p_ctx_internal->SetOutputMLValue(i, FromDlpack(PyTuple_GetItem(ret.get(), i), false)));
    }
  } else {
    ORT_ENFORCE(output_size == 1, "Output size mismatch.");
    ORT_THROW_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, FromDlpack(ret.get(), false)));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
