// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/triton/triton_op.h"

#include "core/framework/op_kernel_context_internal.h"
#include "orttraining/core/framework/torch/dlpack_python.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(TritonOp, kMSDomain, 1, kCpuExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        TritonOp);

Status TritonOp::Compute(OpKernelContext* context) const {
  auto* p_ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  size_t input_size = static_cast<size_t>(p_ctx_internal->InputCount());
  size_t output_size = static_cast<size_t>(p_ctx_internal->OutputCount());

  // Support only device tensor for now.
  // TODO: refactor to avoid mem leak.
  PyObject* p_module_name = PyUnicode_FromString("onnxruntime.training.ortmodule.triton");
  ORT_ENFORCE(p_module_name, "PyUnicode_FromString failed with module name.");
  PyObject* p_module = PyImport_Import(p_module_name);
  ORT_ENFORCE(p_module, "ORTModule Triton module is failed to import.");
  PyObject* p_func = PyObject_GetAttrString(p_module, func_name_.c_str());
  ORT_ENFORCE(p_func, "Function ", func_name_, " is not found in ORTModule Triton module.");

  PyObject* p_args = PyTuple_New(input_size);
  ORT_ENFORCE(p_args, "Failed to create input tuple with size ", input_size, ".");
  for (size_t i = 0; i < input_size; ++i) {
    PyObject* p_value = training::framework::torch::ToDlpack(*p_ctx_internal->GetInputMLValue(i));
    // p_value reference stolen here.
    PyTuple_SetItem(p_args, i, p_value);
  }

  // TODO: bool tensor not supported as output for now.
  PyObject* p_ret = PyObject_CallObject(p_func, p_args);
  if (PyTuple_Check(p_ret)) {
    ORT_ENFORCE(static_cast<size_t>(PyTuple_Size(p_ret)) == output_size, "Output size mismatch.");
    for (size_t i = 0; i < output_size; ++i) {
      ORT_THROW_IF_ERROR(p_ctx_internal->SetOutputMLValue(
          i, training::framework::torch::FromDlpack(PyTuple_GetItem(p_ret, i), false)));
    }
  } else {
    ORT_ENFORCE(output_size == 1, "Output size mismatch.");
    ORT_THROW_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, training::framework::torch::FromDlpack(p_ret, false)));
  }

  Py_DECREF(p_ret);
  Py_DECREF(p_args);
  Py_DECREF(p_func);
  Py_DECREF(p_module);
  Py_DECREF(p_module_name);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
