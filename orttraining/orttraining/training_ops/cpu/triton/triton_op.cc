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

bool TritonOp::IsBoolOutput(size_t index) const {
  ORT_ENFORCE(index < Node().OutputDefs().size(), "Output index out of range.");
  return Node().OutputDefs()[index]->TypeAsProto()->tensor_type().elem_type() ==
         ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL;
}

Status TritonOp::Compute(OpKernelContext* context) const {
  auto* p_ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  size_t input_size = static_cast<size_t>(p_ctx_internal->InputCount());
  size_t output_size = static_cast<size_t>(p_ctx_internal->OutputCount());

  // TODO: add PyDict to pass all attributes as kwargs.
  PythonObjectPtr triton_module_name(PyUnicode_FromString("onnxruntime.training.ortmodule.triton"),
                                     PythonObjectDeleter);
  PythonObjectPtr triton_module(PyImport_Import(triton_module_name.get()), PythonObjectDeleter);
  ORT_ENFORCE(triton_module, "ORTModule Triton module is failed to import.");
  PythonObjectPtr op_func(PyObject_GetAttrString(triton_module.get(), func_name_.c_str()), PythonObjectDeleter);
  ORT_ENFORCE(op_func, "Function ", func_name_, " is not found in ORTModule Triton module.");

  PythonObjectPtr args(PyTuple_New(static_cast<Py_ssize_t>(input_size)), PythonObjectDeleter);
  ORT_ENFORCE(args, "Failed to create input tuple with size ", input_size, ".");
  for (size_t i = 0; i < input_size; ++i) {
    PyObject* p_value = ToDlpack(*p_ctx_internal->GetInputMLValue(static_cast<int>(i)));
    // p_value reference stolen here.
    PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(i), p_value);
  }

  PythonObjectPtr ret(PyObject_CallObject(op_func.get(), args.get()), PythonObjectDeleter);
  if (PyTuple_Check(ret.get())) {
    ORT_ENFORCE(static_cast<size_t>(PyTuple_Size(ret.get())) == output_size, "Output size mismatch.");
    for (size_t i = 0; i < output_size; ++i) {
      ORT_THROW_IF_ERROR(p_ctx_internal->SetOutputMLValue(
          static_cast<int>(i), FromDlpack(PyTuple_GetItem(ret.get(), static_cast<Py_ssize_t>(i)), IsBoolOutput(i))));
    }
  } else {
    ORT_ENFORCE(output_size == 1, "Output size mismatch.");
    ORT_THROW_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, FromDlpack(ret.get(), IsBoolOutput(0))));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
