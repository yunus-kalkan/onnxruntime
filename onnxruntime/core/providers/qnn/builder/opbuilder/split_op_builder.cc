// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"

namespace onnxruntime {
namespace qnn {
class SplitOpBuilder : public BaseOpBuilder {
 public:
  SplitOpBuilder() : BaseOpBuilder("SplitOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SplitOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status SplitOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     std::vector<std::string>& input_names,
                                     bool do_op_validation) const {
  // Only support 1 input, Onnx Opset version < 11, or input 2 is initializer
  // doesn't support input 2 (split data) from dynamic input
  ORT_UNUSED_PARAMETER(do_op_validation);
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  InitializeQuantizeParam(quantize_param, is_quantized_model);
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;

  auto inputs = node_unit.Inputs();
  auto& input_name = inputs[0].node_arg.Name();

  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    input_names.push_back(input_name);
    return Status::OK();
  }

  const auto* type_proto = inputs[0].node_arg.TypeAsProto();
  ORT_RETURN_IF_ERROR(GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(inputs[0].quant_param,
                                                                   quantize_param.scaleOffsetEncoding.scale,
                                                                   quantize_param.scaleOffsetEncoding.offset),
                    "Cannot get quantization parameter");

  std::vector<uint8_t> unpacked_tensor;
  bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
  if (is_initializer_input) {
    const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*input_tensor, unpacked_tensor));
  }

  input_names.push_back(input_name);
  Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_name);

  QnnTensorWrapper input_tensorwrapper(input_name,
                                       tensor_type,
                                       qnn_data_type,
                                       quantize_param,
                                       std::move(input_shape),
                                       std::move(unpacked_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");

  return Status::OK();
}

Status SplitOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                   const NodeUnit& node_unit,
                                                   std::vector<std::string>&& input_names,
                                                   const logging::Logger& logger,
                                                   bool is_quantized_model,
                                                   bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  int32_t axis_value = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis_value));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), qnn_def::axis, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  std::vector<uint32_t> split_index;
  if (node_unit.Inputs().size() > 1) {
    auto& input_name = node_unit.Inputs()[1].node_arg.Name();
    bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
    if (is_initializer_input) {
      std::vector<uint8_t> unpacked_tensor;
      const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
      ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*input_tensor, unpacked_tensor));
      const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
      size_t tensor_byte_size = unpacked_tensor.size();
      size_t size = tensor_byte_size / sizeof(int64_t);
      split_index.push_back(0);  // QNN need the start index of each range and starts from 0
      std::transform(tensor_data, tensor_data + size, std::back_inserter(split_index),
                     [](int64_t item) { return SafeInt<uint32_t>(item); });
      split_index.pop_back();
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN doesn't support dynamic split");
    }
  } else {
    NodeAttrHelper node_helper(node_unit);
    if (node_helper.HasAttr("split")) {
      auto split = node_helper.Get("split", std::vector<int32_t>{0});
      uint32_t split_it = 0;
      for (size_t i = 0; i < split.size(); ++i) {
        split_index.push_back(split_it);
        split_it += split[i];
      }
    }
  }

  // Get the length according to axis and split it equally
  if (split_index.size() == 0) {
    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape),
                      "Cannot get shape");
    ORT_ENFORCE(static_cast<int32_t>(input_shape.size()) > axis_value, "axis not valid!");
    ORT_RETURN_IF_NOT(input_shape.at(axis_value) > 0, "Shape value not valid!");
    auto num_outputs = node_unit.Outputs().size();
    auto step = SafeInt<uint32_t>(input_shape.at(axis_value) / num_outputs);
    uint32_t split_it = 0;
    for (size_t i = 0; i < num_outputs; ++i) {
      split_index.push_back(split_it);
      split_it += step;
    }
  }

  uint32_t split_size = static_cast<uint32_t>(split_index.size());
  std::vector<uint32_t> split_dim{split_size};
  QnnParamWrapper split_param(node_unit.Index(), node_unit.Name(), qnn_def::split_index, std::move(split_dim),
                              std::move(split_index));
  param_tensor_names.push_back(split_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(split_param));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, is_quantized_model, do_op_validation));

  return Status::OK();
}

void CreateSplitOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SplitOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
