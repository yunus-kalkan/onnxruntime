// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class TritonOp final : public OpKernel {
 public:
  TritonOp(const OpKernelInfo& info) : OpKernel(info) { ORT_THROW_IF_ERROR(info.GetAttr("func_name", &func_name_)); }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::string func_name_;
};

}  // namespace contrib
}  // namespace onnxruntime
