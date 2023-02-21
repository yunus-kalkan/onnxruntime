// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime::contrib {

template <typename T>
class LSTMGrad final : public OpKernel {
 public:
  LSTMGrad(const OpKernelInfo& info) : OpKernel(info) {
    std::string direction;
    ORT_ENFORCE(info.GetAttr("direction", &direction).IsOK());
    direction_ = rnn::detail::MakeDirection(direction);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  rnn::detail::Direction direction_;
};

}  // namespace onnxruntime::contrib
