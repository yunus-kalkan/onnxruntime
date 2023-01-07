// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <vector>

namespace onnxruntime {
namespace test {

#ifndef ENABLE_TRAINING  // TRT fused attention is enabled only on non-training builds

// Return packed weights and bias for input projection.
void GetAttentionWeight(std::vector<float>& weight_data, size_t elements = 64 * 3 * 64, size_t offset = 0, size_t step=1);
void GetAttentionBias(std::vector<float>& bias_data, size_t elements = 3 * 64, size_t offset = 0, size_t step=1);
#endif

}  // namespace test
}  // namespace onnxruntime
