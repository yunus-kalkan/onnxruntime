// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env_var_utils.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "test/contrib_ops/attention_op_test_helper.h"

namespace onnxruntime {
using contrib::AttentionMaskType;
namespace test {

static void RunCrossAttentionTest(
    const std::vector<float>& query_data,               // query:  [batch_size, sequence_length, hidden_size]
    const std::vector<float>& key_data,                 // key:    [batch_size, kv_sequence_length, hidden_size]
    const std::vector<float>& value_data,               // value:  [batch_size, kv_sequence_length, v_hidden_size]
    const std::vector<float>& bias_data,                // bias:   [hidden_size + hidden_size + v_hidden_size]
    const std::vector<int32_t>& key_padding_mask_data,  // key_padding_mask: see below
    AttentionMaskType mask_type,                        // 1 for [batch_size], 2 for [batch_size, kv_sequence_length]
    const std::vector<float>& output_data,              // output: [batch_size, sequence_length, v_hidden_size]
    int number_of_heads,
    int batch_size,
    int sequence_length,
    int kv_sequence_length,
    int hidden_size,
    int v_hidden_size,
    bool use_float16 = false,
    bool disable_cpu = true,  // not supported in cpu right now.
    bool disable_cuda = false,
    bool disable_rocm = true)  // not supported in rocm right now.
{
  kv_sequence_length = (kv_sequence_length == 0 ? sequence_length : kv_sequence_length);

  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !disable_cuda;
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !disable_rocm;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;

  if (enable_cpu || enable_cuda || enable_rocm) {
    OpTester tester("CrossAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));

    std::vector<int64_t> query_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> key_dims = {batch_size, kv_sequence_length, hidden_size};
    std::vector<int64_t> value_dims = {batch_size, kv_sequence_length, v_hidden_size};
    std::vector<int64_t> bias_dims = {hidden_size + hidden_size + v_hidden_size};
    std::vector<int64_t> output_dims = {batch_size, sequence_length, v_hidden_size};

    std::vector<int64_t> mask_dims_1 = {batch_size};
    std::vector<int64_t> mask_dims_2 = {batch_size, kv_sequence_length};
    std::vector<int64_t>& key_padding_mask_dims = (mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN)
                                                      ? mask_dims_1
                                                      : mask_dims_2;

    if (use_float16) {
      tester.AddInput<MLFloat16>("query", query_dims, ToFloat16(query_data));
      tester.AddInput<MLFloat16>("key", key_dims, ToFloat16(key_data));
      tester.AddInput<MLFloat16>("value", value_dims, ToFloat16(value_data));

      if (bias_data.size()) {
        tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }

      if (key_padding_mask_data.size()) {
        tester.AddInput<int32_t>("key_padding_mask", key_padding_mask_dims, key_padding_mask_data);
      } else {
        tester.AddOptionalInputEdge<int32_t>();
      }

      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("query", query_dims, query_data);
      tester.AddInput<float>("key", key_dims, key_data);
      tester.AddInput<float>("value", value_dims, value_data);

      if (bias_data.size()) {
        tester.AddInput<float>("bias", bias_dims, bias_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }

      if (key_padding_mask_data.size()) {
        tester.AddInput<int32_t>("key_padding_mask", key_padding_mask_dims, key_padding_mask_data);
      } else {
        tester.AddOptionalInputEdge<int32_t>();
      }

      tester.AddOutput<float>("output", output_dims, output_data);
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_rocm) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultRocmExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

static void RunCrossAttentionTestEnv(
    const std::vector<float>& query_data,               // query:  [batch_size, sequence_length, hidden_size]
    const std::vector<float>& key_data,                 // key:    [batch_size, kv_sequence_length, hidden_size]
    const std::vector<float>& value_data,               // value:  [batch_size, kv_sequence_length, v_hidden_size]
    const std::vector<float>& bias_data,                // bias:   [hidden_size + hidden_size + v_hidden_size]
    const std::vector<int32_t>& key_padding_mask_data,  // key_padding_mask: see below
    AttentionMaskType mask_type,                        // 1 for [batch_size], 2 for [batch_size, kv_sequence_length]
    const std::vector<float>& output_data,              // output: [batch_size, sequence_length, v_hidden_size]
    int number_of_heads,
    int batch_size,
    int sequence_length,
    int kv_sequence_length,
    int hidden_size,
    int v_hidden_size,
    bool use_float16 = false,
    bool disable_cpu = true,  // not supported in cpu right now.
    bool disable_cuda = false,
    bool disable_rocm = true) {
  // Unfused kernel
  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kEnableFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "1"}}};
    RunCrossAttentionTest(
        query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
        number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
        use_float16, disable_cpu, disable_cuda, disable_rocm);
  }

  // // Fused kernel (enable flash attention)
  // {
  //   ScopedEnvironmentVariables scoped_env_vars{
  //       EnvVarMap{
  //           {onnxruntime::contrib::attention::kEnableFlashAttention, "0"},
  //           {onnxruntime::contrib::attention::kDisableFusedAttention, "0"}}};
  //   RunCrossAttentionTest(
  //       query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
  //       number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
  //       use_float16, disable_cpu, disable_cuda, disable_rocm);
  // }

  // // Fused kernel (disable flash attention)
  // {
  //   ScopedEnvironmentVariables scoped_env_vars{
  //       EnvVarMap{
  //           {onnxruntime::contrib::attention::kEnableFlashAttention, "1"},
  //           {onnxruntime::contrib::attention::kDisableFusedAttention, "0"}}};
  //   RunCrossAttentionTest(
  //       query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
  //       number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
  //       use_float16, disable_cpu, disable_cuda, disable_rocm);
  // }
}

TEST(CrossAttentionTest, CrossAttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;
  int kv_sequence_length = 3;
  int v_hidden_size = 2;

  std::vector<float> query_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> key_data = {0.1f, 0.2f, 0.3f, 0.4f,
                                 0.5f, 0.6f, 0.7f, 0.8f,
                                 0.9f, 1.0f, 1.1f, 1.2f};

  std::vector<float> value_data = {0.6f, 0.5f,
                                   0.4f, 0.3f,
                                   0.2f, 0.1f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f,
      0.5f, 0.7f, 0.2f, 1.2f,
      0.5f, 0.4f};

  std::vector<float> output_data = {0.99434918f, 0.0f,
                                    0.9887343f, 0.74572039f};

  std::vector<int32_t> key_padding_mask_data = {2L};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN;

  bool use_float16 = false;

  RunCrossAttentionTest(
      query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
      number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
      use_float16);
}

#ifndef ENABLE_TRAINING  // TRT fused attention is enabled only on non-training builds
TEST(CrossAttentionTest, CrossAttention_NoMask) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 2;
  constexpr int kv_sequence_length = 4;
  constexpr int hidden_size = 64;
  constexpr int v_hidden_size = 64;
  constexpr int number_of_heads = 2;

  std::vector<float> query_data = {
      0.00838f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.005913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.006706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00301f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.001682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.00744f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00738f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.002193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      -0.004917f, 0.003468f, -0.0013685f, -0.007122f, 0.00788f, 0.000825f, 0.00621f, -0.00437f, 0.005653f, 0.009674f,
      0.003576f, 0.00956f, 0.0064f, 0.00283f, -0.00797f, 0.00867f, 0.004536f, -0.00985f, 0.004856f, -0.006878f,
      0.006012f, -0.0042f, -0.00328f, -0.00885f, -0.0079f, 0.004917f, -0.00594f, 0.003452f, -0.006355f, -0.003536f,
      0.0022f, 0.003494f, -0.008865f, 0.00461f, -0.00485f, 0.00889f, -0.002272f, 0.00596f};

  std::vector<float> key_data;
  GetAttentionWeight(key_data, kv_sequence_length * hidden_size);

  std::vector<float> value_data = {
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.002913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.005706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00201f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.003682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.00544f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00838f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.007193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      -0.003917f, 0.003468f, -0.0013685f, -0.007122f, 0.00788f, 0.000825f, 0.00621f, -0.00437f, 0.005653f, 0.009674f,
      0.00576f, 0.00956f, 0.0064f, 0.00283f, -0.00797f, 0.00867f, 0.004536f, -0.00985f, 0.004856f, -0.006878f,
      0.008012f, -0.0042f, -0.00328f, -0.00885f, -0.0079f, 0.004917f, -0.00594f, 0.003452f, -0.006355f, -0.003536f,
      0.00638f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.0012f, 0.003494f, -0.008865f, 0.00461f, -0.00485f, 0.00889f, -0.002272f, 0.00596f,
      0.00544f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00838f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.007193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      -0.003917f, 0.003468f, -0.0013685f, -0.007122f, 0.00788f, 0.000825f, 0.00621f, -0.00437f, 0.005653f, 0.009674f,
      0.00576f, 0.00956f, 0.0064f, 0.00283f, -0.00797f, 0.00867f, 0.004536f, -0.00985f, 0.004856f, -0.006878f,
      0.008012f, -0.0042f, -0.00328f, -0.00885f, -0.0079f, 0.004917f, -0.00594f, 0.003452f, -0.006355f, -0.003536f,
      0.00638f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.002913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.005706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00201f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.003682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.0012f, 0.003494f, -0.008865f, 0.00461f, -0.00485f, 0.00889f, -0.002272f, 0.00596f};

  std::vector<float> bias_data;
  GetAttentionBias(value_data, 2 * hidden_size + v_hidden_size);

  // No mask_index
  std::vector<int32_t> key_padding_mask_data = {};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_NONE;

  std::vector<float> output_data = {
      0.0027942657f, 0.0067901611f, 0.0070953369f, -0.0020713806f, 0.0055351257f, 0.0030479431f, -0.0060462952f,
      -0.0087127686f, 0.0030956268f, -0.00036644936f, 0.0014686584f, -0.0038146973f, 0.0072097778f, -0.0052490234f,
      0.0056114197f, 0.0050926208f, 0.0080947876f, 0.0074501038f, 0.0079498291f, 0.0098876953f, -0.0066146851f,
      0.0064735413f, 0.0093307495f, -0.00051593781f, -0.0047683716f, -0.0069198608f, 0.0094604492f, 0.0066146851f,
      -0.0040054321f, 0.0017976761f, -0.0058059692f, -0.0087051392f, 0.0054740906f, 0.0022010803f, 0.0075340271f,
      0.0047035217f, 0.00340271f, 0.0096969604f, -0.0016756058f, 0.0020771027f, -0.0063018799f, 0.0073280334f,
      -0.0056381226f, 0.004032135f, -0.0082473755f, 0.0045280457f, 0.0045814514f, -0.0026607513f, -0.0031585693f,
      -0.003660202f, -0.0053253174f, -0.0089187622f, -0.0073509216f, 0.0048408508f, 0.0058364868f, 0.0069313049f,
      -0.0071868896f, 0.008392334f, -0.0018663406f, -0.0092163086f, -0.00048780441f, -0.0054283142f, -0.0061683655f,
      0.0078048706f, 0.0025291443f, 0.0065917969f, 0.0072250366f, -0.0018520355f, 0.005531311f, 0.003118515f,
      -0.0061264038f, -0.0090484619f, 0.003276825f, -0.00047063828f, 0.0015802383f, -0.0037345886f, 0.0069732666f,
      -0.0054092407f, 0.0052947998f, 0.004940033f, 0.0085220337f, 0.007194519f, 0.0078659058f, 0.0095214844f,
      -0.0065574646f, 0.0064315796f, 0.0093383789f, -0.00058555603f, -0.0046386719f, -0.0067710876f, 0.0096130371f,
      0.0064315796f, -0.0040740967f, 0.0017337799f, -0.0057067871f, -0.008682251f, 0.0054855347f, 0.0019645691f,
      0.0075149536f, 0.0047187805f, 0.0036354065f, 0.0096282959f, -0.0019168854f, 0.0021934509f, -0.0063018799f,
      0.0072937012f, -0.006187439f, 0.0039825439f, -0.0081253052f, 0.0046577454f, 0.0045700073f, -0.0028266907f,
      -0.0028438568f, -0.0035438538f, -0.0053100586f, -0.0090332031f, -0.0071105957f, 0.004699707f, 0.0058021545f,
      0.0071411133f, -0.0071678162f, 0.0085449219f, -0.0018749237f, -0.0095825195f, -0.00049686432f, -0.0053634644f,
      -0.0057945251f, 0.0078277588f};

  bool use_float16 = true;

  RunCrossAttentionTestEnv(
      query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
      number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
      use_float16);
}

TEST(CrossAttentionTest, CrossAttention_Mask1D) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 2;
  constexpr int kv_sequence_length = 4;
  constexpr int hidden_size = 64;
  constexpr int v_hidden_size = 64;
  constexpr int number_of_heads = 2;

  std::vector<float> query_data = {
      0.00838f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.005913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.006706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00301f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.001682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.00744f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00738f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.002193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      -0.004917f, 0.003468f, -0.0013685f, -0.007122f, 0.00788f, 0.000825f, 0.00621f, -0.00437f, 0.005653f, 0.009674f,
      0.003576f, 0.00956f, 0.0064f, 0.00283f, -0.00797f, 0.00867f, 0.004536f, -0.00985f, 0.004856f, -0.006878f,
      0.006012f, -0.0042f, -0.00328f, -0.00885f, -0.0079f, 0.004917f, -0.00594f, 0.003452f, -0.006355f, -0.003536f,
      0.0022f, 0.003494f, -0.008865f, 0.00461f, -0.00485f, 0.00889f, -0.002272f, 0.00596f};

  std::vector<float> key_data;
  GetAttentionWeight(key_data, kv_sequence_length * hidden_size);

  std::vector<float> value_data = {
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.002913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.005706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00201f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.003682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.00544f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00838f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.007193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      -0.003917f, 0.003468f, -0.0013685f, -0.007122f, 0.00788f, 0.000825f, 0.00621f, -0.00437f, 0.005653f, 0.009674f,
      0.00576f, 0.00956f, 0.0064f, 0.00283f, -0.00797f, 0.00867f, 0.004536f, -0.00985f, 0.004856f, -0.006878f,
      0.008012f, -0.0042f, -0.00328f, -0.00885f, -0.0079f, 0.004917f, -0.00594f, 0.003452f, -0.006355f, -0.003536f,
      0.00638f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.0012f, 0.003494f, -0.008865f, 0.00461f, -0.00485f, 0.00889f, -0.002272f, 0.00596f,
      0.00544f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00838f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.007193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      -0.003917f, 0.003468f, -0.0013685f, -0.007122f, 0.00788f, 0.000825f, 0.00621f, -0.00437f, 0.005653f, 0.009674f,
      0.00576f, 0.00956f, 0.0064f, 0.00283f, -0.00797f, 0.00867f, 0.004536f, -0.00985f, 0.004856f, -0.006878f,
      0.008012f, -0.0042f, -0.00328f, -0.00885f, -0.0079f, 0.004917f, -0.00594f, 0.003452f, -0.006355f, -0.003536f,
      0.00638f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.002913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.005706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00201f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.003682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.0012f, 0.003494f, -0.008865f, 0.00461f, -0.00485f, 0.00889f, -0.002272f, 0.00596f};

  std::vector<float> bias_data;
  GetAttentionBias(value_data, 2 * hidden_size + v_hidden_size);

  // No mask_index
  std::vector<int32_t> key_padding_mask_data = {3};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN;

  std::vector<float> output_data = {
      0.0027942657f, 0.0067901611f, 0.0070953369f, -0.0020713806f, 0.0055351257f, 0.0030479431f, -0.0060462952f,
      -0.0087127686f, 0.0030956268f, -0.00036644936f, 0.0014686584f, -0.0038146973f, 0.0072097778f, -0.0052490234f,
      0.0056114197f, 0.0050926208f, 0.0080947876f, 0.0074501038f, 0.0079498291f, 0.0098876953f, -0.0066146851f,
      0.0064735413f, 0.0093307495f, -0.00051593781f, -0.0047683716f, -0.0069198608f, 0.0094604492f, 0.0066146851f,
      -0.0040054321f, 0.0017976761f, -0.0058059692f, -0.0087051392f, 0.0054740906f, 0.0022010803f, 0.0075340271f,
      0.0047035217f, 0.00340271f, 0.0096969604f, -0.0016756058f, 0.0020771027f, -0.0063018799f, 0.0073280334f,
      -0.0056381226f, 0.004032135f, -0.0082473755f, 0.0045280457f, 0.0045814514f, -0.0026607513f, -0.0031585693f,
      -0.003660202f, -0.0053253174f, -0.0089187622f, -0.0073509216f, 0.0048408508f, 0.0058364868f, 0.0069313049f,
      -0.0071868896f, 0.008392334f, -0.0018663406f, -0.0092163086f, -0.00048780441f, -0.0054283142f, -0.0061683655f,
      0.0078048706f, 0.0025291443f, 0.0065917969f, 0.0072250366f, -0.0018520355f, 0.005531311f, 0.003118515f,
      -0.0061264038f, -0.0090484619f, 0.003276825f, -0.00047063828f, 0.0015802383f, -0.0037345886f, 0.0069732666f,
      -0.0054092407f, 0.0052947998f, 0.004940033f, 0.0085220337f, 0.007194519f, 0.0078659058f, 0.0095214844f,
      -0.0065574646f, 0.0064315796f, 0.0093383789f, -0.00058555603f, -0.0046386719f, -0.0067710876f, 0.0096130371f,
      0.0064315796f, -0.0040740967f, 0.0017337799f, -0.0057067871f, -0.008682251f, 0.0054855347f, 0.0019645691f,
      0.0075149536f, 0.0047187805f, 0.0036354065f, 0.0096282959f, -0.0019168854f, 0.0021934509f, -0.0063018799f,
      0.0072937012f, -0.006187439f, 0.0039825439f, -0.0081253052f, 0.0046577454f, 0.0045700073f, -0.0028266907f,
      -0.0028438568f, -0.0035438538f, -0.0053100586f, -0.0090332031f, -0.0071105957f, 0.004699707f, 0.0058021545f,
      0.0071411133f, -0.0071678162f, 0.0085449219f, -0.0018749237f, -0.0095825195f, -0.00049686432f, -0.0053634644f,
      -0.0057945251f, 0.0078277588f};

  bool use_float16 = true;

  RunCrossAttentionTestEnv(
      query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
      number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
      use_float16);
}

TEST(CrossAttentionTest, CrossAttention_Mask2D) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 2;
  constexpr int kv_sequence_length = 4;
  constexpr int hidden_size = 40;
  constexpr int v_hidden_size = 40;
  constexpr int number_of_heads = 2;

  std::vector<float> query_data = {
      0.00838f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.005913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.006706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00301f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.001682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.00744f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00738f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f};

  std::vector<float> key_data;
  GetAttentionWeight(key_data, kv_sequence_length * hidden_size);

  std::vector<float> value_data = {
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.002913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.005706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00201f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.003682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.00544f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00838f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.007193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      0.00544f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00838f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.007193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      -0.003917f, 0.003468f, -0.0013685f, -0.007122f, 0.00788f, 0.000825f, 0.00621f, -0.00437f, 0.005653f, 0.009674f,
      0.00576f, 0.00956f, 0.0064f, 0.00283f, -0.00797f, 0.00867f, 0.004536f, -0.00985f, 0.004856f, -0.006878f,
      0.008012f, -0.0042f, -0.00328f, -0.00885f, -0.0079f, 0.004917f, -0.00594f, 0.003452f, -0.006355f, -0.003536f,
      0.00638f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f};

  std::vector<float> bias_data;
  GetAttentionBias(value_data, 2 * hidden_size + v_hidden_size);

  // No mask_index
  std::vector<int32_t> key_padding_mask_data = {1, 1, 1, 0};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;

  std::vector<float> output_data = {
      0.0027942657f, 0.0067901611f, 0.0070953369f, -0.0020713806f, 0.0055351257f, 0.0030479431f, -0.0060462952f,
      -0.0087127686f, 0.0030956268f, -0.00036644936f, 0.0014686584f, -0.0038146973f, 0.0072097778f, -0.0052490234f,
      0.0056114197f, 0.0050926208f, 0.0080947876f, 0.0074501038f, 0.0079498291f, 0.0098876953f, -0.0066146851f,
      0.0064735413f, 0.0093307495f, -0.00051593781f, -0.0047683716f, -0.0069198608f, 0.0094604492f, 0.0066146851f,
      -0.0040054321f, 0.0017976761f, -0.0058059692f, -0.0087051392f, 0.0054740906f, 0.0022010803f, 0.0075340271f,
      0.0047035217f, 0.00340271f, 0.0096969604f, -0.0016756058f, 0.0020771027f, -0.0063018799f, 0.0073280334f,
      -0.0056381226f, 0.004032135f, -0.0082473755f, 0.0045280457f, 0.0045814514f, -0.0026607513f, -0.0031585693f,
      -0.003660202f, -0.0053253174f, -0.0089187622f, -0.0073509216f, 0.0048408508f, 0.0058364868f, 0.0069313049f,
      -0.0071868896f, 0.008392334f, -0.0018663406f, -0.0092163086f, -0.00048780441f, -0.0054283142f, -0.0061683655f,
      0.0078048706f, 0.0025291443f, 0.0065917969f, 0.0072250366f, -0.0018520355f, 0.005531311f, 0.003118515f,
      -0.0061264038f, -0.0090484619f, 0.003276825f, -0.00047063828f, 0.0015802383f, -0.0037345886f, 0.0069732666f,
      -0.0054092407f, 0.0052947998f, 0.004940033f, 0.0085220337f, 0.007194519f, 0.0078659058f, 0.0095214844f,
      -0.0065574646f, 0.0064315796f, 0.0093383789f, -0.00058555603f, -0.0046386719f, -0.0067710876f, 0.0096130371f,
      0.0064315796f, -0.0040740967f, 0.0017337799f, -0.0057067871f, -0.008682251f, 0.0054855347f, 0.0019645691f,
      0.0075149536f, 0.0047187805f, 0.0036354065f, 0.0096282959f, -0.0019168854f, 0.0021934509f, -0.0063018799f,
      0.0072937012f, -0.006187439f, 0.0039825439f, -0.0081253052f, 0.0046577454f, 0.0045700073f, -0.0028266907f,
      -0.0028438568f, -0.0035438538f, -0.0053100586f, -0.0090332031f, -0.0071105957f, 0.004699707f, 0.0058021545f,
      0.0071411133f, -0.0071678162f, 0.0085449219f, -0.0018749237f, -0.0095825195f, -0.00049686432f, -0.0053634644f,
      -0.0057945251f, 0.0078277588f};

  bool use_float16 = true;
  RunCrossAttentionTestEnv(
      query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
      number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
      use_float16);
}

TEST(CrossAttentionTest, CrossAttention_HiddenSizeV) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 2;
  constexpr int kv_sequence_length = 4;
  constexpr int hidden_size = 40;
  constexpr int v_hidden_size = 20;
  constexpr int number_of_heads = 2;

  std::vector<float> query_data = {
      0.00838f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.005913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.006706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00301f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.001682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.00744f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00738f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f};

  std::vector<float> key_data;
  GetAttentionWeight(key_data, kv_sequence_length * hidden_size);

  std::vector<float> value_data = {
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      -0.005706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      0.003682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      -0.00838f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.00544f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      0.007193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      0.00576f, 0.00956f, 0.0064f, 0.00283f, -0.00797f, 0.00867f, 0.004536f, -0.00985f, 0.004856f, -0.006878f,
      0.00638f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f};

  std::vector<float> bias_data;
  GetAttentionBias(value_data, 2 * hidden_size + v_hidden_size);

  // No mask_index
  std::vector<int32_t> key_padding_mask_data = {1, 0, 0, 1};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;

  std::vector<float> output_data = {
      0.0027942657f, 0.0067901611f, 0.0070953369f, -0.0020713806f, 0.0055351257f, 0.0030479431f, -0.0060462952f,
      -0.0087127686f, 0.0030956268f, -0.00036644936f, 0.0014686584f, -0.0038146973f, 0.0072097778f, -0.0052490234f,
      0.0056114197f, 0.0050926208f, 0.0080947876f, 0.0074501038f, 0.0079498291f, 0.0098876953f, -0.0066146851f,
      0.0064735413f, 0.0093307495f, -0.00051593781f, -0.0047683716f, -0.0069198608f, 0.0094604492f, 0.0066146851f,
      -0.0040054321f, 0.0017976761f, -0.0058059692f, -0.0087051392f, 0.0054740906f, 0.0022010803f, 0.0075340271f,
      0.0047035217f, 0.00340271f, 0.0096969604f, -0.0016756058f, 0.0020771027f, -0.0063018799f, 0.0073280334f,
      -0.0056381226f, 0.004032135f, -0.0082473755f, 0.0045280457f, 0.0045814514f, -0.0026607513f, -0.0031585693f,
      -0.003660202f, -0.0053253174f, -0.0089187622f, -0.0073509216f, 0.0048408508f, 0.0058364868f, 0.0069313049f,
      -0.0071868896f, 0.008392334f, -0.0018663406f, -0.0092163086f, -0.00048780441f, -0.0054283142f, -0.0061683655f,
      0.0078048706f, 0.0025291443f, 0.0065917969f, 0.0072250366f, -0.0018520355f, 0.005531311f, 0.003118515f,
      -0.0061264038f, -0.0090484619f, 0.003276825f, -0.00047063828f, 0.0015802383f, -0.0037345886f, 0.0069732666f,
      -0.0054092407f, 0.0052947998f, 0.004940033f, 0.0085220337f, 0.007194519f, 0.0078659058f, 0.0095214844f,
      -0.0065574646f, 0.0064315796f, 0.0093383789f, -0.00058555603f, -0.0046386719f, -0.0067710876f, 0.0096130371f,
      0.0064315796f, -0.0040740967f, 0.0017337799f, -0.0057067871f, -0.008682251f, 0.0054855347f, 0.0019645691f,
      0.0075149536f, 0.0047187805f, 0.0036354065f, 0.0096282959f, -0.0019168854f, 0.0021934509f, -0.0063018799f,
      0.0072937012f, -0.006187439f, 0.0039825439f, -0.0081253052f, 0.0046577454f, 0.0045700073f, -0.0028266907f,
      -0.0028438568f, -0.0035438538f, -0.0053100586f, -0.0090332031f, -0.0071105957f, 0.004699707f, 0.0058021545f,
      0.0071411133f, -0.0071678162f, 0.0085449219f, -0.0018749237f, -0.0095825195f, -0.00049686432f, -0.0053634644f,
      -0.0057945251f, 0.0078277588f};

  bool use_float16 = true;
  RunCrossAttentionTestEnv(
      query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
      number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
      use_float16);
}
#endif

}  // namespace test
}  // namespace onnxruntime
