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

  // Fused kernel (enable flash attention)
  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kEnableFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "0"}}};
    RunCrossAttentionTest(
        query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
        number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
        use_float16, disable_cpu, disable_cuda, disable_rocm);
  }

  // Fused kernel (disable flash attention)
  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kEnableFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "0"}}};
    RunCrossAttentionTest(
        query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
        number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
        use_float16, disable_cpu, disable_cuda, disable_rocm);
  }
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
  GetAttentionWeight(key_data, batch_size * kv_sequence_length * hidden_size);

  std::vector<float> value_data;
  GetAttentionWeight(value_data, batch_size * kv_sequence_length * v_hidden_size);

  std::vector<float> bias_data;
  GetAttentionBias(bias_data, 2 * hidden_size + v_hidden_size);

  // No mask_index
  std::vector<int32_t> key_padding_mask_data = {};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_NONE;

  std::vector<float> output_data = {
      0.0021858215f, 0.00084114075f, 0.0070838928f, 0.002576828f, 0.0029411316f, 0.0053520203f, -0.0054779053f,
      -0.0084915161f, 0.004776001f, -0.0020275116f, 0.0030059814f, -0.00096559525f, 0.010726929f, -0.0082473755f,
      0.008430481f, 0.0028343201f, 0.011825562f, 0.0033950806f, 0.0058860779f, 0.012382507f, -0.0088424683f,
      0.0075111389f, 0.0086898804f, -0.005645752f, -0.0071487427f, -0.010246277f, 0.010742188f, 0.0074691772f,
      -0.0087280273f, 0.0032615662f, -0.0013141632f, -0.010406494f, 0.0014724731f, 0.014640808f, 0.014328003f,
      -0.0024299622f, 0.00022125244f, 0.013153076f, 0.0070457458f, -0.0036869049f, -0.012573242f, 0.0025901794f,
      0.0027999878f, -0.011383057f, -0.0059547424f, 0.005317688f, 0.001865387f, 0.011138916f, 0.0052261353f,
      0.0035648346f, 0.00079536438f, -0.0028533936f, -0.004524231f, 0.0084075928f, -0.0048217773f, 0.0080108643f,
      -0.0089416504f, 0.0067710876f, 0.018920898f, 0.0037651062f, 0.0033416748f, -0.014434814f, 0.015258789f, -0.0013656616f,

      0.0021858215f, 0.00084114075f, 0.0070838928f, 0.002576828f, 0.0029411316f, 0.0053520203f, -0.0054779053f,
      -0.0084915161f, 0.004776001f, -0.0020275116f, 0.0030059814f, -0.00096559525f, 0.010726929f, -0.0082473755f,
      0.008430481f, 0.0028343201f, 0.011825562f, 0.0033950806f, 0.0058860779f, 0.012382507f, -0.0088424683f,
      0.0075111389f, 0.0086898804f, -0.005645752f, -0.0071487427f, -0.010246277f, 0.010742188f, 0.0074691772f,
      -0.0087280273f, 0.0032615662f, -0.0013141632f, -0.010406494f, 0.0055160522f, 0.004196167f, 0.012260437f,
      0.014923096f, 0.0029716492f, 0.018463135f, 0.006477356f, 0.0026435852f, -0.017547607f, 0.0046920776f,
      0.0075416565f, -0.016647339f, -0.0030784607f, -0.0026321411f, 0.010955811f, 0.0046958923f, -0.0078811646f,
      -0.011367798f, -0.0087814331f, 0.00036239624f, -0.0035305023f, 0.0070915222f, -0.0088500977f, 0.00066757202f,
      0.0023040771f, 0.0042762756f, 8.392334e-05f, 0.009979248f, -0.0089416504f, 0.0029678345f, 0.00756073f, 0.0070037842f};

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

  std::vector<float> query_data;
  GetAttentionWeight(query_data, batch_size * sequence_length * hidden_size, 1, 3);

  std::vector<float> key_data;
  GetAttentionWeight(key_data, batch_size * kv_sequence_length * hidden_size);

  std::vector<float> value_data;
  GetAttentionWeight(value_data, batch_size * kv_sequence_length * v_hidden_size, 3, 2);

  std::vector<float> bias_data;
  GetAttentionBias(bias_data, 2 * hidden_size + v_hidden_size);

  std::vector<int32_t> key_padding_mask_data = {3};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN;

  std::vector<float> output_data = {
      0.0043716431f, 0.014266968f, 0.009262085f, -0.0013360977f, 0.0062141418f, 0.001077652f, -0.0098800659f,
      -0.014007568f, 0.0026988983f, 0.00018560886f, -0.0019931793f, -0.0085754395f, 0.014709473f,
      -0.0026683807f, 0.0045852661f, 0.0017185211f, 0.010574341f, 0.012619019f, 0.0059394836f, 0.0080871582f,
      -0.011787415f, 0.013832092f, 0.010375977f, -0.0043983459f, -0.0025005341f, -0.0061950684f, 0.0087661743f,
      0.0045585632f, -0.0040664673f, -0.00045919418f, -0.0084075928f, -0.011062622f, 0.0054321289f, 0.014854431f,
      0.0041923523f, 0.0088806152f, -0.0098266602f, 0.00012207031f, 0.0091018677f, 0.00021088123f, -0.0059432983f,
      -0.011032104f, 0.0025482178f, -0.0044555664f, 0.001914978f, 0.0093383789f, -0.0018730164f, 0.0052223206f,
      -0.0092315674f, 0.0038738251f, -0.0067443848f, 0.00049209595f, 0.0094299316f, 0.0020523071f, -0.0046157837f,
      0.012199402f, -0.0039558411f, 0.00099468231f, 0.00098419189f, 0.0095291138f, 0.0049743652f, -0.0042266846f, 0.017822266f, 0.0031089783f,

      0.0043716431f, 0.014266968f, 0.009262085f, -0.0013360977f, 0.0062141418f, 0.001077652f, -0.0098800659f,
      -0.014007568f, 0.0026988983f, 0.00018560886f, -0.0019931793f, -0.0085754395f, 0.014709473f, -0.0026683807f,
      0.0045852661f, 0.0017185211f, 0.010574341f, 0.012619019f, 0.0059394836f, 0.0080871582f, -0.011787415f,
      0.013832092f, 0.010375977f, -0.0043983459f, -0.0025005341f, -0.0061950684f, 0.0087661743f, 0.0045585632f,
      -0.0040664673f, -0.00045919418f, -0.0084075928f, -0.011062622f, -0.0056495667f, 0.0079421997f, 0.010955811f,
      -0.0030021667f, 0.0019645691f, 0.010253906f, 0.00018107891f, 0.0013246536f, -0.0056915283f, 0.0027427673f,
      -0.0023956299f, -0.0045166016f, -0.0021133423f, 0.013137817f, -0.0049781799f, 0.0059776306f, 0.0038986206f,
      0.0043182373f, -0.0061035156f, -0.012260437f, 3.8146973e-06f, 0.0024757385f, -0.0040664673f, 0.0044898987f,
      -0.0029563904f, 0.0092315674f, -3.0517578e-05f, 0.0058021545f, -0.0095977783f, 0.003326416f, 0.016464233f, -0.0040283203f};

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

  std::vector<float> query_data;
  GetAttentionWeight(query_data, batch_size * sequence_length * hidden_size, 1, 3);

  std::vector<float> key_data;
  GetAttentionWeight(key_data, batch_size * kv_sequence_length * hidden_size);

  std::vector<float> value_data;
  GetAttentionWeight(value_data, batch_size * kv_sequence_length * v_hidden_size, 3, 2);

  std::vector<float> bias_data;
  GetAttentionBias(bias_data, 2 * hidden_size + v_hidden_size);

  // mask in the start of sequence
  std::vector<int32_t> key_padding_mask_data = {0, 1, 1, 1};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;

  std::vector<float> output_data = {
      -0.0010671616f, -0.0095367432f, -0.0046539307f, -0.0019245148f, -0.0083618164f, -0.0016756058f, -0.0057907104f,
      -0.0014486313f, -0.0048179626f, 0.00060939789f, -0.0092773438f, 0.0046691895f, 0.0092697144f, -0.010345459f,
      0.0060653687f, -0.0034122467f, 0.00011694431f, -0.0044403076f, -0.00050735474f, -0.0036201477f, 0.002281189f,
      -0.00093889236f, 0.012008667f, -0.0010223389f, -0.0076026917f, 0.014175415f, 0.018920898f, -0.015151978f,
      -0.0067520142f, -0.012115479f, -0.0072097778f, 0.0039558411f, 0.0054321289f, 0.014854431f, 0.0041923523f,
      0.0088806152f, -0.0098266602f, 0.00012207031f, 0.0091018677f, 0.00021088123f,

      -0.0010671616f, -0.0095367432f, -0.0046539307f, -0.0019245148f, -0.0083618164f, -0.0016756058f, -0.0057907104f,
      -0.0014486313f, -0.0048179626f, 0.00060939789f, -0.0092773438f, 0.0046691895f, 0.0092697144f, -0.010345459f,
      0.0060653687f, -0.0034122467f, 0.00011694431f, -0.0044403076f, -0.00050735474f, -0.0036201477f, 0.014862061f,
      -0.00020027161f, 0.01121521f, 0.0076293945f, -0.010070801f, 0.0028800964f, -0.00057983398f, -0.0093002319f,
      -0.0035896301f, -0.0020256042f, 5.1498413e-05f, -0.006942749f, -0.0010890961f, 0.0012245178f, 0.00054931641f,
      -0.0036659241f, -0.0039749146f, -8.392334e-05f, 0.0032367706f, -0.0053367615f};

  bool use_float16 = true;
  RunCrossAttentionTestEnv(
      query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
      number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
      use_float16);
}

TEST(CrossAttentionTest, CrossAttention_Batch2_HiddenSizeV) {
  constexpr int batch_size = 2;
  constexpr int sequence_length = 2;
  constexpr int kv_sequence_length = 4;
  constexpr int hidden_size = 80;
  constexpr int v_hidden_size = 20;
  constexpr int number_of_heads = 2;

  std::vector<float> query_data;
  GetAttentionWeight(query_data, batch_size * sequence_length * hidden_size, 1, 3);

  std::vector<float> key_data;
  GetAttentionWeight(key_data, batch_size * kv_sequence_length * hidden_size, 2, 2);

  std::vector<float> value_data;
  GetAttentionWeight(value_data, batch_size * kv_sequence_length * v_hidden_size, 3, 1);

  std::vector<float> bias_data;
  GetAttentionBias(bias_data, 2 * hidden_size + v_hidden_size);

  // mask in the middle of sequence
  std::vector<int32_t> key_padding_mask_data = {
      1, 0, 0, 1,
      1, 1, 1, 1};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;
  std::vector<float> output_data = {
      0.0073471069f, -0.0014753342f, 0.0083236694f, 0.0065841675f, 0.012115479f, 0.014762878f, 0.00010681152f,
      -0.00024986267f, -0.0011262894f, 0.0027828217f, 0.002281189f, -0.00093889236f, 0.012008667f, -0.0010223389f,
      -0.0076026917f, 0.014175415f, 0.018920898f, -0.015151978f, -0.0067520142f, -0.012115479f,

      0.0073471069f, -0.0014753342f, 0.0083236694f, 0.0065841675f, 0.012115479f, 0.014762878f, 0.00010681152f,
      -0.00024986267f, -0.0011262894f, 0.0027828217f, -0.0072097778f, 0.0039558411f, 0.0054321289f, 0.014854431f,
      0.0041923523f, 0.0088806152f, -0.0098266602f, 0.00012207031f, 0.0091018677f, 0.00021088123f,

      -0.0037136078f, 0.006072998f, -0.0068435669f, 0.0097503662f, -0.002828598f, -0.00047302246f, 0.00623703f,
      -0.0031242371f, -0.00034999847f, -0.015411377f, 0.008354187f, -0.00053501129f, 0.0033378601f, 0.0070800781f,
      -0.0037517548f, 0.015060425f, 0.0040130615f, -0.0034275055f, 0.0055580139f, 0.0013923645f,

      -0.0037136078f, 0.006072998f, -0.0068435669f, 0.0097503662f, -0.002828598f, -0.00047302246f, 0.00623703f,
      -0.0031242371f, -0.00034999847f, -0.015411377f, -0.013496399f, -0.0085983276f, 0.0078048706f, 0.015396118f,
      0.0013885498f, -0.0024604797f, -0.0030097961f, 0.0077323914f, -0.0015010834f, 9.7155571e-05f};

  bool use_float16 = true;
  RunCrossAttentionTestEnv(
      query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
      number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
      use_float16);
}
#endif

}  // namespace test
}  // namespace onnxruntime
