// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/rnn/lstm_grad.h"

namespace onnxruntime::contrib {

namespace {

// struct ComputeArgs {
// }

// template <typename T>
// void PrepareForCompute(OpKernelContext* context, rnn::detail::Direction direction, size_t hidden_size, ComputeArgs** compute_args) {
//   const Tensor* W = context->Input<Tensor>(1);
//   const Tensor* R = context->Input<Tensor>(2);

//   // Load input weight parameters and recurrent weight parameters
//   const auto load_weights = [](const Tensor* weights, size_t index) {
//     // index represents the direction of the weight to be loaded.
//     // For example,
//     //   in a uni-directional lstm, index can only ever be 0.
//     //   in a bi-directional lstm, index 0 represents forward weights and index 1 represents backward weights
//     const auto& weights_shape = weights->Shape();
//     const auto* weights_data = weights->Data<float>();
//     const size_t weights_size_per_direction = SafeInt<size_t>(weights_shape[1]) * weights_shape[2];
//     return rnn::detail::GemmWeights<float>(index, weights_data, weights_size_per_direction, rnn::detail::PackedWeights());
//   };

//   rnn::detail::GemmWeights<float> weights1 = load_weights(W, 0);
//   rnn::detail::GemmWeights<float> recurrent_weights1 = load_weights(R, 0);

//   rnn::detail::GemmWeights<float> weights2 =
//       direction == rnn::detail::Direction::kBidirectional ? load_weights(W, 1) : rnn::detail::GemmWeights<float>();
//   rnn::detail::GemmWeights<float> recurrent_weights2 =
//       direction == rnn::detail::Direction::kBidirectional ? load_weights(R, 1) : rnn::detail::GemmWeights<float>();

//   const Tensor* B = context.Input<Tensor>(3);  // bias. [num_directions, 8*hidden_size]
//   const auto load_bias = [](const Tensor* B, size_t index, size_t hidden_size) {
//     const size_t bias_size_per_direction = 8 * hidden_size;
//     gsl::span<const T> bias = B != nullptr ? B->DataAsSpan<T>() : gsl::span<const T>();
//     return bias.empty() ? bias : bias.subspan(index * bias_size_per_direction, bias_size_per_direction);
//   };
//   gsl::span<const T> bias1 = load_bias(B, 0, hidden_size);

//   const Tensor* P = context.Input<Tensor>(7);  // peephole weights. [num_directions, 3*hidden_size]
//   const auto load_peephole_weights = [](const Tensor* P, size_t index, size_t hidden_size) {
//     const size_t peephole_weights_size_per_direction = 3 * hidden_size;
//     gsl::span<const InputT> peephole_weights = P != nullptr ? P->DataAsSpan<InputT>() : gsl::span<const InputT>();
//     return peephole_weights.empty() ? peephole_weights : peephole_weights.subspan(index * peephole_weights_size_per_direction, peephole_weights_size_per_direction);
//   };

//   const Tensor* X = context.Input<Tensor>(0);  // input sequence [seq_length, batch_size, input_size]

//   const Tensor* sequence_length = context.Input<Tensor>(4);        // [batch_size]
//   const Tensor* previous_hidden_state = context.Input<Tensor>(5);  // [num_directions, batch_size, hidden_size]
//   const Tensor* previous_cell_state = context.Input<Tensor>(6);    // [num_directions, batch_size, hidden_size]
//   gsl::span<const T> peephole_weights = P != nullptr ? P->DataAsSpan<T>() : gsl::span<const T>();

//   AllocatorPtr alloc;
//   ORT_RETURN_IF_ERROR(context.GetTempSpaceAllocator(&alloc));
// }

}  // namespace

#define REGISTER_LSTMGRAD_KERNEL_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LSTMGrad,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LSTMGrad<T>);

REGISTER_LSTMGRAD_KERNEL_TYPED(float)

template <typename T>
Status LSTMGrad<T>::Compute(OpKernelContext*) const {
  // const Tensor* X = context.Input<Tensor>(0);                      // input sequence [seq_length, batch_size, input_size]
  // const Tensor* W = context->Input<Tensor>(1);                     // weights [seq_length, batch_size, input_size]
  // const Tensor* R = context->Input<Tensor>(2);                     // recurrence weights [seq_length, batch_size, input_size]
  // const Tensor* B = context.Input<Tensor>(3);                      // bias. [num_directions, 8*hidden_size]
  // const Tensor* sequence_length = context.Input<Tensor>(4);        // [batch_size]
  // const Tensor* previous_hidden_state = context.Input<Tensor>(5);  // [num_directions, batch_size, hidden_size]
  // const Tensor* previous_cell_state = context.Input<Tensor>(6);    // [num_directions, batch_size, hidden_size]
  // const Tensor* P = context.Input<Tensor>(7);                      // peephole weights. [num_directions, 3*hidden_size]
  // const Tensor* dY = context.Input<Tensor>(8);

  // // Calculate all gates
  // // Tensor* input_sequence_grad = ctx->Output<Tensor>(0);
  // // Tensor* previous_hidden_state_grad = ctx->Output<Tensor>(1);
  // // Tensor* weights_grad = ctx->Output<Tensor>(2);
  // // Tensor* recurrence_weights_grad = ctx->Output<Tensor>(3);
  // // Tensor* bias_grad = ctx->Output<Tensor>(4);
  // // Tensor* peephole_weights_grad = ctx->Output<Tensor>(5);
  // // AllocatorPtr alloc;
  // // ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
  // // Tensor previous_cell_state_grad(previous_cell_state->DataType(), previous_cell_state->Shape(), alloc);

  // // Prepare for compute
  // PrepareForCompute();

  // // Run forward
  // RunForward();

  // // Compute gradient
  // ComputeGradients();

  // // Load input weight parameters and recurrent weight parameters
  // const auto load_weights = [](const Tensor* weights, size_t index) {
  // index represents the direction of the weight to be loaded.
  // For example,
  //   in a uni-directional lstm, index can only ever be 0.
  //   in a bi-directional lstm, index 0 represents forward weights and index 1 represents backward weights
  //   const auto& weights_shape = weights->Shape();
  //   const auto* weights_data = weights->Data<float>();
  //   const size_t weights_size_per_direction = SafeInt<size_t>(weights_shape[1]) * weights_shape[2];
  //   return rnn::detail::GemmWeights<float>(index, weights_data, weights_size_per_direction, rnn::detail::PackedWeights());
  // };

  // rnn::detail::GemmWeights<float> weights1 = load_weights(W, 0);
  // rnn::detail::GemmWeights<float> recurrent_weights1 = load_weights(R, 0);

  // rnn::detail::GemmWeights<float> weights2 =
  //     direction_ == rnn::detail::Direction::kBidirectional ? load_weights(W, 1) : rnn::detail::GemmWeights<float>();
  // rnn::detail::GemmWeights<float> recurrent_weights2 =
  //     direction_ == rnn::detail::Direction::kBidirectional ? load_weights(R, 1) : rnn::detail::GemmWeights<float>();

  // Information available to us:
  //   - dL/dHt: partial of the loss with respect to all the hidden state at time stamp t.
  //   - dL/dCt: partial of the loss with respect to the cell state at time stamp t.
  // What is needed to be calculated:
  //   - dL/dX: partial of the loss with respect to the input sequence.
  //   - dL/dHtminus1: partial of the loss with respect to the hidden state at time stamp t-1.
  //   - dL/dW: partial of the loss with respect to the weight parameters.
  //   - dL/dR: partial of the loss with respect to the recurrence weight parameters.
  //   - dL/dB: partial of the loss with respect to the bias parameters.
  //   - dL/dP: partial of the loss with respect to the peephole weight parameters.
  // Intermediate results that will need to be computed to calculate the partials above:
  //   - dL/dit, dL/dft, dL/dct, dL/dot: partial of the loss with respect to the outputs at the
  //                                     input, forget, cell and output gates
  // Information from forward that is needed during backward. A future consideration may be to stash these intermediate
  // computations to avoid recomputation during backward.
  //   - it: output at the input gate
  //   - ft: output at the forget gate
  //   - ct: output at the cell gate
  //   - ot: output at the output gate

  // Calculate iofg
  // for (size_t idx = 0; idx < sequence_length; ++idx) {
  // }

  return Status::OK();
}

}  // namespace onnxruntime::contrib
