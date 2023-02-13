# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
from onnx import helper
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack

import triton
import triton.language as tl

from .._graph_transformer_registry import register_graph_transformer


@triton.jit
def _elementwise_softmax_dropout_kernel(
    output_ptr,
    mask_ptr,
    softmax_output_ptr,
    fst_ptr,
    snd_ptr,
    trd_ptr,
    strdie_0,
    stride_1,
    scale,
    p,
    seed,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    row_fst_start_ptr = fst_ptr + row_start
    row_snd_start_ptr = snd_ptr + row_start
    q = row_idx // strdie_0
    r = (row_idx % strdie_0) % stride_1
    row_trd_start_ptr = trd_ptr + (q * stride_1 + r) * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    fst_ptrs = row_fst_start_ptr + col_offsets
    snd_ptrs = row_snd_start_ptr + col_offsets
    trd_ptrs = row_trd_start_ptr + col_offsets
    fst_row = tl.load(fst_ptrs, mask=mask)
    snd_row = tl.load(snd_ptrs, mask=mask)
    trd_row = tl.load(trd_ptrs, mask=mask)
    row = ((fst_row + snd_row) * scale - trd_row).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = (numerator / denominator).to(tl.float16)
    random = tl.rand(seed, col_offsets)
    mask_output = random < p
    output = tl.where(mask_output, softmax_output / p, 0.0)
    output_row_start_ptr = output_ptr + row_start
    mask_row_start_ptr = mask_ptr + row_start
    softmax_output_start_ptr = softmax_output_ptr + row_start
    output_ptrs = output_row_start_ptr + col_offsets
    mask_ptrs = mask_row_start_ptr + col_offsets
    softmax_output_ptrs = softmax_output_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)
    tl.store(mask_ptrs, mask_output, mask=mask)
    tl.store(softmax_output_ptrs, softmax_output, mask=mask)


def triton_elementwise_softmax_dropout(fst, snd, trd):
    fst = _from_dlpack(fst)
    snd = _from_dlpack(snd)
    trd = _from_dlpack(trd)
    p = 1.0 - 0.1
    scale = 0.125
    seed = 2333
    s0, s1, s2, s3 = fst.shape
    n_cols = s3
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    output = torch.empty_like(fst)
    # Torch's DLPack doesn't support bool tensors, use uint8 instead.
    mask = torch.empty_like(fst, dtype=torch.uint8)
    softmax_output = torch.empty_like(fst)
    _elementwise_softmax_dropout_kernel[(s0 * s1 * s2,)](
        output,
        mask,
        softmax_output,
        fst,
        snd,
        trd,
        s1 * s2,
        s2,
        scale,
        p,
        seed,
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return to_dlpack(output), to_dlpack(mask), to_dlpack(softmax_output)


@triton.jit
def _elementwise_softmax_dropout_backward_kernel(
    dx_ptr,
    dy_ptr,
    mask_ptr,
    softmax_output_ptr,
    scale,
    p,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_dy_start_ptr = dy_ptr + row_idx * n_cols
    row_mask_start_ptr = mask_ptr + row_idx * n_cols
    row_softmax_output_start_ptr = softmax_output_ptr + row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    dy_ptrs = row_dy_start_ptr + col_offsets
    mask_ptrs = row_mask_start_ptr + col_offsets
    softmax_output_ptrs = row_softmax_output_start_ptr + col_offsets
    dy_row = tl.load(dy_ptrs, mask=col_offsets < n_cols)
    mask_row = tl.load(mask_ptrs, mask=col_offsets < n_cols)
    softmax_output_row = tl.load(softmax_output_ptrs, mask=col_offsets < n_cols).to(tl.float32)
    row = tl.where(mask_row, dy_row / p, 0.0).to(tl.float32)
    production = row * softmax_output_row
    sum = tl.sum(production, axis=0)
    sum_production = sum * softmax_output_row
    dx_row = (production - sum_production).to(tl.float16) * scale
    dx_row_start_ptr = dx_ptr + row_idx * n_cols
    dx_ptrs = dx_row_start_ptr + col_offsets
    tl.store(dx_ptrs, dx_row, mask=col_offsets < n_cols)


def triton_elementwise_softmax_dropout_backward(dy, mask, softmax_output):
    dy = _from_dlpack(dy)
    mask = _from_dlpack(mask)
    softmax_output = _from_dlpack(softmax_output)
    p = 1.0 - 0.1
    scale = 0.125
    s0, s1, s2, s3 = dy.shape
    n_cols = s3
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    dx = torch.empty_like(dy)
    _elementwise_softmax_dropout_backward_kernel[(s0 * s1 * s2,)](
        dx,
        dy,
        mask,
        softmax_output,
        scale,
        p,
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return to_dlpack(dx)


def get_producer(graph, arg, op_type):
    for node in graph.node:
        if node.op_type == op_type:
            for output in node.output:
                if output == arg:
                    return node
    return None


def get_consumer(graph, arg, op_type):
    for node in graph.node:
        if node.op_type == op_type:
            for input in node.input:
                if input == arg:
                    return node
    return None


@register_graph_transformer(priority=0)
def transform_triton_elementwise_softmax_dropout(graph):
    remove_nodes = []
    triton_nodes = []
    id = 0
    for node in graph.node:
        if node.op_type == "Softmax":
            sub_node = get_producer(graph, node.input[0], "Sub")
            if sub_node is None:
                continue
            mul1_node = get_producer(graph, sub_node.input[0], "Mul")
            if mul1_node is None:
                continue
            add_node = get_producer(graph, mul1_node.input[0], "Add")
            if add_node is None:
                continue
            cast1_node = get_consumer(graph, node.output[0], "Cast")
            if cast1_node is None:
                continue
            dropout_node = get_consumer(graph, cast1_node.output[0], "Dropout")
            if dropout_node is None:
                continue
            cast2_node = get_consumer(graph, dropout_node.output[0], "Cast")
            if cast2_node is None:
                continue
            softmaxgrad_node = get_consumer(graph, node.output[0], "SoftmaxGrad_13")
            if softmaxgrad_node is None:
                continue
            cast3_node = get_producer(graph, softmaxgrad_node.input[0], "Cast")
            if cast3_node is None:
                continue
            dropoutgrad_node = get_producer(graph, cast3_node.input[0], "DropoutGrad")
            if dropoutgrad_node is None:
                continue
            cast4_node = get_producer(graph, dropoutgrad_node.input[0], "Cast")
            if cast4_node is None:
                continue
            identity_node = get_consumer(graph, softmaxgrad_node.output[0], "Identity")
            if identity_node is None:
                continue
            mul2_node = get_consumer(graph, identity_node.output[0], "Mul")
            if mul2_node is None:
                continue
            remove_nodes.extend(
                [
                    node,
                    sub_node,
                    mul1_node,
                    add_node,
                    cast1_node,
                    dropout_node,
                    cast2_node,
                    softmaxgrad_node,
                    cast3_node,
                    dropoutgrad_node,
                    cast4_node,
                    identity_node,
                    mul2_node,
                ]
            )
            triton_node = helper.make_node(
                "TritonOp",
                [add_node.input[0], add_node.input[1], sub_node.input[1]],
                [cast2_node.output[0], dropout_node.output[1], node.output[0]],
                "TritonOp_Elementwise_Softmax_Dropout_" + str(id),
                None,
                "com.microsoft",
                func_name="triton_elementwise_softmax_dropout",
            )
            triton_nodes.append(triton_node)
            triton_node = helper.make_node(
                "TritonOp",
                [cast4_node.input[0], dropoutgrad_node.input[1], softmaxgrad_node.input[1]],
                [mul2_node.output[0]],
                "TritonOp_Elementwise_Softmax_Dropout_Backward_" + str(id),
                None,
                "com.microsoft",
                func_name="triton_elementwise_softmax_dropout_backward",
            )
            triton_nodes.append(triton_node)
            id += 1

    all_nodes = []
    for node in graph.node:
        if node not in remove_nodes:
            all_nodes.append(node)

    for node in triton_nodes:
        all_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(all_nodes)
