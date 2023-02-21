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
def _div_where_softmax_dropout_kernel(
    output_ptr,
    mask_ptr,
    softmax_output_ptr,
    fst_ptr,  # [batch_size, 12, seq_len, seq_len]
    snd_ptr,  # scalar
    trd_ptr,  # [1, 1, seq_len, seq_len]
    fth_ptr,  # [batch_size, 1, 1, seq_len]
    strdie_0,  # 12 * seq_len
    stride_1,  # seq_len
    p,
    seed,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # dropout(softmax(where(trd, (fst.to(float32) / snd).to(float16), -65504) + fth, axis=-1))
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    row_fst_start_ptr = fst_ptr + row_start
    row_trd_start_ptr = trd_ptr + (row_idx % stride_1) * n_cols
    row_fth_start_ptr = fth_ptr + (row_idx // strdie_0) * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    fst_ptrs = row_fst_start_ptr + col_offsets
    trd_ptrs = row_trd_start_ptr + col_offsets
    fth_ptrs = row_fth_start_ptr + col_offsets
    fst_row = tl.load(fst_ptrs, mask=mask).to(tl.float32)
    snd_row = tl.broadcast_to(tl.load(snd_ptr + (0)), [BLOCK_SIZE])
    trd_row = tl.load(trd_ptrs, mask=mask)
    fth_row = tl.load(fth_ptrs, mask=mask).to(tl.float32)
    row = tl.where(trd_row, fst_row / snd_row, -65504.0) + fth_row
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = (numerator / denominator).to(tl.float16)
    random = tl.rand(seed, row_start + col_offsets)
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


def triton_div_where_softmax_dropout(fst, snd, trd, fth):
    fst = _from_dlpack(fst)
    snd = _from_dlpack(snd)
    trd = _from_dlpack(trd)
    fth = _from_dlpack(fth)
    p = 1.0 - 0.1
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
    _div_where_softmax_dropout_kernel[(s0 * s1 * s2,)](
        output,
        mask,
        softmax_output,
        fst,
        snd,
        trd,
        fth,
        s1 * s2,
        s2,
        p,
        seed,
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return to_dlpack(output), to_dlpack(mask), to_dlpack(softmax_output)


@triton.jit
def _div_where_softmax_dropout_backward_kernel(
    dx_ptr,
    dy_ptr,  # [batch_size, 12, seq_len, seq_len]
    dropout_mask_ptr,  # [batch_size, 12, seq_len, seq_len]
    softmax_output_ptr,  # [batch_size, 12, seq_len, seq_len]
    where_mask_ptr,  # [1, 1, seq_len, seq_len]
    div_rhs_ptr,  # scalar
    stride,  # seq_len
    p,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # where(where_mask, softmax_grad(dropout_grad(dy, dorpout_mask), softmax_output), axis=-1), 0) / div_rhs
    row_idx = tl.program_id(0)
    row_dy_start_ptr = dy_ptr + row_idx * n_cols
    row_dropout_mask_start_ptr = dropout_mask_ptr + row_idx * n_cols
    row_softmax_output_start_ptr = softmax_output_ptr + row_idx * n_cols
    row_where_mask_start_ptr = where_mask_ptr + (row_idx % stride) * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dy_ptrs = row_dy_start_ptr + col_offsets
    dropout_mask_ptrs = row_dropout_mask_start_ptr + col_offsets
    softmax_output_ptrs = row_softmax_output_start_ptr + col_offsets
    where_mask_ptrs = row_where_mask_start_ptr + col_offsets
    dy_row = tl.load(dy_ptrs, mask=mask)
    dropout_mask_row = tl.load(dropout_mask_ptrs, mask=mask)
    softmax_output_row = tl.load(softmax_output_ptrs, mask=mask).to(tl.float32)
    where_mask_row = tl.load(where_mask_ptrs, mask=mask)
    div_rhs_row = tl.broadcast_to(tl.load(div_rhs_ptr + (0)), [BLOCK_SIZE])
    row = tl.where(dropout_mask_row, dy_row / p, 0.0).to(tl.float32)
    production = row * softmax_output_row
    sum = tl.sum(production, axis=0)
    sum_production = sum * softmax_output_row
    dx_row = (tl.where(where_mask_row, production - sum_production, 0.0) / div_rhs_row).to(tl.float16)
    dx_row_start_ptr = dx_ptr + row_idx * n_cols
    dx_ptrs = dx_row_start_ptr + col_offsets
    tl.store(dx_ptrs, dx_row, mask=col_offsets < n_cols)


def triton_div_where_softmax_dropout_backward(dy, dropout_mask, softmax_output, where_mask, div_rhs):
    dy = _from_dlpack(dy)
    dropout_mask = _from_dlpack(dropout_mask)
    softmax_output = _from_dlpack(softmax_output)
    where_mask = _from_dlpack(where_mask)
    div_rhs = _from_dlpack(div_rhs)
    p = 1.0 - 0.1
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
    _div_where_softmax_dropout_backward_kernel[(s0 * s1 * s2,)](
        dx,
        dy,
        dropout_mask,
        softmax_output,
        where_mask,
        div_rhs,
        s2,
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
def transform_triton_div_where_softmax_dropout(graph):
    remove_nodes = []
    triton_nodes = []
    id = 0
    for node in graph.node:
        if node.op_type == "Softmax":
            add_node = get_producer(graph, node.input[0], "Add")
            if add_node is None:
                continue
            where1_node = get_producer(graph, add_node.input[0], "Where")
            if where1_node is None:
                continue
            cast1_node = get_producer(graph, where1_node.input[1], "Cast")
            if cast1_node is None:
                continue
            div1_node = get_producer(graph, cast1_node.input[0], "Div")
            if div1_node is None:
                continue
            cast2_node = get_producer(graph, div1_node.input[0], "Cast")
            if cast2_node is None:
                continue
            dropout_node = get_consumer(graph, node.output[0], "Dropout")
            if dropout_node is None:
                continue
            softmaxgrad_node = get_consumer(graph, node.output[0], "SoftmaxGrad_13")
            if softmaxgrad_node is None:
                continue
            dropoutgrad_node = get_producer(graph, softmaxgrad_node.input[0], "DropoutGrad")
            if dropoutgrad_node is None:
                continue
            identity1_node = get_consumer(graph, softmaxgrad_node.output[0], "Identity")
            if identity1_node is None:
                continue
            where2_node = get_consumer(graph, identity1_node.output[0], "Where")
            if where2_node is None:
                continue
            cast3_node = get_consumer(graph, where2_node.output[0], "Cast")
            if cast3_node is None:
                continue
            div2_node = get_consumer(graph, cast3_node.output[0], "Div")
            if div2_node is None:
                continue
            identity2_node = get_consumer(graph, div2_node.output[0], "Identity")
            if identity2_node is None:
                continue
            cast4_node = get_consumer(graph, identity2_node.output[0], "Cast")
            if cast4_node is None:
                continue
            remove_nodes.extend(
                [
                    node,
                    add_node,
                    where1_node,
                    cast1_node,
                    div1_node,
                    cast2_node,
                    dropout_node,
                    softmaxgrad_node,
                    dropoutgrad_node,
                    identity1_node,
                    where2_node,
                    cast3_node,
                    div2_node,
                    identity2_node,
                    cast4_node,
                ]
            )
            triton_node = helper.make_node(
                "TritonOp",
                [cast2_node.input[0], div1_node.input[1], where1_node.input[0], add_node.input[1]],
                [dropout_node.output[0], dropout_node.output[1], node.output[0]],
                "TritonOp_Div_Where_Softmax_Dropout_" + str(id),
                None,
                "com.microsoft",
                func_name="triton_div_where_softmax_dropout",
            )
            triton_nodes.append(triton_node)
            triton_node = helper.make_node(
                "TritonOp",
                [
                    dropoutgrad_node.input[0],
                    dropoutgrad_node.input[1],
                    softmaxgrad_node.input[1],
                    where2_node.input[0],
                    div2_node.input[1],
                ],
                [cast4_node.output[0]],
                "TritonOp_Div_Where_Softmax_Dropout_Backward_" + str(id),
                None,
                "com.microsoft",
                func_name="triton_div_where_softmax_dropout_backward",
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
