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
def _where_softmax_kernel(
    output_ptr,
    softmax_output_ptr,
    input_ptr,
    mask_ptr,
    input_strdie_0,
    input_stride_1,
    mask_stride_0,
    mask_stride_1,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_input_start_ptr = input_ptr + row_idx * n_cols
    q0 = row_idx // input_strdie_0
    r0 = row_idx % input_strdie_0
    q1 = r0 // input_stride_1
    r1 = r0 % input_stride_1
    row_mask_start_ptr = mask_ptr + (q0 * mask_stride_0 + q1 * mask_stride_1 + r1) * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_input_start_ptr + col_offsets
    mask_ptrs = row_mask_start_ptr + col_offsets
    input_row = tl.load(input_ptrs, mask=col_offsets < n_cols).to(tl.float32)
    mask_row = tl.load(mask_ptrs, mask=col_offsets < n_cols)
    row = tl.where(mask_row, -3.4028234663852886e38, input_row)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output = softmax_output.to(tl.float16)
    output_row_start_ptr = output_ptr + row_idx * n_cols
    softmax_output_start_ptr = softmax_output_ptr + row_idx * n_cols
    output_ptrs = output_row_start_ptr + col_offsets
    softmax_output_ptrs = softmax_output_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=col_offsets < n_cols)
    tl.store(softmax_output_ptrs, softmax_output, mask=col_offsets < n_cols)


def triton_where_softmax(x, mask):
    x = _from_dlpack(x)
    mask = _from_dlpack(mask)
    x0, x1, x2, x3 = x.shape
    _, m1, m2, _ = mask.shape
    n_cols = x3
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
    softmax_output = torch.empty_like(x, dtype=torch.float32)
    output = torch.empty_like(x)
    _where_softmax_kernel[(x0 * x1 * x2,)](
        output,
        softmax_output,
        x,
        mask,
        x1 * x2,
        x2,
        m1 * m2,
        m2,
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return to_dlpack(softmax_output), to_dlpack(output)


@triton.jit
def _where_softmax_backward_kernel(
    dx_ptr,
    dy_ptr,
    softmax_output_ptr,
    mask_ptr,
    input_strdie_0,
    input_stride_1,
    mask_stride_0,
    mask_stride_1,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_dy_start_ptr = dy_ptr + row_idx * n_cols
    row_softmax_output_start_ptr = softmax_output_ptr + row_idx * n_cols
    q0 = row_idx // input_strdie_0
    r0 = row_idx % input_strdie_0
    q1 = r0 // input_stride_1
    r1 = r0 % input_stride_1
    row_mask_start_ptr = mask_ptr + (q0 * mask_stride_0 + q1 * mask_stride_1 + r1) * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    dy_ptrs = row_dy_start_ptr + col_offsets
    softmax_output_ptrs = row_softmax_output_start_ptr + col_offsets
    mask_ptrs = row_mask_start_ptr + col_offsets
    dy_row = tl.load(dy_ptrs, mask=col_offsets < n_cols).to(tl.float32)
    softmax_output_row = tl.load(softmax_output_ptrs, mask=col_offsets < n_cols)
    mask_row = tl.load(mask_ptrs, mask=col_offsets < n_cols)
    production = dy_row * softmax_output_row
    sum = tl.sum(production, axis=0)
    sum_production = sum * softmax_output_row
    grad_row = production - sum_production
    dx_row = tl.where(mask_row, 0.0, grad_row).to(tl.float16)
    dx_row_start_ptr = dx_ptr + row_idx * n_cols
    dx_ptrs = dx_row_start_ptr + col_offsets
    tl.store(dx_ptrs, dx_row, mask=col_offsets < n_cols)


def triton_where_softmax_backward(dy, softmax_output, mask):
    dy = _from_dlpack(dy)
    softmax_output = _from_dlpack(softmax_output)
    mask = _from_dlpack(mask)
    x0, x1, x2, x3 = dy.shape
    _, m1, m2, _ = mask.shape
    n_cols = x3
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
    _where_softmax_backward_kernel[(x0 * x1 * x2,)](
        dx,
        dy,
        softmax_output,
        mask,
        x1 * x2,
        x2,
        m1 * m2,
        m2,
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
def transform_triton_where_softmax(graph):
    remove_nodes = []
    triton_nodes = []
    id = 0
    for node in graph.node:
        if node.op_type == "Softmax":
            where_node = get_producer(graph, node.input[0], "Where")
            if where_node is None:
                continue
            cast1_node = get_producer(graph, where_node.input[2], "Cast")
            if cast1_node is None:
                continue
            cast2_node = get_consumer(graph, node.output[0], "Cast")
            if cast2_node is None:
                continue
            remove_nodes.extend([node, where_node, cast1_node, cast2_node])
            triton_node = helper.make_node(
                "TritonOp",
                [cast1_node.input[0], where_node.input[0]],
                [node.output[0], cast2_node.output[0]],
                "TritonOp_" + str(id),
                None,
                "com.microsoft",
                func_name="triton_where_softmax",
            )
            triton_nodes.append(triton_node)
            id += 1
        elif node.op_type == "SoftmaxGrad_13":
            cast1_node = get_producer(graph, node.input[0], "Cast")
            if cast1_node is None:
                continue
            where_node = get_consumer(graph, node.output[0], "Where")
            if where_node is None:
                continue
            cast2_node = get_consumer(graph, where_node.output[0], "Cast")
            if cast2_node is None:
                continue
            remove_nodes.extend([node, where_node, cast1_node, cast2_node])
            triton_node = helper.make_node(
                "TritonOp",
                [cast1_node.input[0], node.input[1], where_node.input[0]],
                [cast2_node.output[0]],
                "TritonOp_" + str(id),
                None,
                "com.microsoft",
                func_name="triton_where_softmax_backward",
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
