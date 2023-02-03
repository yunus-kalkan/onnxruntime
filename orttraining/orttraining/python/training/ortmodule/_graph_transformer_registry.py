# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Callable

from onnx.onnx_ml_pb2 import GraphProto


class GraphTransformerRegistry:
    _TRANSFORMER_FUNCS = {}

    @classmethod
    def register(cls, priority: int, fn: Callable[[GraphProto], None]):
        if priority in cls._TRANSFORMER_FUNCS:
            cls._TRANSFORMER_FUNCS[priority].append(fn)
        else:
            cls._TRANSFORMER_FUNCS[priority] = [fn]

    @classmethod
    def transform_all(cls, graph: GraphProto):
        keys = list(cls._TRANSFORMER_FUNCS.keys())
        keys.sort(reverse=True)
        for key in keys:
            for fn in cls._TRANSFORMER_FUNCS[key]:
                fn(graph)


def register_graph_transformer(priority: int = 0):
    def graph_transformer_wrapper(fn):
        GraphTransformerRegistry.register(priority, fn)
        return fn

    return graph_transformer_wrapper
