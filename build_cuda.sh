#!/bin/bash

set -ex

THIS_DIR=$(dirname $(realpath $0))

build_dir="build_cuda"
config="Release"

${THIS_DIR}/build.sh \
    --build_dir ${THIS_DIR}/${build_dir} \
    --config ${config} \
    --cmake_generator Ninja \
    --cmake_extra_defines \
        CMAKE_CUDA_ARCHITECTURES=70 \
        CMAKE_EXPORT_COMPILE_COMMANDS=ON \
        onnxruntime_BUILD_KERNEL_EXPLORER=ON \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr \
    --enable_training \
    --use_tensorrt --tensorrt_home /usr \
    --build_wheel \
    --skip_submodule_sync --skip_tests

        # onnxruntime_DEV_MODE=OFF \
