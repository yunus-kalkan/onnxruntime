if (onnxruntime_USE_FLASH_ATTENTION)
  set(PATCH ${PROJECT_SOURCE_DIR}/patches/cutlass/fmha.patch)

  include(FetchContent)
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/nvidia/cutlass.git
    GIT_TAG        66d9cddc832c1cdc2b30a8755274f7f74640cfe6
    PATCH_COMMAND  git apply --reverse --check ${PATCH} || git apply --ignore-whitespace --whitespace=nowarn ${PATCH}
  )

  FetchContent_GetProperties(cutlass)
  if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
  endif()
endif()
