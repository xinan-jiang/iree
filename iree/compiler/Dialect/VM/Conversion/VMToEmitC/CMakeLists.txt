# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(${IREE_ENABLE_EMITC})
  iree_add_all_subdirs()
  
  iree_cc_library(
    NAME
      VMToEmitC
    HDRS
      "ConvertVMToEmitC.h"
    SRCS
      "ConvertVMToEmitC.cpp"
    DEPS
      MLIRIR
      MLIRPass
      MLIREmitC
      MLIRTransforms
      iree::compiler::Dialect::VM::IR
    INCLUDES
      "${PROJECT_SOURCE_DIR}/third_party/mlir-emitc/include"
      "${PROJECT_BINARY_DIR}/third_party/mlir-emitc/include"
    PUBLIC
  )
endif()
