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

iree_add_all_subdirs()

iree_cc_library(
  NAME
    LinkerTools_internal
  SRCS
    "LinkerTools.cpp"
    "UnixLinkerTool.cpp"
    "WindowsLinkerTool.cpp"
  DEPS
    LLVMCore
    LLVMSupport
    MLIRSupport
    iree::base::core_headers
    iree::compiler::Dialect::HAL::Target::LLVM::LinkerTool_hdrs
  PUBLIC
)
