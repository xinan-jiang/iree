// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_DIALECT_RUY_BASE
#define IREE_DIALECT_RUY_BASE

include "mlir/IR/OpBase.td"

def RUY_Dialect : Dialect {
  let name = "ruy";
  let cppNamespace = "::mlir::iree_compiler::ruy";

   let summary = [{
    An experimental dialect to represent specific lowerings of matmul oprations
  }];
  
}

class Ruy_OP<string mnemonic, list<OpTrait> traits = []> :
    Op<RUY_Dialect, mnemonic, traits>;

#endif  // IREE_DIALECT_RUY_BASE
