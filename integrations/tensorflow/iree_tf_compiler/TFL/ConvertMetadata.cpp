// Copyright 2021 Google LLC
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

#include "iree_tf_compiler/TFL/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

// Extract the input and output names
static void splitFunctionIONames(StringAttr namesAttr,
                                 llvm::SmallVectorImpl<std::string> &names) {
  SmallVector<StringRef, 4> namesRef;
  llvm::SplitString(namesAttr.getValue(), namesRef, ",");
  for (auto nameRef : namesRef) {
    names.push_back(nameRef.str());
  }
}

class ConvertModuleMetadataPass
    : public PassWrapper<ConvertModuleMetadataPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    // None currently handled.
    // auto moduleOp = getOperation();
  }
};

class ConvertFunctionMetadataPass
    : public PassWrapper<ConvertFunctionMetadataPass, OperationPass<FuncOp>> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();

    // TF/TFL pack their I/O names on an annoying dictionary. We want our shape
    // names to match up with those for readability so we extract them here.
    // Is this ugly? Yeah - but such is what we have to deal with here.
    auto entryFuncAttr = funcOp->getAttr("tf.entry_function")
                             .template dyn_cast<DictionaryAttr>();
    if (entryFuncAttr) {
      funcOp->setAttr("iree.module.export", UnitAttr::get(&getContext()));

      SmallVector<std::string, 4> inputNames;
      SmallVector<std::string, 4> outputNames;
      splitFunctionIONames(
          entryFuncAttr.get("inputs").template cast<StringAttr>(), inputNames);
      splitFunctionIONames(
          entryFuncAttr.get("outputs").template cast<StringAttr>(),
          outputNames);
      if (inputNames.size() != funcOp.getNumArguments() ||
          outputNames.size() != funcOp.getNumResults()) {
        funcOp.emitError()
            << "tf.entry_function attribute malformed: inputs/outputs don't "
               "match the function signature";
        signalPassFailure();
        return;
      }
      for (unsigned i = 0; i < inputNames.size(); ++i) {
        funcOp.setArgAttr(i, "iree.identifier",
                          StringAttr::get(inputNames[i], &getContext()));
      }
      for (unsigned i = 0; i < outputNames.size(); ++i) {
        funcOp.setResultAttr(i, "iree.identifier",
                             StringAttr::get(outputNames[i], &getContext()));
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertModuleMetadataPass() {
  return std::make_unique<ConvertModuleMetadataPass>();
}

std::unique_ptr<OperationPass<FuncOp>> createConvertFunctionMetadataPass() {
  return std::make_unique<ConvertFunctionMetadataPass>();
}

static PassRegistration<ConvertModuleMetadataPass> modulePass(
    "iree-tflite-convert-module-metadata",
    "Converts TFLite attributes to IREE attributes on modules");

static PassRegistration<ConvertFunctionMetadataPass> funcPass(
    "iree-tflite-convert-function-metadata",
    "Converts TFLite attributes to IREE attributes on functions");

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
