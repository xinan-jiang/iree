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

#include "iree/base/signature_mangle.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree_tf_compiler/TF/Passes.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

static LogicalResult rewriteTFVariableOpToFlowVariableOp(Operation &op,
                                                         Value flowPtr) {
  OpBuilder builder(&op);
  if (auto readVariable = dyn_cast<mlir::TF::ReadVariableOp>(op)) {
    auto load =
        builder.create<iree_compiler::IREE::Flow::VariableLoadIndirectOp>(
            readVariable.getLoc(), readVariable.value().getType(), flowPtr);
    readVariable.value().replaceAllUsesWith(load.result());
    readVariable.erase();
  } else if (auto assignVariable = dyn_cast<mlir::TF::AssignVariableOp>(op)) {
    builder.create<iree_compiler::IREE::Flow::VariableStoreIndirectOp>(
        assignVariable.getLoc(), assignVariable.value(), flowPtr);
    assignVariable.erase();
  } else {
    return op.emitError() << "could not lower resource op to flow: "
                          << op.getName();
  }
  return success();
}

static LogicalResult convertTFGlobalTensorsToFlowVariables(ModuleOp module) {
  OpBuilder globalBuilder(module.getBodyRegion());
  SymbolTable symbolTable(module);

  if (auto sessionInitializer =
          tf_saved_model::GetSessionInitializerOp(module)) {
    return sessionInitializer.emitError()
           << "Session initializer is not supported yet";
  }

  DenseMap<StringRef, std::string> symNameToFlowSymName;
  for (auto globalTensor : module.getOps<tf_saved_model::GlobalTensorOp>()) {
    auto exportedNames = tf_saved_model::GetExportedNames(globalTensor);
    std::string flowSymName;
    if (exportedNames.empty()) {
      flowSymName = "__iree_flow_" + globalTensor.sym_name().str();
    } else if (exportedNames.size() == 1) {
      flowSymName = exportedNames[0].str();
    } else {
      return globalTensor.emitError()
             << "Multiple exported names for global tensor not supported yet";
    }
    symNameToFlowSymName[globalTensor.sym_name()] = flowSymName;
    auto variableOp =
        globalBuilder.create<iree_compiler::IREE::Flow::VariableOp>(
            globalTensor.getLoc(), flowSymName, globalTensor.is_mutable(),
            globalTensor.type(), globalTensor.value());
    variableOp.setPrivate();
  }

  // TODO(silvasean): Make this conversion interprocedural.
  for (auto func : module.getOps<FuncOp>()) {
    if (!tf_saved_model::IsExported(func)) {
      continue;
    }
    SmallVector<unsigned, 4> argsToErase;
    OpBuilder builder(func.getBody());
    SmallVector<Value, 8> typeConversionWorklist;
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      auto globalTensor = tf_saved_model::LookupBoundInputOfType<
          tf_saved_model::GlobalTensorOp>(func, i, symbolTable);
      if (!globalTensor) {
        continue;
      }
      auto variableAddressOp =
          builder.create<iree_compiler::IREE::Flow::VariableAddressOp>(
              globalTensor.getLoc(),
              iree_compiler::IREE::PtrType::get(globalTensor.type()),
              builder.getSymbolRefAttr(
                  symNameToFlowSymName[globalTensor.sym_name()]));
      typeConversionWorklist.push_back(variableAddressOp.getResult());
      func.getArgument(i).replaceAllUsesWith(variableAddressOp.getResult());
      argsToErase.push_back(i);
    }
    func.eraseArguments(argsToErase);

    while (!typeConversionWorklist.empty()) {
      Value v = typeConversionWorklist.pop_back_val();
      Type desiredType = v.getType();
      for (OpOperand &use : llvm::make_early_inc_range(v.getUses())) {
        Operation *owner = use.getOwner();

        // If the user is already in the flow dialect, then everything is ok.
        if (iree_compiler::IREE::Flow::FlowDialect::isDialectOp(owner)) {
          continue;
        }

        // If a user is just a terminator passing the value through a successor
        // operand, propagate through the successor operand.
        // TODO(silvasean): Handle case of different types in preds.
        // This would require calculating a common type.
        // This won't be a problem unless we see IR that effectively phi's
        // together different resources, which I don't think tensorflow does.
        if (BranchOpInterface branchOp = dyn_cast<BranchOpInterface>(owner)) {
          if (auto arg =
                  branchOp.getSuccessorBlockArgument(use.getOperandNumber())) {
            if (arg->getType() != desiredType) {
              arg->setType(desiredType);
              typeConversionWorklist.push_back(*arg);
            }
            continue;
          }
        }

        // Resource types can have subtypes (or lack thereof) and casting
        // between them is allowed. Here we just pass through.
        if (auto castOp = dyn_cast<mlir::TF::CastOp>(owner)) {
          assert(v == castOp.x());
          castOp.y().replaceAllUsesWith(castOp.x());
          castOp.erase();
          // The RAUW could have added more uses of `v`, so put it back on the
          // worklist and process it again.
          typeConversionWorklist.push_back(v);
          break;
        }

        if (failed(rewriteTFVariableOpToFlowVariableOp(*owner, v))) {
          return failure();
        }
      }
    }
  }

  // Erase all the global tensors.
  for (auto globalTensor : llvm::make_early_inc_range(
           module.getOps<tf_saved_model::GlobalTensorOp>())) {
    globalTensor.erase();
  }
  return success();
}

class LowerGlobalTensors
    : public PassWrapper<LowerGlobalTensors, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::tf_saved_model::TensorFlowSavedModelDialect,
                    iree_compiler::IREE::Flow::FlowDialect,
                    iree_compiler::IREEDialect>();
  }

  void runOnOperation() override {
    if (failed(convertTFGlobalTensorsToFlowVariables(getOperation()))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLowerGlobalTensorsPass() {
  return std::make_unique<LowerGlobalTensors>();
}

static PassRegistration<LowerGlobalTensors> pass(
    "iree-tf-saved-model-lower-global-tensors",
    "Lowers tf_saved_model global tensors to IREE flow dialect variables");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
