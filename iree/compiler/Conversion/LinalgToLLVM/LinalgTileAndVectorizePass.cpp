// Copyright 2020 Google LLC
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

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-to-llvm-tile-and-vectorize"

namespace mlir {
namespace iree_compiler {

namespace {
template <typename LinalgOpTy>
struct TileWorkgroups : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileWorkgroups(MLIRContext *context, linalg::LinalgTilingOptions options,
                 linalg::LinalgMarker marker, PatternBenefit benefit = 1)
      : Base(LinalgOpTy::getOperationName(), context, options, marker,
             benefit) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> tensorResults;
    if (failed(Base::matchAndRewriteBase(op, rewriter, tensorResults)) ||
        !tensorResults.empty()) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

namespace {
struct TileAndVectorizeWorkgroups
    : public PassWrapper<TileAndVectorizeWorkgroups, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AffineDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnFunction() override;
};
}  // namespace

void TileAndVectorizeWorkgroups::runOnFunction() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();
  CPUKernelDispatch cpuKernelDispatch;

  if (!isEntryPoint(funcOp)) return;

  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body.getBlocks())) {
    funcOp.emitError("unhandled dispatch function with multiple blocks");
    return signalPassFailure();
  }
  Block &block = body.front();
  auto linalgOps = block.getOps<linalg::LinalgOp>();
  if (linalgOps.empty()) return;

  SmallVector<linalg::LinalgOp, 4> linalgOpsVec =
      llvm::to_vector<4>(llvm::map_range(
          linalgOps, [](Operation *op) { return cast<linalg::LinalgOp>(op); }));
  linalg::Aliases aliases;
  linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOpsVec);
  Optional<LaunchConfig> launchConfigOpt =
      initCPULaunchConfig(context, dependenceGraph, linalgOpsVec);
  if (!launchConfigOpt) {
    funcOp.emitError("unable to find launch configuration");
    return signalPassFailure();
  }
  LaunchConfig &launchConfig = *launchConfigOpt;

  TileAndFuseOptions tileAndFuseOptions;

  tileAndFuseOptions.tileLevel = 2;
  tileAndFuseOptions.loopType = linalg::LinalgTilingLoopType::Loops;

  if (failed(tileAndFuseLinalgBufferOps(funcOp, linalgOpsVec, dependenceGraph,
                                        launchConfig, tileAndFuseOptions,
                                        getVectorizeMarker()))) {
    return signalPassFailure();
  }

  launchConfig.finalize(funcOp);

  // Apply vectorization patterns.
  {
    OwningRewritePatternList vectorizationPatterns;
    vectorizationPatterns
        .insert<linalg::LinalgVectorizationPattern<linalg::FillOp>,
                linalg::LinalgVectorizationPattern<linalg::MatmulOp>,
                linalg::LinalgVectorizationPattern<linalg::BatchMatmulOp>>(
            context, linalg::LinalgMarker(
                         Identifier::get(getVectorizeMarker(), context)));
    applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));
  }

  // Apply vector specific operation lowering.
  {
    vector::VectorTransformsOptions vectorTransformsOptions =
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct);
    OwningRewritePatternList vectorContractLoweringPatterns;
    vectorContractLoweringPatterns
        .insert<ContractionOpToOuterProductOpLowering,
                ContractionOpToMatmulOpLowering, ContractionOpLowering>(
            vectorTransformsOptions, context);
    applyPatternsAndFoldGreedily(funcOp,
                                 std::move(vectorContractLoweringPatterns));
  }

  // Programmatic controlled lowering of vector.transfer only.
  {
    VectorTransferToSCFOptions vectorToSCFOptions =
        VectorTransferToSCFOptions().setUnroll(true);
    OwningRewritePatternList vectorToLoopsPatterns;
    populateVectorToSCFConversionPatterns(vectorToLoopsPatterns, context,
                                          vectorToSCFOptions);
    // Hosit hierarchical tiling indexing and other loop invariant transfer ops
    // computation.
    linalg::hoistViewAllocOps(funcOp);
    linalg::hoistRedundantVectorTransfers(funcOp);

    // TODO(ataei): Move this to common vector dialect patterns.
    populateStdLegalizationPatternsForSPIRVLowering(context,
                                                    vectorToLoopsPatterns);
    applyPatternsAndFoldGreedily(funcOp, std::move(vectorToLoopsPatterns));
  }
}

std::unique_ptr<FunctionPass> createLinalgTileAndVectorizeWorkgroupsPass() {
  return std::make_unique<TileAndVectorizeWorkgroups>();
}

static PassRegistration<TileAndVectorizeWorkgroups> pass(
    "iree-codegen-linalg-to-llvm-workgroups-vectorization-pass",
    "Tile and vectorize llvm workgroups",
    [] { return std::make_unique<TileAndVectorizeWorkgroups>(); });

}  // namespace iree_compiler
}  // namespace mlir
