#include "iree/compiler/Dialect/Ruy/IR/RuyDialect.h"

#include "iree/compiler/Dialect/Ruy/IR/RuyOps.h"

namespace mlir {
namespace iree_compiler {
namespace ruy {
RuyDialect::RuyDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<RuyDialect>()) {
  // context->loadDialect<RuyDialect>();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Ruy/IR/RuyOps.cpp.inc"
      >();
}
}  // namespace ruy
}  // namespace iree_compiler
}  // namespace mlir
