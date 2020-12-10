// RUN: iree-opt -split-input-file --convert-linalg-matmul-to-ruy-padded-matmul -verify-diagnostics %s | IreeFileCheck %s

func @linalg_static_memrefs(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>, %out: memref<2x2xf32>) {
    linalg.matmul ins(%lhs, %rhs : memref<2x2xf32>, memref<2x2xf32>)
                 outs(%out : memref<2x2xf32>)
    return
}
// CHECK-LABEL: linalg_static_memrefs
// CHECK: ruy.padded_matmul
