// RUN: iree-opt -iree-codegen-convert-to-llvm-2 -split-input-file %s | IreeFileCheck %s

// CHECK_LABEL: @convert_dynamic_shape
func @convert_dynamic_shape() -> f32 {
  %c0 = constant 0 : index
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x?xf32>
  %1 = hal.interface.load.constant offset = 0 : index
  %2 = hal.interface.load.constant offset = 1 : index
  %3 = shapex.make_ranked_shape %1, %2 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %6 = shapex.tie_shape %0, %3 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
  %7 = load %6[%c0, %c0] : memref<?x?xf32>
  return %7 : f32
}
hal.interface @legacy_io attributes {push_constants = 2 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Write"
}
// CHECK: llvm.func @convert_dynamic_shape(%[[ARG0:.+]]: !llvm.ptr<ptr<i8>>, %[[ARG1:.+]]: !llvm.ptr<i32>, %[[WORKGROUP_ID:.+]]: !llvm.ptr<i32>, %[[WORKGROUP_COUNT:.+]]: !llvm.ptr<i32>, %[[WORKGROUP_SIZE:.+]]: !llvm.ptr<i32>)
// CHECK: %[[PACKED_ARGS_PTR:.+]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<ptr<i8>> to !llvm.ptr<struct<(ptr<f32>)>>
// CHECK: %[[PACKED_ARGS:.+]] = llvm.load %[[PACKED_ARGS_PTR]] : !llvm.ptr<struct<(ptr<f32>)>>
// CHECK: %[[MEMREF0_DATA_PTR:.+]] = llvm.extractvalue %[[PACKED_ARGS]][0] : !llvm.struct<(ptr<f32>)>
// CHECK: %[[MEMREF0:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_0:.+]] = llvm.insertvalue %[[MEMREF0_DATA_PTR]], %[[MEMREF0]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_1:.+]] = llvm.insertvalue %[[MEMREF0_DATA_PTR]], %[[MEMREF0_0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST0:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[DIM0_PTR:.+]] = llvm.getelementptr %[[ARG1]][%[[CONST0]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK: %[[DIM0:.+]] = llvm.load %[[DIM0_PTR]] : !llvm.ptr<i32>
// CHECK: %[[DIM0CASTED:.+]] = llvm.zext %[[DIM0]] : i32 to i64
// CHECK: %[[CONST1:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[DIM1_PTR:.+]] = llvm.getelementptr %[[ARG1]][%[[CONST1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK: %[[DIM1:.+]] = llvm.load %[[DIM1_PTR]] : !llvm.ptr<i32>
// CHECK: %[[DIM1CASTED:.+]] = llvm.zext %[[DIM1]] : i32 to i64
// CHECK: %[[MEMREF0_2:.+]] = llvm.insertvalue %[[DIM0CASTED]], %[[MEMREF0_1]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_3:.+]] = llvm.insertvalue %[[DIM1CASTED]], %[[MEMREF0_2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST1_STRIDE:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[MEMREF0_4:.+]] = llvm.insertvalue %[[CONST1_STRIDE]], %[[MEMREF0_3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE_DIM1:.+]] = llvm.extractvalue %[[MEMREF0_4]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DIM1_0:.+]] = llvm.extractvalue %[[MEMREF0_4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE_DIM0:.+]] = llvm.mul %[[STRIDE_DIM1]], %[[DIM1_0]] : i64
// CHECK: llvm.insertvalue %[[STRIDE_DIM0]], %[[MEMREF0_4]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

// -----

// CHECK_LABEL: @convert_dynamic_shape2
func @convert_dynamic_shape2() -> f32 {
  %c0 = constant 0 : index
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io2::@arg0} : memref<2x?xf32>
  %1 = hal.interface.load.constant offset = 0 : index
  %2 = shapex.make_ranked_shape %1 : (index) -> !shapex.ranked_shape<[2,?]>
  %3 = shapex.tie_shape %0, %2 : memref<2x?xf32>, !shapex.ranked_shape<[2,?]>
  %4 = load %3[%c0, %c0] : memref<2x?xf32>
  return %4 : f32
}
hal.interface @legacy_io2 attributes {push_constants = 1 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
}
// CHECK: llvm.func @convert_dynamic_shape2(%[[ARG0:.+]]: !llvm.ptr<ptr<i8>>, %[[ARG1:.+]]: !llvm.ptr<i32>, %[[WORKGROUP_ID:.+]]: !llvm.ptr<i32>, %[[WORKGROUP_COUNT:.+]]: !llvm.ptr<i32>, %[[WORKGROUP_SIZE:.+]]: !llvm.ptr<i32>)
// CHECK: %[[PACKED_ARGS_PTR:.+]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<ptr<i8>> to !llvm.ptr<struct<(ptr<f32>)>>
// CHECK: %[[PACKED_ARGS:.+]] = llvm.load %[[PACKED_ARGS_PTR]] : !llvm.ptr<struct<(ptr<f32>)>>
// CHECK: %[[MEMREF0_DATA_PTR:.+]] = llvm.extractvalue %[[PACKED_ARGS]][0] : !llvm.struct<(ptr<f32>)>
// CHECK: %[[MEMREF0:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_0:.+]] = llvm.insertvalue %[[MEMREF0_DATA_PTR]], %[[MEMREF0]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_1:.+]] = llvm.insertvalue %[[MEMREF0_DATA_PTR]], %[[MEMREF0_0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST0:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[DIM0_PTR:.+]] = llvm.getelementptr %[[ARG1]][%[[CONST0]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK: %[[DIM0:.+]] = llvm.load %[[DIM0_PTR]] : !llvm.ptr<i32>
// CHECK: %[[DIM0CASTED:.+]] = llvm.zext %[[DIM0]] : i32 to i64
// CHECK: %[[CONST0_2:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[CONST2:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK: %[[MEMREF1:.+]] = llvm.insertvalue %[[CONST2:.+]], %[[MEMREF0_1:.+]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF2:.+]] = llvm.insertvalue %[[DIM0CASTED:.+]], %[[MEMREF1:.+]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST1:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[MEMREF3:.+]] = llvm.insertvalue %[[CONST1:.+]], %[[MEMREF2:.+]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE_DIM1:.+]] = llvm.extractvalue %[[MEMREF3:.+]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DIM1_0:.+]] = llvm.extractvalue %[[MEMREF3:.+]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE_DIM0:.+]] = llvm.mul %[[STRIDE_DIM1:.+]], %[[DIM1_0:.+]]: i64
// CHECK: %[[INSERT_DIM0:.+]] = llvm.insertvalue %[[STRIDE_DIM0:.+]], %[[MEMREF3:.+]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[EXTRACT1:.+]] = llvm.extractvalue %[[INSERT_DIM0:.+]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[EXTRACT2:.+]] = llvm.extractvalue %[[INSERT_DIM0:.+]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MUL1:.+]] = llvm.mul %[[CONST0_2]], %[[EXTRACT2]] : i64
// CHECK: %[[ADD1:.+]] = llvm.add %[[MUL1]], %[[CONST0_2]] : i64
// CHECK: %[[GET_PTR:.+]] = llvm.getelementptr %[[EXTRACT1:.+]][%[[ADD2:.+]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[LOAD:.+]] = llvm.load %[[GET_PTR:.+]] : !llvm.ptr<f32>

// -----

// CHECK_LABEL: @distribute_lookup
func @distribute_lookup() -> f32 {
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io3::@arg0} : memref<2x2x2xf32>
  %1 = hal.interface.workgroup.id[0] : index
  %2 = hal.interface.workgroup.id[1] : index
  %3 = hal.interface.workgroup.id[2] : index
  %4 = load %0[%1, %2, %3] : memref<2x2x2xf32>
  return %4 : f32
}
hal.interface @legacy_io3 attributes {push_constants = 1 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
}
