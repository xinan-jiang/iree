hal.executable @executable_0 {
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.interface @io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.entry_point @dispatch_0 attributes {
      interface = @io,
      ordinal = 0 : i32,
      signature = (!flow.dispatch.input<4xf32>, !flow.dispatch.input<4xf32>, !flow.dispatch.output<4xf32>) -> (),
      workgroup_size = [1 : index, 1 : index, 1 : index]
    }
    module {
      // TODO(benvanik): remove this duplication (it's a symbol lookup issue):
      hal.interface @io {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
      func @dispatch_0() {
        // I/O buffers are accessed via the hal.interface:
        %c0 = constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4xf32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<4xf32>
        %ret0 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4xf32>

        // Access the workgroup information (XYZ in grid, grid count, etc):
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %count_x = hal.interface.workgroup.count[0] : index
        %count_y = hal.interface.workgroup.count[1] : index
        %count_z = hal.interface.workgroup.count[2] : index
        %size_x = hal.interface.workgroup.size[0] : index
        %size_y = hal.interface.workgroup.size[1] : index
        %size_z = hal.interface.workgroup.size[2] : index

        // Dynamic shapes are trickier - don't use this and instead use a
        // flow.dispatch.workgroups op as it takes care of marshaling dynamic
        // dimensions across the HAL boundary.

        // Loads are tiled and we need to calculate that from the workgroup ID:
        %c1 = constant 1 : index
        %c2 = constant 2 : index
        %base_i = muli %workgroup_id_x, %c2 : index
        %arg0_tile = subview %arg0[%base_i] [2] [1] : memref<4xf32> to memref<2xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
        %arg1_tile = subview %arg1[%base_i] [2] [1] : memref<4xf32> to memref<2xf32, affine_map<(d0)[s0] -> (d0 + s0)>>

        // Stores are also tiled, and you can store directly into the buffers:
        %ret0_tile = subview %ret0[%base_i] [2] [1] : memref<4xf32> to memref<2xf32, affine_map<(d0)[s0] -> (d0 + s0)>>

        // YOUR MATH HERE (produce some tile)
        scf.for %i = %c0 to %c2 step %c1 {
          %t = load %arg0_tile[%i] : memref<2xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
          store %t, %ret0_tile[%i] : memref<2xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
        }

        return
      }
    }
  }
}
func @entryFn() attributes { iree.module.export } {
  %input0 = iree.unfoldable_constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  %input1 = iree.unfoldable_constant dense<[0.5, 0.25, 0.5, 0.25]> : tensor<4xf32>
  %workload_x = constant 2 : index
  %workload_y = constant 1 : index
  %workload_z = constant 1 : index
  %output0 = flow.dispatch @executable_0::@llvm_aot::@dispatch_0[%workload_x, %workload_y, %workload_z] (%input0, %input1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(
    %output0,
    //dense<[-1.0, 0.5, -1.5, 1.0]> : tensor<4xf32>
    dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  ) : tensor<4xf32>
  return
}
