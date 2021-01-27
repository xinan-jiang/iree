func @entryFn() attributes { iree.module.export } {
  %input0 = iree.unfoldable_constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  %input1 = iree.unfoldable_constant dense<[0.5, 0.25, 0.5, 0.25]> : tensor<4xf32>
  %workload_x = constant 2 : index
  %workload_y = constant 1 : index
  %workload_z = constant 1 : index

  // Dispatch X*Y*Z workgroups:
  %output0 = flow.dispatch.workgroups[%workload_x, %workload_y, %workload_z](
      %input0, %input1) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>) =
      (%arg0 : !flow.dispatch.input<4xf32>,
       %arg1 : !flow.dispatch.input<4xf32>,
       %ret0 : !flow.dispatch.output<4xf32>) {
    // Access the workgroup information (XYZ in grid, grid count, etc):
    %id_x = flow.dispatch.workgroup.id[0] : index
    %id_y = flow.dispatch.workgroup.id[1] : index
    %id_z = flow.dispatch.workgroup.id[2] : index
    %count_x = flow.dispatch.workgroup.count[0] : index
    %count_y = flow.dispatch.workgroup.count[1] : index
    %count_z = flow.dispatch.workgroup.count[2] : index
    %size_x = flow.dispatch.workgroup.size[0] : index
    %size_y = flow.dispatch.workgroup.size[1] : index
    %size_z = flow.dispatch.workgroup.size[2] : index

    // If the I/Os have dynamic shapes you can query them like this:
    %arg0_shape = flow.dispatch.shape %arg0 : !flow.dispatch.input<4xf32> -> !shapex.ranked_shape<[4]>
    %arg0_dim0 = shapex.ranked_dim %arg0_shape[0] : !shapex.ranked_shape<[4]> -> index

    // Tensors are loaded either as a whole or as tiles. See FlowOps.td for
    // more information.
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %base_i = muli %id_x, %c2 : index
    %arg0_value = flow.dispatch.input.load %arg0,
        offsets = [%base_i], sizes = [%c2], strides = [%c1]
        : !flow.dispatch.input<4xf32> -> tensor<2xf32>
    %arg1_value = flow.dispatch.input.load %arg1,
        offsets = [%base_i], sizes = [%c2], strides = [%c1]
        : !flow.dispatch.input<4xf32> -> tensor<2xf32>

    // Your math here:
    // %ret0_value = "mhlo.multiply"(%arg0_value, %arg1_value) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

    // Stores (like loads) are either as a whole or as tiles. Of course, if
    // dispatching multiple workgroups you need to store back each section of
    // the output only once.
    flow.dispatch.output.store %arg0_value, %ret0,
        offsets = [%base_i], sizes = [%c2], strides = [%c1]
        : tensor<2xf32> -> !flow.dispatch.output<4xf32>

    flow.return
  }

  check.expect_almost_eq_const(
    %output0,
    //dense<[-1.0, 0.5, -1.5, 1.0]> : tensor<4xf32>
    dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  ) : tensor<4xf32>
  return
}
