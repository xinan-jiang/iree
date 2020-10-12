func @broadcast_sub_exp() -> tensor<1x4x384x384xf32> attributes { iree.module.export }  {
    %arg0 = iree.unfoldable_constant dense<1.0> : tensor<1x4x384x384xf32>
    %arg1 = iree.unfoldable_constant dense<1.0> : tensor<1x4x384xf32>
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1 = mhlo.subtract %arg0, %0 : tensor<1x4x384x384xf32>
    %2 = "mhlo.exponential"(%1) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    return %2 : tensor<1x4x384x384xf32>
}
