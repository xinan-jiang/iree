func @tensor() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "mhlo.sqrt"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 1.4142, 1.7321, 2.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func @scalar() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<16.0> : tensor<f32>
  %result = "mhlo.sqrt"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<4.0> : tensor<f32>) : tensor<f32>
  return
}
