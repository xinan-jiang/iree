flow.executable @ex0 {
  flow.dispatch.entry @dispatch0 attributes {workload = 4 : index}
  module {
    func @dispatch0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
func @entryFn() -> tensor<4xf32> {
  %input = iree.unfoldable_constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  %workload = constant 4 : index
  %0 = flow.dispatch @ex0::@dispatch0[%workload] (%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%0, dense<[-2.0, 4.0, -6.0, 8.0]> : tensor<4xf32>) : tensor<4xf32>
}
