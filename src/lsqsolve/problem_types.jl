"""
  LeastSquaresProblem(residuals::VectorObjective)

A LeastSquaresProblem is used to represent the
mathematical problem of finding the smallest possible sum of squared values of
the elements of the residual function. The problem is defined by `residual` which is
a `VectorObjective`.

Options are stored in `options` and are of the `NLsqOptions` type. See more information
about options using `?NLsqOptions`.

The package NLSolversAD.jl adds automatic conversion of problems to match algorithms
that require higher order derivates than provided by the user. It also adds AD
constructors for a target number of derivatives.
"""
struct LeastSquaresProblem{R, B, M, C}
  residuals::R
  bounds::B
  manifold::M
  constraints::C
end
function LeastSquaresProblem(residuals, bounds = nothing)
  manifold = Euclidean(0)
  LeastSquaresProblem{typeof(residuals), typeof(bounds), typeof(manifold), Nothing}(residuals, bounds, manifold, nothing)
end

struct LeastSquaresOptions{Tmi}
  maxiter::Tmi
end

# observations are useful if you only want to specify it as a ScalarObjective
function solve(problem::LeastSquaresProblem, x, method, options::LeastSquaresOptions)
  _error = ArgumentError("solve not implemented for $(summary(method)) and LeastSquaresProblem.")
  throw(_error)
end