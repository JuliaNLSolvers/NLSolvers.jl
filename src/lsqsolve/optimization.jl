function solve(problem::LeastSquaresProblem, x, method::LineSearch, options::LeastSquaresOptions)
    Fx_outer = copy(x)
    x_outer = copy(x)
    normed_residual = NormedResiduals(x_outer, Fx_outer, problems.residuals)

    td = OptimizationProblem(normed_residual)
    res = solve(td, x, method, OptimizationOptions())
end