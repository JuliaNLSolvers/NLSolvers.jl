function solve(problem::LeastSquaresProblem, x, method::Union{Adam, LineSearch, APSO, TrustRegion}, options::LeastSquaresOptions)
    td = OptimizationProblem(problem.residuals, problem.bounds)
    res = solve(td, x, method, OptimizationOptions())
end