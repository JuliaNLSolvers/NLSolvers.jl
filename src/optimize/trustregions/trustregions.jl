abstract type TrustRegionUpdater end
struct TrustRegion{M,SP,D}
    scheme::M
    spsolve::SP
    Δupdate::D
end
summary(tr::TrustRegion) = "$(summary(modelscheme(tr))) with $(summary(algorithm(tr)))"
function initial_preconditioner(approach::TrustRegion, x)
    nothing
end
"""
  BTR() <: TrustRegionUpdater

Basic trust region updater following, and named after [CGT].
"""
struct BTR{T}
    Δmin::T
end
TrustRegion(; deltamin = nothing) = TrustRegion(Newton(), NTR(), BTR(deltamin))
TrustRegion(m, sp = NTR(); deltamin = nothing) = TrustRegion(m, sp, BTR(deltamin))
modelscheme(tr::TrustRegion) = tr.scheme
algorithm(tr::TrustRegion) = tr.spsolve

# annotate scheme here
solve(problem::OptimizationProblem, x0, scheme, options::OptimizationOptions) =
    solve(problem, (x0, nothing), TrustRegion(scheme, NWI()), options)

function solve(
    problem::OptimizationProblem,
    x0,
    scheme::Newton,
    options::OptimizationOptions,
)
    solve(problem, (x0, nothing), TrustRegion(scheme, NTR()), options)
end
function solve(
    problem::OptimizationProblem,
    x0,
    approach::TrustRegion,
    options::OptimizationOptions,
)
    solve(problem, (x0, nothing), approach, options)
end
include("optimize/inplace_loop.jl")
include("optimize/outofplace_loop.jl")
