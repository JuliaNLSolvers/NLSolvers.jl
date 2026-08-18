"""
  LineSearch(scheme, linesearcher)
"""
struct LineSearch{S,LS,K}
    scheme::S
    linesearcher::LS
    scaling::K
end
LineSearch() = LineSearch(DBFGS(), Backtracking(), InitialScaling(ShannoPhua()))
# A `nothing` line searcher is resolved against the problem type when solve is
# called, see resolve_linesearch.
LineSearch(m) = LineSearch(m, nothing, InitialScaling(ShannoPhua()))
LineSearch(m, ls) = LineSearch(m, ls, InitialScaling(ShannoPhua()))

# Resolve a `nothing` line searcher against the problem type. Called at the
# start of solve, so both the line search loop and the reported method see the
# resolved line searcher. Explicitly chosen line searchers pass through.
resolve_linesearch(approach::LineSearch, prob) = LineSearch(
    modelscheme(approach),
    resolve_linesearch(algorithm(approach), prob),
    approach.scaling,
)
resolve_linesearch(ls, prob) = ls
resolve_linesearch(::Nothing, prob::OptimizationProblem) = HZAW()

hasprecon(ls::LineSearch) = hasprecon(modelscheme(ls))
summary(ls::LineSearch) = summary(modelscheme(ls)) * " with " * summary(algorithm(ls))

function initial_preconditioner(approach::LineSearch, x)
    method = modelscheme(approach)
    initial_preconditioner(method, x, hasprecon(method))
end

modelscheme(ls::LineSearch) = ls.scheme
algorithm(ls::LineSearch) = ls.linesearcher
include("conjugategradient.jl")

include("quasinewton.jl")
include("limitedquasinewton.jl")
