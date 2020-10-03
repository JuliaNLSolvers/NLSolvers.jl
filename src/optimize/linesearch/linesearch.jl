"""
  LineSearch(scheme, linesearcher)
"""
struct LineSearch{S, LS, K}
  scheme::S
  linesearcher::LS
  scaling::K
end
LineSearch() = LineSearch(DBFGS(), Backtracking(), InitialScaling(ShannoPhua()))
LineSearch(m) = LineSearch(m, Backtracking(), InitialScaling(ShannoPhua()))
LineSearch(m, ls) = LineSearch(m, ls, InitialScaling(ShannoPhua()))

hasprecon(ls::LineSearch) = hasprecon(modelscheme(ls))
summary(ls::LineSearch) = summary(modelscheme(ls))*" with "*summary(algorithm(ls))

function initial_preconditioner(approach::LineSearch, x)
  method = modelscheme(approach)
  initial_preconditioner(method, x, hasprecon(method))
end

modelscheme(ls::LineSearch) = ls.scheme
algorithm(ls::LineSearch) = ls.linesearcher
include("conjugategradient.jl")
export ConjugateGradient

include("quasinewton.jl")
include("limitedquasinewton.jl")