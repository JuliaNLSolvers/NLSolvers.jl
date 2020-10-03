# These should be solvable by optimizers (by ssr) and rootfinders (by subtraction)
# it's more of a convenience functin than anything
struct FixedPointProblem{O<:ObjWrapper, Opt}
  mapping::O
end
struct FixedPointOptions{Tint}
  maxiter::Tint
end
