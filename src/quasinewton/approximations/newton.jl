# The simple
struct Newton{T1, Tlin, Tage, TFact} <: QuasiNewton{T1}
   approx::T1
   linsolve::Tlin
   reset_age::Tage
   factorizer::TFact
end
hasprecon(::Newton) = NoPrecon()
# struct DefaultSequence end
default_newton_linsolve(B::Number, g) = B\g
function default_newton_linsolve(B, g)
  B\g
end
function default_newton_linsolve(d, B, g)
  d .= (B\g)
end
Newton(;approx=Direct(), linsolve=default_newton_linsolve, reset_age=nothing) = Newton(approx, linsolve, reset_age, nothing)
summary(::Newton{<:Direct, typeof(default_newton_linsolve)}) = "Newton's method with default linsolve"
summary(::Newton{<:Direct, <:Any}) = "Newton's method with user-supplied linsolve"