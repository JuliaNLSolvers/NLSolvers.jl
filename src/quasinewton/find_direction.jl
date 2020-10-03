function find_direction(B, P, ∇f, scheme::QuasiNewton{<:Direct})
   return -(B\∇f)
end
function find_direction(B, P, ∇f, scheme::Newton{<:Direct})
   return scheme.linsolve(B, -∇f)
end
function find_direction(A, P, ∇f, scheme::QuasiNewton{<:Inverse})
   return -A*∇f
end
function find_direction!(d, B, P,∇f, scheme::QuasiNewton{<:Direct})
   d .= -(B\∇f)
   d
end
function find_direction!(d, B, P,∇f, scheme::Newton{<:Direct})
   scheme.linsolve(d, B, -∇f)
end
function find_direction!(d, A, P, ∇f, scheme::QuasiNewton{<:Inverse})
   rmul!(mul!(d, A, ∇f), -1)
   d
end
function find_direction(B, P, ∇f, scheme::GradientDescent)
   -apply_preconditioner(OutOfPlace(), P, nothing, ∇f)
end
function find_direction!(d, B, P,∇f, scheme::GradientDescent)
   d = apply_preconditioner(InPlace(), P, d, ∇f)
   d .= .-d
end
