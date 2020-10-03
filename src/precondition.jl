#===============================================================================
  Preconditioning

  Todo: maybe add an inverse?
===============================================================================#

struct NoPrecon end
struct HasPrecon end
function initial_preconditioner(method, x, ::NoPrecon)
  nothing
end
function initial_preconditioner(method, x, ::HasPrecon)
  method.P(x)
end
update_preconditioner(method, x, P) = update_preconditioner(method, x, P, hasprecon(method))
update_preconditioner(method, x, P, ::NoPrecon) = nothing
update_preconditioner(method, x, P::Nothing, ::HasPrecon) = method.P(x)
update_preconditioner(method, x, P, ::HasPrecon) = method.P(x, P)
apply_preconditioner(mstyle::InPlace, ::Nothing, Pg, ∇f) = copyto!(Pg, ∇f)
apply_preconditioner(mstyle::InPlace, P, Pg, ∇f) = mul!(Pg, P, ∇f)
apply_preconditioner(mstyle::OutOfPlace, ::Nothing, Pg, ∇f) = ∇f
apply_preconditioner(mstyle::OutOfPlace, P, Pg, ∇f) = P*∇f
apply_inverse_preconditioner(mstyle::InPlace, ::Nothing, Pg, ∇f) = copyto!(Pg, ∇f)
apply_inverse_preconditioner(mstyle::InPlace, P, Pg, ∇f) = ldiv!(Pg, factorize(P), ∇f) # add special for Diagonal?
function apply_inverse_preconditioner(mstyle::InPlace, P::Diagonal, Pg, ∇f)
	Pg .= P.diag.\∇f
end
apply_inverse_preconditioner(mstyle::OutOfPlace, ::Nothing, Pg, ∇f) = ∇f
apply_inverse_preconditioner(mstyle::OutOfPlace, P, pg, ∇f) = P\∇f

