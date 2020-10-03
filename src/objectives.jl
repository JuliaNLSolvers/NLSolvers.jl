abstract type ObjWrapper end

"""
    ScalarObjective

"""
struct ScalarObjective{Tf, Tg, Tfg, Tfgh, Th, Thv, Tbf, P}
    f::Tf
    g::Tg
    fg::Tfg
    fgh::Tfgh
    h::Th
    hv::Thv
    batched_f::Tbf
    param::P
end
ScalarObjective(f) = ScalarObjective(f, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
has_param(so::ScalarObjective) = so.param === nothing ? false : true
function value(so::ScalarObjective, x)
    if has_param(so)
        return so.f(x, so.param)
    else  
        return so.f(x)
    end
end
# need fall back for the case where fgh is not there
function upto_gradient(so::ScalarObjective, ∇f, x)
    if has_param(so)
        return so.fg(∇f, x, so.param)
    else
        return so.fg(∇f, x)
    end
end
function upto_hessian(so::ScalarObjective, ∇f, ∇²f, x)
    if has_param(so)
        return so.fgh(∇f, ∇²f, x, so.param)
    else
        return so.fgh(∇f, ∇²f, x)
    end
end
has_batched_f(so::ScalarObjective) = !(so.batched_f === nothing)
"""
    batched_value(obj, X)

Return the objective evaluated at all elements of X. If obj contains
a batched_f it will have X passed collectively, else f will be broadcasted
across the elements of X.
"""
function batched_value(so::ScalarObjective, X)
    if has_batched_f(so)
        if has_param(so)
            return so.batched_f(X, so.param)
        else
            return so.batched_f(X)
        end
    else
        if has_param(so)
            return so.f.(X, Ref(so.param))
        else
            return so.f.(X)
        end
    end
end
function batched_value(so::ScalarObjective, F, X)
    if has_batched_f(so)# add
        if has_param(so)
            return F = so.batched_f(F, X, so.param)
        else
            return F = so.batched_f(F, X)
        end
    else
        if has_param(so)
            F .= so.f.(X, Ref(so.param))
            return F
        else
            F .= so.f.(X, Ref(so.param))
            return F
        end
    end
end

## If prob is a NEqProblem, then we can just dispatch to least squares MeritObjective
# if fast JacVec exists then maybe even line searches that updates the gradient can be used??? 
struct LineObjective!{TP, T1, T2, T3}
    prob::TP
    ∇fz::T1
    z::T2
    x::T2
    d::T2
    φ0::T3
    dφ0::T3
end
function (le::LineObjective!)(λ)
    z = retract!(_manifold(le.prob), le.z, le.x, le.d, λ)
    ϕ = value(le.prob, z)
    (ϕ=ϕ, z=z)
end
function (le::LineObjective!)(λ, calc_grad::Bool)
    f, g = upto_gradient(le.prob, le.∇fz, retract!(_manifold(le.prob), le.z, le.x, le.d, λ))
    (ϕ=f, dϕ=real(dot(g, le.d))) # because complex dot might not have exactly zero im part and it's the wrong type
end
struct LineObjective{TP, T1, T2, T3}
    prob::TP
    ∇fz::T1
    z::T2
    x::T2
    d::T2
    φ0::T3
    dφ0::T3
end
function (le::LineObjective)(λ)
    z = retract(_manifold(le.prob), le.x, le.d, λ)
    _value = value(le.prob, z)
    if le.prob.objective isa MeritObjective
        return (ϕ = _value.ϕ, _value.Fx)
    end
    (ϕ = _value,)
end
function (le::LineObjective)(λ, calc_grad::Bool)
    f, g = upto_gradient(le.prob, le.∇fz, retract(_manifold(le.prob), le.x, le.d, λ))
    (ϕ=f, dϕ=real(dot(g, le.d))) # because complex dot might not have exactly zero im part and it's the wrong type
end

# We call real on dφ0 because x and df might be complex
_lineobjective(mstyle::InPlace, prob::AbstractProblem, ∇fz, z, x, d, φ0, dφ0) = LineObjective!(prob, ∇fz, z, x, d, φ0, real(dφ0))
_lineobjective(mstyle::OutOfPlace, prob::AbstractProblem, ∇fz, z, x, d, φ0, dφ0) = LineObjective(prob, ∇fz, z, x, d, φ0, real(dφ0))

struct MeritObjective{TP, T1, T2, T3, T4, T5}
  prob::TP
  F::T1
  FJ::T2
  Fx::T3
  Jx::T4
  d::T5
end
function value(mo::MeritObjective, x)
  Fx = mo.F(mo.Fx, x)
  (ϕ=(norm(Fx)^2)/2, Fx = Fx)
end

struct LsqWrapper{Tobj, TF, TJ} <: ObjWrapper
  R::Tobj
  F::TF
  J::TJ
end
function (lw::LsqWrapper)(x)
  F = lw.R(lw.F, x)
  sum(abs2, F)/2
end
function (lw::LsqWrapper)(∇f, x)
  _F, _J = lw.R(lw.F, lw.J, x)
  copyto!(∇f, sum(_J; dims=1))
  sum(abs2, _F), ∇f
end