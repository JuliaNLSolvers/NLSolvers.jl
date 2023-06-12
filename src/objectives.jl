abstract type ObjWrapper end

"""
    ScalarObjective

"""
struct ScalarObjective{Tf,Tg,Tfg,Tfgh,Th,Thv,Tbf,P}
    f::Tf
    g::Tg
    fg::Tfg
    fgh::Tfgh
    h::Th
    hv::Thv
    batched_f::Tbf
    param::P
end
ScalarObjective(; f = nothing, g = nothing, fg = nothing, fgh = nothing, h = nothing) =
    ScalarObjective(f, g, fg, fgh, h, nothing, nothing, nothing)
has_param(so::ScalarObjective) = so.param === nothing ? false : true
function value(so::ScalarObjective, x)
    if has_param(so)
        return so.f(x, so.param)
    else
        return so.f(x)
    end
end
# need fall back for the case where fg is not there
function upto_gradient(so::ScalarObjective, ∇f, x)
    if has_param(so)
        if so.fg === nothing
            return so.f(x, so.param), so.g(∇f, x, so.param)
        else
            return so.fg(∇f, x, so.param)
        end
    else
        if so.fg === nothing
            return so.f(x), so.g(∇f, x)
        else
            return so.fg(∇f, x)
        end
    end
end
# need fall back for the case where fgh is not there
function upto_hessian(so::ScalarObjective, ∇f, ∇²f, x)
    if has_param(so)
        if so.fgh === nothing
            return so.f(x, so.param), so.g(∇f, x, so.param), so.h(∇²f, x, so.param)
        else
            return so.fgh(∇f, ∇²f, x, so.param)
        end
    else
        if so.fgh === nothing
            return so.f(x), so.g(∇f, x), so.h(∇²f, x)
        else
            return so.fgh(∇f, ∇²f, x)
        end
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
            F .= so.f.(X)
            return F
        end
    end
end

## If prob is a NEqProblem, then we can just dispatch to least squares MeritObjective
# if fast JacVec exists then maybe even line searches that updates the gradient can be used??? 
ls_has_gradient(prob) = nothing
ls_upto_gradient(prob,∇f,x) = upto_gradient(prob,∇f,x)
ls_value(prob,z) = (value(prob,z),z)

struct LineObjective!{TP,T1,T2,T3}
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
    ϕ,z = ls_value(le.prob, z)
    return (ϕ = ϕ, z = z)
end

function (le::LineObjective!)(λ, calc_grad::Bool)
    ls_has_gradient(le.prob)
    f, g = ls_upto_gradient(le.prob, le.∇fz, retract!(_manifold(le.prob), le.z, le.x, le.d, λ))
    (ϕ = f, dϕ = real(dot(g, le.d))) # because complex dot might not have exactly zero im part and it's the wrong type
end

struct LineObjective{TP,T1,T2,T3}
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
    ϕ,z = ls_value(le.prob, z)
    return (ϕ = ϕ, z = z)
end

function (le::LineObjective)(λ, calc_grad::Bool)
    ls_has_gradient(le.prob)
    f, g = ls_upto_gradient(le.prob, le.∇fz, retract(_manifold(le.prob), le.x, le.d, λ))
    (ϕ = f, dϕ = real(dot(g, le.d))) # because complex dot might not have exactly zero im part and it's the wrong type
end

# We call real on dφ0 because x and df might be complex
_lineobjective(mstyle::InPlace, prob::AbstractProblem, ∇fz, z, x, d, φ0, dφ0) =
    LineObjective!(prob, ∇fz, z, x, d, φ0, real(dφ0))
_lineobjective(mstyle::OutOfPlace, prob::AbstractProblem, ∇fz, z, x, d, φ0, dφ0) =
    LineObjective(prob, ∇fz, z, x, d, φ0, real(dφ0))

struct MeritObjective{TP,T1,T2,T3,T4,T5}
    prob::TP
    F::T1
    FJ::T2
    Fx::T3
    Jx::T4
    d::T5
end
function value(mo::MeritObjective, x)
    Fx = mo.F(mo.Fx, x)
    return norm(Fx)^2 / 2
    #(ϕ = (norm(Fx)^2) / 2, Fx = Fx)
end

function ls_value(mo::MeritObjective, x)
    Fx = mo.F(mo.Fx, x)
    return (norm(Fx)^2 / 2,Fx)
end

struct LsqWrapper{Tobj,TF,TJ} <: ObjWrapper
    R::Tobj
    F::TF
    J::TJ
end
function (lw::LsqWrapper)(x)
    F = lw.R(lw.F, x)
    sum(abs2, F) / 2
end
function (lw::LsqWrapper)(∇f, x)
    _F, _J = lw.R(lw.F, lw.J, x)
    copyto!(∇f, sum(_J; dims = 1))
    sum(abs2, _F), ∇f
end

struct LeastSquaresObjective{TFx,TJx,Tf,Tfj,Td}
    Fx::TFx
    Jx::TJx
    F::Tf
    FJ::Tfj
    ydata::Td
end
has_ydata(lsq::LeastSquaresObjective) = !(lsq.ydata === nothing)
function value(lsq::LeastSquaresObjective, x)
    # Evaluate the residual system or the "predicted" value for LeastSquaresObjective
    Fx = lsq.F(lsq.Fx, x)

    # If this comes from a LeastSquaresProblem there will be a lhs to subtract
    if has_ydata(lsq)
        Fx .= Fx .- lsq.ydata
    end
    lsq.Fx .= Fx

    # Least Squares
    f = (norm(Fx)^2) / 2
    return f
end
function batched_value(lsq::LeastSquaresObjective, F, X)
    F .= value.(Ref(lsq), X)
end
function upto_gradient(lsq::LeastSquaresObjective, Fx, x)
    # Evaluate the residual system or the "predicted" value for LeastSquares
    # and the Jacobian of either one
    Fx_sq, Jx_sq = lsq.FJ(lsq.Fx, lsq.Fx * x', x)

    # If this comes from a LeastSquaresProblem there will be a lhs to subtract
    if has_ydata(lsq)
        Fx_sq .= Fx_sq .- lsq.ydata
    end
    lsq.Fx .= Fx_sq

    # Least Squares
    f = (norm(Fx_sq)^2) / 2
    Fx .= Jx_sq' * Fx_sq
    return f, Fx
end
function upto_hessian(lsq::LeastSquaresObjective, Fx, Jx, x)  #Fx is the gradient and Jx is the Hessian
    Fx_sq, Jx_sq = lsq.FJ(lsq.Fx, lsq.Jx, x)

    lsq.Fx .= Fx_sq
    f = (norm(Fx)^2) / 2
    # this is the gradient
    Fx .= Jx_sq' * Fx_sq
    # As you may notice, this can be expensive... Because the gradient
    # is going to be very simple. May want to create a
    # special type or way to hook into trust regions here. We can exploit
    # that we only need the cauchy and the newton steps, not any shifted
    # systems. There is no need to get the full hessian. because these two
    # steps are don't need these multiplies
    # This is the Hessian
    Jx .= Jx_sq' * Jx_sq
    return f, Fx, Jx
end
