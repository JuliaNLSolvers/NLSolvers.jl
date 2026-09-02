# Initial sizing scales the whole approximation by γ. The mutating drivers hand
# `update!` a dense B that it overwrites immediately after, so scale it in place
# when γ fits B's eltype. UniformScaling, static and other immutable B fall back
# to the allocating form, as does a real B with a complex γ.
_rescale!!(B, γ) = γ * B
function _rescale!!(B::Array, γ)
    if promote_type(typeof(γ), eltype(B)) === eltype(B)
        return rmul!(B, γ)
    end
    return γ * B
end

function update_obj!(problem, s, y, ∇fx, z, ∇fz, B, scheme, scale, dφ0)
    fz, ∇fz = upto_gradient(problem, ∇fz, z)
    @. y = ∇fz - ∇fx

    # Check PD skip condition (dφ0 == nothing means no skip check)
    if dφ0 !== nothing && should_skip(qn_skip(scheme), s, y, skip_aux(qn_skip(scheme), dφ0, ∇fx))
        return fz, ∇fz, B, s, y
    end

    # Initial Hessian sizing (the scheme picks ShannoPhua, OrenLuenberger, …)
    if scale == nothing
        γ = qn_scaling(scheme)(scheme.approx, s, y, B)
        if !isfinite(γ) || iszero(γ)
            return fz, ∇fz, B, s, y
        end
        Badj = _rescale!!(B, γ)
    else
        Badj = B
    end
    B = update!(scheme, Badj, s, y)
    return fz, ∇fz, B, s, y
end

function update_obj!(problem, s, y, ∇fx, z, ∇fz, B, scheme::Newton, scale, dφ0)
    fz, ∇fz, B = upto_hessian(problem, ∇fz, B, z)
    return fz, ∇fz, B, s, s
end

# The trust-region driver splits the trial-point evaluation from the model
# update: it needs the objective value before it can decide acceptance, and
# whether the approximation update runs depends on that decision and on the
# TrustRegion policy flags.
function tr_trial_eval!(problem, z, ∇fz, B, scheme)
    fz, ∇fz = upto_gradient(problem, ∇fz, z)
    return fz, ∇fz, B
end
function tr_trial_eval!(problem, z, ∇fz, B, scheme::Newton)
    fz, ∇fz, B = upto_hessian(problem, ∇fz, B, z)
    return fz, ∇fz, B
end

function tr_update_approx!(y, s, ∇fx, ∇fz, B, scheme, scale)
    @. y = ∇fz - ∇fx
    if scale == nothing
        γ = qn_scaling(scheme)(scheme.approx, s, y, B)
        if !isfinite(γ) || iszero(γ)
            return B, s, y
        end
        Badj = _rescale!!(B, γ)
    else
        Badj = B
    end
    B = update!(scheme, Badj, s, y)
    return B, s, y
end
# Newton's "model update" is the Hessian evaluation in tr_trial_eval!
tr_update_approx!(y, s, ∇fx, ∇fz, B, scheme::Newton, scale) = B, s, y

function update_obj(problem, s, ∇fx, z, ∇fz, B, scheme, scale, dφ0)
    fz, ∇fz = upto_gradient(problem, ∇fz, z)
    y = ∇fz - ∇fx

    # Check PD skip condition (dφ0 == nothing means no skip check)
    if dφ0 !== nothing && should_skip(qn_skip(scheme), s, y, skip_aux(qn_skip(scheme), dφ0, ∇fx))
        return fz, ∇fz, B, s, y
    end

    # Initial Hessian sizing (the scheme picks ShannoPhua, OrenLuenberger, …)
    if scale == nothing
        γ = qn_scaling(scheme)(scheme.approx, s, y, B)
        if !isfinite(γ) || iszero(γ)
            return fz, ∇fz, B, s, y
        end
        Badj = γ * B
    else
        Badj = B
    end
    B = update(scheme, Badj, s, y)
    return fz, ∇fz, B, s, y
end

function update_obj(problem, s, ∇fx, z, ∇fz, B, scheme::Newton, is_first, dφ0)
    fz, ∇fz, B = upto_hessian(problem, ∇fx, B, z)
    return fz, ∇fz, B, s, nothing
end
