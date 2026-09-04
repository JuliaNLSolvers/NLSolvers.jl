# Initial sizing scales the whole approximation by γ, once per solve, and
# `update!` consumes the result immediately.
#
# The caller says whether it owns B, as it does for every other mutating helper
# here. It says nothing about what B is, and nothing needs to: whatever B is,
# `update!` is about to write into it under the same style, so a B that cannot
# be scaled in place cannot be updated in place either, and this fails where
# that would. Broadcasting rather than `lmul!` keeps it to `setindex!`, which is
# what the updates need too, and off the scalar indexing that array types on a
# device reject. Assigning into B also fixes its element type: γ is B's element
# type by construction, and one that cannot be represented throws here instead
# of silently widening B from one iteration to the next.
_rescale!!(::OutOfPlace, B, γ::Number) = γ * B
_rescale!!(::InPlace, B, γ::Number) = B .= γ .* B

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
        Badj = _rescale!!(mstyle(problem), B, γ)
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

function tr_update_approx!(mstyle, y, s, ∇fx, ∇fz, B, scheme, scale)
    @. y = ∇fz - ∇fx
    if scale == nothing
        γ = qn_scaling(scheme)(scheme.approx, s, y, B)
        if !isfinite(γ) || iszero(γ)
            return B, s, y
        end
        Badj = _rescale!!(mstyle, B, γ)
    else
        Badj = B
    end
    B = update!(scheme, Badj, s, y)
    return B, s, y
end
# Newton's "model update" is the Hessian evaluation in tr_trial_eval!
tr_update_approx!(mstyle, y, s, ∇fx, ∇fz, B, scheme::Newton, scale) = B, s, y

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
