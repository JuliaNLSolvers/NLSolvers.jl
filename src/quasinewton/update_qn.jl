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
        Badj = γ * B
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
