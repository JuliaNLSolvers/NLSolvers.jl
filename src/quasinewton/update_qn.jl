function update_obj!(problem, s, y, ∇fx, z, ∇fz, B, scheme, scale = nothing; skip_data = nothing)
    fz, ∇fz = upto_gradient(problem, ∇fz, z)
    # add Project gradient

    # Update y
    @. y = ∇fz - ∇fx

    # Check skip condition
    if skip_data !== nothing && should_skip(qn_skip(scheme), s, y, skip_data)
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
    # Quasi-Newton update
    B = update!(scheme, Badj, s, y)

    return fz, ∇fz, B, s, y
end

function update_obj!(problem, s, y, ∇fx, z, ∇fz, B, scheme::Newton, scale = nothing; skip_data = nothing)
    fz, ∇fz, B = upto_hessian(problem, ∇fz, B, z)

    return fz, ∇fz, B, s, s
end

function update_obj(problem, s, ∇fx, z, ∇fz, B, scheme, scale = nothing; skip_data = nothing)
    fz, ∇fz = upto_gradient(problem, ∇fz, z)
    # add Project gradient

    # Update y
    y = ∇fz - ∇fx

    # Check skip condition
    if skip_data !== nothing && should_skip(qn_skip(scheme), s, y, skip_data)
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

    # Quasi-Newton update
    B = update(scheme, Badj, s, y)

    return fz, ∇fz, B, s, y
end

function update_obj(problem, s, ∇fx, z, ∇fz, B, scheme::Newton, is_first = nothing; skip_data = nothing)
    fz, ∇fz, B = upto_hessian(problem, ∇fx, B, z)

    return fz, ∇fz, B, s, nothing
end
