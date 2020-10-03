function update_obj!(problem, s, y, ∇fx, z, ∇fz, B, scheme, scale=nothing)
    fz, ∇fz = upto_gradient(problem, ∇fz, z)
    # add Project gradient

    # Update y
    @. y = ∇fz - ∇fx

    # Update B
    if scale == nothing
        if isa(scheme.approx, Direct)
            yBy = dot(y, B*y)
            if !iszero(yBy)
              Badj = dot(y, s)/yBy.*B
            end
        else
            ys = dot(y, s)
            if !iszero(ys)
                Badj = dot(y, B*y)/ys.*B
            else
                return fz, ∇fz, B, s, y
            end
        end
    else
        Badj = B
    end
    # Quasi-Newton update
    B = update!(scheme, Badj, s, y)

    return fz, ∇fz, B, s, y
end

function update_obj!(problem, s, y, ∇fx, z, ∇fz, B, scheme::Newton, scale=nothing)
    fz, ∇fz, B = upto_hessian(problem, ∇fz, B, z)

    return fz, ∇fz, B, s, s
end

function update_obj(problem, s, ∇fx, z, ∇fz, B, scheme, scale=nothing)
    fz, ∇fz = upto_gradient(problem, ∇fz, z)
    # add Project gradient

    # Update y
    y = ∇fz - ∇fx

    # Update B
    if scale == nothing
        if isa(scheme.approx, Direct)
            yBy = dot(y, B*y)
            if !iszero(yBy) && isfinite(yBy)
                Badj = dot(y, s)/yBy*B # this is different than above?
            else
                Badj = B
            end
        else
            ys = dot(y, s)
            if !iszero(ys) && isfinite(ys)
                Badj = dot(y, B*y)/ys*B
            else
                Badj = B
            end
        end
    else
         Badj = B
    end

    # Quasi-Newton update
    B = update(scheme, Badj, s, y)

    return fz, ∇fz, B, s, y
end

function update_obj(problem, s, ∇fx, z, ∇fz, B, scheme::Newton, is_first=nothing)
    fz, ∇fz, B = upto_hessian(problem, ∇fx, B, z)

    return fz, ∇fz, B, s, nothing
end
