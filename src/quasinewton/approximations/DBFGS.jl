struct DBFGS{T1,T2,T3,Tscaling} <: QuasiNewton{T1}
    approx::T1
    theta::T2
    P::T3
    scaling::Tscaling
end
DBFGS(approx) = DBFGS(approx, 0.2, nothing, ShannoPhua())
DBFGS(; inverse = true, theta = 0.2, scaling = ShannoPhua()) =
    DBFGS(inverse ? Inverse() : Direct(), theta, nothing, scaling)
hasprecon(::DBFGS{<:Any,<:Any,<:Nothing}) = NoPrecon()
hasprecon(::DBFGS{<:Any,<:Any,<:Any}) = HasPrecon()
qn_scaling(scheme::DBFGS) = scheme.scaling
summary(dbfgs::DBFGS{Inverse}) = "Inverse Damped BFGS"
summary(dbfgs::DBFGS{Direct}) = "Direct Damped BFGS"

function update!(scheme::DBFGS{<:Direct,<:Any,<:Any}, B, s, y)
    # We could write this as
    #     B .+= (y*y')/dot(s, y) - (B*s)*(s'*B)/(s'*B*s)
    #     B .+= (y*y')/dot(s, y) - b*b'/dot(s, b)
    # where b = B*s
    # But instead, we split up the calculations. First calculate the denominator
    # in the first term
    σ = dot(s, y)
    ρ = inv(σ) # scalar
    # Then calculate the vector b
    b = B * s # vector temporary
    sb = dot(s, b)
    if σ ≥ scheme.theta * sb
        θ = 1.0
        # Calculate one vector divided by dot(s, b)
        ρbb = inv(sb) * b
        # And calculate
        B .+= (inv(σ) * y) * y' .- ρbb * b'
    else
        θ = 0.8 * sb / (sb - σ)
        r = y * θ + (1 - θ) * b
        # Calculate one vector divided by dot(s, b)
        ρbb = inv(dot(s, b)) * b
        # And calculate
        B .+= (inv(dot(s, r)) * r) * r' .- ρbb * b'
    end
end
function update(scheme::DBFGS{<:Direct,<:Any}, B, s, y)
    # As above, but out of place

    σ = dot(s, y)
    b = B * s
    sb = dot(s, b)
    if σ ≥ scheme.theta * sb
        θ = 1.0
        # Calculate one vector divided by dot(s, b)
        ρbb = inv(sb) * b
        # And calculate
        return B .+ (inv(σ) * y) * y' .- ρbb * b'
    else
        θ = 0.8 * sb / (sb - σ)
        r = y * θ + (1 - θ) * b
        # Calculate one vector divided by dot(s, b)
        ρbb = inv(dot(s, b)) * b
        # And calculate
        return B = B .+ (inv(dot(s, r)) * r) * r' .- ρbb * b'
    end
end
# For the inverse form, we need Bs = H\s to evaluate the damping condition.
# Powell (1978) damping: if s'y < θ·s'Bs, replace y with ȳ = θ_d·y + (1-θ_d)·Bs
# where θ_d = 0.8·s'Bs / (s'Bs - s'y), ensuring s'ȳ ≥ θ·s'Bs > 0.
function _damp_y(scheme::DBFGS, H, s, y)
    Bs = H \ s
    σ = dot(s, y)
    sb = dot(s, Bs)
    if σ ≥ scheme.theta * sb
        return y
    else
        θ_d = oftype(σ, 0.8) * sb / (sb - σ)
        return θ_d * y + (1 - θ_d) * Bs
    end
end

function update(scheme::DBFGS{<:Inverse,<:Any}, H, s, y)
    ȳ = _damp_y(scheme, H, s, y)
    σ = dot(s, ȳ)
    ρ = inv(σ)
    C = (I - ρ * s * ȳ')
    H = C * H * C' + ρ * s * s'
    H
end
function update!(scheme::DBFGS{<:Inverse,<:Any}, H, s, y)
    ȳ = _damp_y(scheme, H, s, y)
    σ = dot(s, ȳ)
    ρ = inv(σ)

    if isfinite(ρ)
        Hȳ = H * ȳ
        H .= H .+ ((σ + ȳ' * Hȳ) .* ρ^2) * (s * s')
        Hȳs = Hȳ * s'
        Hȳs .= Hȳs .+ Hȳs'
        H .= H .- Hȳs .* ρ
    end
    H
end

function update!(scheme::DBFGS{<:Inverse,<:Any}, A::UniformScaling, s, y)
    update(scheme, A, s, y)
end
function update!(scheme::DBFGS{<:Direct,<:Any}, A::UniformScaling, s, y)
    update(scheme, A, s, y)
end
