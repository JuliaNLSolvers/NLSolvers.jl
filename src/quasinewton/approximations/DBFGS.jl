struct DBFGS{T1,T2,T3} <: QuasiNewton{T1}
    approx::T1
    theta::T2
    P::T3
end
DBFGS(approx) = DBFGS(approx, 0.2, nothing)
DBFGS(; inverse = true, theta = 0.2) = DBFGS(inverse ? Inverse() : Direct(), theta, nothing)
hasprecon(::DBFGS{<:Any,<:Any,<:Nothing}) = NoPrecon()
hasprecon(::DBFGS{<:Any,<:Any,<:Any}) = HasPrecon()
summary(dbfgs::DBFGS{Inverse}) = "Inverse Damped BFGS"
summary(dbfgs::DBFGS{Direct}) = "Direct Damped BFGS"

function update!(scheme::DBFGS{<:Direct,<:Any,<:Any}, B, s, y)
    # Procedure 18.2 (Damped BFGS Updating) Nocedal Wright 2nd edition
    σ = dot(s, y)
    ρ = inv(σ) # scalar
    # Then calculate the vector b
    b = B * s # vector temporary
    sb = dot(s, b)
    if σ ≥ scheme.theta * sb
        θ = 1
        r = y
        σ̂ = σ
    else
        θ = 0.8 * sb / (sb - σ)
        r = y * θ + (1 - θ) * b
        σ̂ = dot(s, r)
    end
    B .= B .+ r*r'/σ̂ - b*b'/sb
end
function update(scheme::DBFGS{<:Direct,<:Any}, B, s, y)
    # As above, but out of place
    σ = dot(s, y)
    b = B * s
    sb = dot(s, b)
    if σ ≥ scheme.theta * sb
        θ = 1
        r = y
        σ̂ = σ
    else
        θ = 0.8 * sb / (sb - σ)
        r = y * θ + (1 - θ) * b
        σ̂ = dot(s, r)
    end
    B = B + r*r'/σ̂ - b*b'/sb
end
function update(scheme::DBFGS{<:Inverse,<:Any}, H, s, y)
   
end
function update!(scheme::DBFGS{<:Inverse,<:Any}, H, s, y)
 
end

function update!(scheme::DBFGS{<:Inverse,<:Any}, A::UniformScaling, s, y)
    update(scheme, A, s, y)
end
function update!(scheme::DBFGS{<:Direct,<:Any}, A::UniformScaling, s, y)
    update(scheme, A, s, y)
end
