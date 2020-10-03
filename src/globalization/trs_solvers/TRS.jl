struct TRSolver{T} <: NearlyExactTRSP
    abstol::T
    maxiter::Integer
end
function (ms::TRSolver)(∇f, H, Δ, p)
    T = eltype(p)
    x, info = trs(H, ∇f, Δ)
    p .= x[:,1]

    m = dot(∇f, p) + dot(p, H * p)/2
    interior = norm(p, 2) ≤ Δ
    return (p=p, mz=m, interior=interior, λ=info.λ, hard_case=info.hard_case, solved=true)
end
"""
    initial_safeguards(B, g, Δ)

Returns a tuple of initial safeguarding values for λ. Newton's method might not
work well without these safeguards when the Hessian is not positive definite.
"""
function initial_safeguards(B, g, Δ)
    # equations are on p. 560 of [MORESORENSEN]
    T = eltype(g)
    λS = maximum(-diag(B))

    # they state on the first page that ||⋅|| is the Euclidean norm
    gnorm = norm(g)
    Bnorm = opnorm(B, 1)
    λL = max(T(0), λS, gnorm/Δ - Bnorm)
    λU = gnorm/Δ + Bnorm
    (L=λL, U=λU, S=λS)
end
function safeguard_λ(λ::T, λsg) where T
    # p. 558
    λ = min(max(λ, λsg.L), λsg.U)
    if λ ≤ λsg.S
        λ = max(T(1)/1000*λsg.U, sqrt(λsg.L*λsg.U))
    end
    λ
end
