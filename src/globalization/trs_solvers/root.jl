# [SOREN] TRUST REGION MODIFICATION OF NEWTON'S METHOD
# [N&W] Numerical optimization
# [Yuan] A review of trust region algorithms for optimization
abstract type TRSPSolver end
abstract type NearlyExactTRSP <: TRSPSolver end

trs_supports_outofplace(trs) = false

function trs_outofplace_check(trs,prob)
    if !trs_supports_outofplace(trs)
        throw(
            ErrorException("solve() not defined for OutOfPlace() with $(typeof(trs).name.wrapper) for $(typeof(prob).name.wrapper)"),
        )
    end
end

include("solvers/NWI.jl")
include("solvers/Dogleg.jl")
include("solvers/NTR.jl")
include("solvers/TCG.jl")
#include("subproblemsolvers/TRS.jl") just make an example instead of relying onTRS.jl

function tr_return(; λ, ∇f, H, s, interior, solved, hard_case, Δ, m = nothing)
    m = m isa Nothing ? dot(∇f, s) + dot(s, H, s) / 2 : m
    (
        p = s,
        mz = m,
        interior = interior,
        λ = λ,
        hard_case = hard_case,
        solved = solved,
        Δ = Δ,
    )
end

update_H!(mstyle::OutOfPlace,H, h, λ) = _update_H(H, h, λ)
update_H!(mstyle::OutOfPlace,H, h) = _update_H(H, h, nothing)
update_H!(mstyle::InPlace,H, h, λ) = _update_H!(H, h, λ)
update_H!(mstyle::InPlace,H, h) = _update_H!(H, h, nothing)

function _update_H!(H, h, λ)
    T = eltype(h)
    n = length(h)
    if λ == nothing
        for i = 1:n
            @inbounds H[i, i] = h[i]
        end
    elseif !(λ == T(0))
        for i = 1:n
            @inbounds H[i, i] = h[i] + λ
        end
    end
    H
end

function _update_H(H, h, λ = nothing)
    T = eltype(h)
    if λ == nothing
        Hd = Diagonal(h)
        return H + Hd
    elseif !(λ == T(0))
        Hd = Diagonal(h)
        return H + Hd + λ*I
    else
        return H
    end
end