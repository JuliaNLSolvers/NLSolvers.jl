# Missing!
# An, Mo, Liu https://www.sciencedirect.com/science/article/pii/S0377042705007958
# Shanno https://www.sciencedirect.com/science/article/abs/pii/0098135483800027

abstract type ForcingSequence end
struct FixedForceTerm{T} <: ForcingSequence
    η::T
end
η(fft::FixedForceTerm, info) = fft.η

# Truncated-Newton algorithms for large-scale unconstrained optimization
# https://link.springer.com/article/10.1007/BF02592055
struct DemboSteihaug <: ForcingSequence end
η(fft::DemboSteihaug, info) = min(1 / (info.k + 2), info.ρFz)

# See page 19 https://epubs.siam.org/doi/abs/10.1137/0917003
# Choice 1 (2.2) - it appears that the extra F'(x)s evaluation would
# not be worth it. It is at least as small as (2.1). If it's signifi-
# cantly smaller we risk oversolving.
# η₀ ∈ [0, 1)
struct EisenstatWalkerA{T} <: ForcingSequence
    s::T
    ηmax::T
end
EisenstatWalkerA() = EisenstatWalkerA(0.1, 0.9999)
function η(fft::EisenstatWalkerA, info)
    η = (info.ρFz - info.residual_old) / info.ρFx
    β = info.η_old^(1 + sqrt(5) / 2)
    if β ≥ fft.s
        η = max(η, β)
    end
    min(η, fft.ηmax)
end

# See page 20 https://epubs.siam.org/doi/abs/10.1137/0917003
# Choice 2 (2.6) seems to result in less oversolving.
# γ ∈ [0, 1] - but recommend γ ≥ 0.7
# α ∈ (1, 2]
# η₀ ∈ [0, 1)
# Default to EisenstatA-values. The paper [[[]]] suggests that γ ≥ 0.7 and
# ω ≥ (1+sqrt(5))/2
struct EisenstatWalkerB{T} <: ForcingSequence
    s::T
    α::T
    γ::T
    ηmax::T
end
EisenstatWalkerB() = EisenstatWalkerB(0.1, 1 + sqrt(5) / 2, 1.0, 100.0)

# https://epubs.siam.org/doi/10.1137/0911026
struct BrownSaad{T}
end

function η(fft::EisenstatWalkerB, info)
    T = typeof(info.ρFz)
    γ, α = fft.γ, fft.α

    ηx = info.η_old

    ρ = info.ρFz / info.ρFx
    η = γ * ρ^α
    β = γ * ηx^α
    if β ≥ fft.s
        η = max(η, β)
    end
    min(η, fft.ηmax)
end

# INB here:
# https://pdfs.semanticscholar.org/b321/3084f663260076dcb92f2fa6031b362dc5bc.pdf
# Originally Walker
# S. C. EiSENSTAT AND H. F. Walker, Globally convergent inexact Newton
# methods, SIAM J. Optim., 4 (1994), pp. 393-422.
"""
    InexactNewton(; force_seq, eta0, maxiter)

Constructs a method type for the Inexact Newton's method with Linesearch.

"""
struct InexactNewton{LinSolve,ForcingType<:ForcingSequence,Tη}
    linsolve::LinSolve
    force_seq::ForcingType
    η₀::Tη
    maxiter::Int
end
InexactNewton(; linsolve, force_seq = DemboSteihaug(), eta0 = 1e-4, maxiter = 300) =
    InexactNewton(linsolve, force_seq, eta0, maxiter)
# map from method to forcing sequence
η(fft::InexactNewton, info) = η(fft.force_seq, info)

init(::NEqProblem, ::InexactNewton; x, z = copy(x), xp = copy(x), d = copy(x), Fx = copy(x), Jx = x * x') =
    (; z, xp, d, Fx, Jx)
function solve(problem::NEqProblem, x, method::InexactNewton, options::NEqOptions, state = init(problem, method; x))
    if !(mstyle(problem) === InPlace())
        throw(ErrorException("solve() not defined for OutOfPlace() with InexactNewton"))
    end
    t0 = time()

    F = problem.R.F
    JvGen = problem.R.Jv

    Tx = eltype(x)
    z, xp, Fx = state.z, state.xp, state.Fx
    Fx = F(Fx, x) # FIXME: use the value interface
    JvOp = JvGen(x)

    ρ2F0 = norm(Fx, 2)
    ρFz = ρ2F0
    ρF0 = norm(Fx, Inf)
    ρs = norm(x, 2)
    stoptol = Tx(options.f_reltol) * ρFz + Tx(options.f_abstol)

    force_info = (k = 1, ρFz = ρFz, ρFx = nothing, η_old = nothing)

    iter = 0

    while iter < options.maxiter
        iter += 1
        x .= z
        # Refactor this
        if iter == 1 && !isa(method.force_seq, FixedForceTerm)
            ηₖ = method.η₀
        else
            ηₖ = η(method, force_info)
        end
        JvOp = JvGen(x)

        xp .= 0
        xp, res = method.linsolve(xp, JvOp, Fx, ηₖ)

        ρs = norm(xp)
        t = 1e-4
        # use line searc functions here
        btk_conv = false
        ρFx = force_info.ρFz

        # Backtracking
        it = 0
        while !btk_conv
            it += 1
            z = retract(problem, z, x, xp, -1)

            Fx = problem.R.F(Fx, z)
            btk_conv = norm(Fx, 2) ≤ (1 - t * (1 - ηₖ)) * ρFx || it > 20
        end
        if norm(Fx, 2) < stoptol
            break
        end
        η_old = ηₖ
        ρFz = norm(Fx, 2)
        force_info = (k = iter, ρFz = ρFz, ρFx = ρFx, η_old = η_old, residual_old = res)
    end
    return ConvergenceInfo(
        method,
        (
            solution = x,
            best_residual = Fx,
            ρF0 = ρF0,
            ρ2F0 = ρ2F0,
            ρs = ρs,
            iter = iter,
            time = time() - t0,
        ),
        options,
    )
end
