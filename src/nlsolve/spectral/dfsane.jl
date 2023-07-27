#===============================================================================
 Robust NonMonotone Line search
 From DF-Sane paper https://www.ams.org/journals/mcom/2006-75-255/S0025-5718-06-01840-0/S0025-5718-06-01840-0.pdf
 This line search doesn't fit the rest of the API so I need to come up with an
 interface for nonmonotone line searches (it has to accept an fbar (or Q in Hager
 Zhang's notion) and a forcing term)
===============================================================================#
struct RNMS{T}
    γ::T
    α0::T
end
RNMS(T = Float64; gamma = T(1) / 10000, alpha_0 = T(1)) = RNMS(gamma, alpha_0)

function find_steplength(rnms::RNMS, φ, φ0::T1, fbar, ηk::T2, τmin, τmax) where {T1,T2}
    α₊ = T1(rnms.α0)
    α₋ = α₊
    γ = T1(rnms.γ)
    for k = 1:100
        φα₊ = φ(α₊)
        if φα₊ ≤ fbar + ηk - γ * α₊^2 * φ0
            return α₊, φα₊
        end
        φα₋ = φ(-α₋)
        if φα₋ ≤ fbar + ηk - γ * α₋^2 * φ0
            return -α₋, φα₋
        end

        # update alpha+ and alpha-
        αt = α₊^2 * φ0 / (φα₊ + (2 * α₊ - 1) * φ0)
        α₊ = clamp(αt, τmin * α₊, τmax * α₊)

        αt = α₋^2 * φ0 / (φα₋ + (2 * α₋ - 1) * φ0)
        α₋ = clamp(αt, τmin * α₋, τmax * α₋)
    end
    return T1(NaN), T2(NaN)
end

function safeguard_σ(σ::T, σmin, σmax, F) where {T}
    if abs(σ) < σmin || abs(σ) > σmax
        normF = norm(F)
        if normF > T(1)
            σ = T(1)
        elseif T(1) / 10^5 ≤ normF ≤ T(1)
            σ = inv(normF)
        elseif normF < T(1) / 10^5
            σ = T(10)^5
        end
    end
    return σ
end
"""
    DFSANE(memory[ = 4], sigma_limits[ = (1e-5, 1e5)])

Construct a method instance of the DFSANE algorithm for solving systems of non-linear equations. The memory keyword is used to control how many residual norm values to store for the non-monotoneous line search (`M` in the paper). The `sigma_limit` keyword is used to provide a tuple of values used to bound the spectral coefficient (`σ_min` and `σ_max` in the paper).

# Notes
To re-implement the setup in the numerical section of [PAPER] use

DFSANE(memory=10, sigma_limit=(1e-10, 1e10), gamma=..., epsilon=..., c1=0.1, c2,0.5, alpha0=1))
"""
struct DFSANE{T,Σ}
    memory::T
    sigma_limits::Σ
end
DFSANE(; memory = 4, sigma_limits = (1e-5, 1e5)) = DFSANE(memory, sigma_limits)

init(::NEqProblem, ::DFSANE; x, Fx = copy(x), y = copy(x), d = copy(x), z = copy(x)) =
    (; x, Fx, y, d, z)
function solve(
    prob::NEqProblem,
    x0,
    method::DFSANE,
    options::NEqOptions,
    state = init(prob, method; x = copy(x0)),
)
    if !(mstyle(prob) === InPlace())
        throw(ErrorException("solve() not defined for OutOfPlace() with DFSANE"))
    end

    t0 = time()
    F = prob.R.F
    T = eltype(x0)

    σmin, σmax = method.sigma_limits
    τmin, τmax = T(1) / 10, T(1) / 2
    nexp = 2


    x, Fx, y, d, z = state.x, state.Fx, state.y, state.d, state.z
    Fx = F(Fx, x)

    x .= x0

    ρs = norm(x)
    ρ2F0 = norm(Fx, 2)
    ρF0 = norm(Fx, Inf)
    fx = ρ2F0^nexp

    fvals = [fx]

    abstol = 1e-5
    reltol = 1e-8

    σ₀ = T(1)
    σ = safeguard_σ(σ₀, σmin, σmax, Fx)
    iter = 0

    status = :iterating
    while iter < options.maxiter
        iter += 1
        push!(fvals, fx)
        if length(fvals) > method.memory
            popfirst!(fvals)
        end
        fbar = maximum(fvals)
        @. d = -σ * Fx
        ηk = ρ2F0 / (1 + iter)^2
        φ(α) = norm(F(Fx, (z .= x .+ α .* d)))^nexp
        φ0 = fx
        α, φα = find_steplength(RNMS(), φ, φ0, fbar, ηk, τmin, τmax)
        if isnan(α) || isnan(φα)
            status = :linesearch_failed
            break
        end
        s = α * d
        ρs = norm(s)
        x .+= s
        y .= -Fx
        Fx = F(Fx, x)
        y .+= Fx
        ρ2fx = norm(Fx, 2)
        ρfx = norm(Fx, Inf)
        fx = ρ2fx^nexp

        σ = norm(s)^2 / dot(s, y)
        σ = safeguard_σ(σ, σmin, σmax, Fx)

        # use sqrt(length(x))*abs or abstol?
        if ρfx < abstol + reltol * ρF0
            status = :converged
            break
        end
    end
    ConvergenceInfo(
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
