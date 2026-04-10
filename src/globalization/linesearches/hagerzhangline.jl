"""
    HZAW

An object that controls the Hager-Zhang approximate Wolfe line search
algorithm.[^HZ2005]

    HZAW(; kwargs...)

The `HZAW` constructor takes the following keyword arguments. Default values
correspond to those used in section 5 of [^HZ2005].
 - `decrease`: parameter between 0 and 1, less than or equal to `curvature`,
   specifying sufficient decrease in the objective per the Armijo rule.
   Defaults to 0.1.
 - `curvature`: parameter between 0 and 1, greater than or equal to `decrease`,
   specifying sufficient decrease in the gradient per the curvature condition.
   Defaults to 0.9.
 - `theta`: parameter between 0 and 1 that controls the bracketing interval
   update. Defaults to 1/2, which indicates bisection. (See step U3 of the
   interval update procedure in section 4 of [^HZ2005].)
 - `gamma`: factor by which the length of the bracketing interval should
   decrease at each iteration of the algorithm. Defaults to `2/3`. If such
   a decrease is not achieved, the interval is bisected instead of using the
   output of the secant^2 step.
 - `epsilon`: parameter that controls the approximate Wolfe conditions.
   Defaults to `1e-6`. See [p. 122, CG_DESCENT_851] for more details.
 - `maxiter`: maximum number of iterations in the main loop. Defaults to `50`.
 - `maxiter_U3`: maximum number of iterations in the U3 bisection step.
   Defaults to `50`.
 - `maxiter_finite_check`: maximum backtracking iterations to find a finite
   function value from a non-finite initial step. Defaults to `100`.
 - `rho`: expansion factor in the bracket procedure. Defaults to `5.0`.
 - `rho_finite_check`: contraction factor for backtracking from non-finite
   step. Defaults to `1/10`.

We tweak the original algorithm slightly, by backtracking into a feasible
region if the original step length results in function values that are not
finite. This allows us to set up an interval from this point that satisfies
the `bracket` procedure (bottom of [p. 123, CG_DESCENT_851]).

[^HZ2005]: Hager, W. W., & Zhang, H. (2005). A New Conjugate Gradient Method
           with Guaranteed Descent and an Efficient Line Search. SIAM Journal
           on Optimization, 16(1), 170-192. doi:10.1137/030601880
[^CG_DESCENT_851]: Hager, W. W., & Zhang, H. (2006). Algorithm 851: CG_DESCENT,
                   a Conjugate Gradient Method with Guaranteed Descent. ACM
                   Transactions on Mathematical Software, 32(1), 113-137.
                   doi:10.1145/1139480.1139484
"""
struct HZAW{T} <: LineSearcher
    decrease::T
    curvature::T
    θ::T
    γ::T
    ϵ::T
    maxiter::Int
    maxiter_U3::Int
    maxiter_finite_check::Int
    ρ::T
    ρ_finite_check::T
end

Base.summary(::HZAW) = "Approximate Wolfe Line Search (Hager & Zhang)"

HZAW{T}(h::HZAW) where {T} = HZAW(
    T(h.decrease), T(h.curvature), T(h.θ), T(h.γ), T(h.ϵ),
    h.maxiter, h.maxiter_U3, h.maxiter_finite_check,
    T(h.ρ), T(h.ρ_finite_check),
)

function HZAW(;
    decrease = 0.1,
    curvature = 0.9,
    theta = 0.5,
    gamma = 2 / 3,
    epsilon = 1e-6,
    maxiter = 50,
    maxiter_U3 = 50,
    maxiter_finite_check = 100,
    rho = 5.0,
    rho_finite_check = 1 / 10,
)
    if !(0 < decrease ≤ curvature)
        throw(ArgumentError(
            "Decrease constant must be positive and ≤ curvature. Got decrease=$decrease, curvature=$curvature.",
        ))
    end
    if decrease >= 1 / 2
        throw(ArgumentError(
            "Decrease constant must be < 1/2. Got decrease=$decrease.",
        ))
    end
    if curvature >= 1
        throw(ArgumentError(
            "Curvature constant must be < 1. Got curvature=$curvature.",
        ))
    end
    HZAW(
        decrease, curvature, theta, gamma, epsilon,
        maxiter, maxiter_U3, maxiter_finite_check,
        rho, rho_finite_check,
    )
end

struct TrialBundle{T}
    p::T
    φ::T
    dφ::T
end
Base.isfinite(tb::TrialBundle) = isfinite(tb.φ) && isfinite(tb.dφ)

function _evaltrial(φ, c)
    r = φ(c, true)
    TrialBundle(promote(c, r.ϕ, r.dϕ)...)
end

# At the core of this line search we have the (approximate) Wolfe conditions.
struct WolfeSetup{T}
    φ0::T
    dφ0::T
    δ::T
    σ::T
    ϵ::T
end
function WolfeSetup(Σ0::TrialBundle, δ, σ, ϵ)
    WolfeSetup(Σ0.φ, Σ0.dφ, δ, σ, ϵ)
end

# Wolfe conditions [eq (22), p.120, CG_DESCENT_851]
function _is_wolfe(wc::WolfeSetup, Σc::TrialBundle)
    (; φ0, dφ0, δ, σ) = wc
    φc, dφc, c = Σc.φ, Σc.dφ, Σc.p
    δ * dφ0 ≥ (φc - φ0) / c && dφc ≥ σ * dφ0
end

# Approximate Wolfe conditions [eq (23), p.120, CG_DESCENT_851]
function _is_approx_wolfe(wc::WolfeSetup, Σc::TrialBundle)
    (; φ0, dφ0, δ, σ, ϵ) = wc
    φc, dφc = Σc.φ, Σc.dφ
    # Satisfies T2 and eqn (27) [p. 122, CG_DESCENT_851]
    (2 * δ - 1) * dφ0 ≥ dφc ≥ σ * dφ0 && φc ≤ φ0 + ϵ * abs(φ0)
end

_is_converged(ws::WolfeSetup, Σ::TrialBundle) = _is_wolfe(ws, Σ) || _is_approx_wolfe(ws, Σ)

_in_bounds(c, Σa, Σb) = Σa.p <= c <= Σb.p

function find_steplength(mstyle, hzl::HZAW, φ, c::T) where {T}
    hzl = HZAW{T}(hzl)
    δ = hzl.decrease
    σ = hzl.curvature
    ρ = hzl.ρ
    ρ_finite_check = hzl.ρ_finite_check
    ϵ = hzl.ϵ
    φ0, dφ0 = T(φ.φ0), T(φ.dφ0)

    Σ0 = TrialBundle(T(0), φ0, dφ0)
    if !isfinite(Σ0)
        return T(NaN), T(NaN), false
    end

    Σc = _evaltrial(φ, c)

    # Backtrack into feasible region; not part of original algorithm
    iter = 0
    while !isfinite(Σc) && iter <= hzl.maxiter_finite_check
        iter += 1
        c = c * ρ_finite_check
        Σc = _evaltrial(φ, c)
    end
    if iter > hzl.maxiter_finite_check
        return T(NaN), T(NaN), false
    end

    wolfesetup = WolfeSetup(Σ0, δ, σ, ϵ)

    # Check initial convergence
    _is_converged(wolfesetup, Σc) && return Σc.p, Σc.φ, true

    # Set up bracket
    Σaj, Σbj, wolfe_in_bracket = _hz_bracket(hzl, Σ0, Σc, φ, ρ, wolfesetup)
    if wolfe_in_bracket
        return Σaj.p, Σaj.φ, true
    end

    # Main loop
    for j = 1:hzl.maxiter
        # === Step L1: Secant^2 update ===
        Σa, Σb, iswolfe = _hz_secant²(hzl, φ, φ0, Σaj, Σbj, ϵ, wolfesetup)
        if iswolfe
            return Σa.p, Σa.φ, true
        end

        # === Step L2: Bisection if insufficient decrease ===
        aj, bj = Σaj.p, Σbj.p
        a, b = Σa.p, Σb.p
        Σaj, Σbj = if b - a > hzl.γ * (bj - aj)
            c = (a + b) / 2
            _hz_update(hzl, Σa, Σb, c, φ, φ0, ϵ)
        else
            Σa, Σb
        end

        # Check bracket endpoints for convergence
        a_conv = _is_converged(wolfesetup, Σaj)
        b_conv = _is_converged(wolfesetup, Σbj)
        if a_conv && b_conv
            if Σaj.φ < Σbj.φ
                return Σaj.p, Σaj.φ, true
            else
                return Σbj.p, Σbj.φ, true
            end
        elseif a_conv
            return Σaj.p, Σaj.φ, true
        elseif b_conv
            return Σbj.p, Σbj.φ, true
        end
    end
    return T(NaN), T(NaN), false
end

"""
    _hz_update_U3

Step U3 of the updating procedure [p.123, CG_DESCENT_851]. The other steps
are in `_hz_update`, but this step is separated out to be able to use it in
step B2 of `_hz_bracket`. Initialization of a_bar and b_bar is done outside
this call.
"""
function _hz_update_U3(hzl::HZAW, φ, φ0, Σā::TrialBundle{T}, Σb̄::TrialBundle{T}, ϵ) where {T}
    # verified against paper description [p. 123, CG_DESCENT_851]
    θ = hzl.θ

    for j = 1:hzl.maxiter_U3
        # === Step U3.a === convex combination of a_bar and b_bar
        ā, b̄ = Σā.p, Σb̄.p
        d = (1 - θ) * ā + θ * b̄
        Σd = _evaltrial(φ, d)

        if Σd.dφ ≥ T(0)
            # found point of increasing objective; return with upper bound d
            return Σā, Σd
        else # now Σd.dφ < T(0)
            if Σd.φ ≤ φ0 + ϵ * abs(φ0)
                # === Step U3.b ===
                Σā = Σd
            else
                # === Step U3.c ===
                Σb̄ = Σd
            end
        end
    end
    return Σā, Σb̄
end

# Full update: bounds check + evaluate + U1-U3. Used by L2 bisection.
function _hz_update(hzl::HZAW, Σa, Σb, c, φ, φ0, ϵ)
    # === Step U0: Check c is interior to interval ===
    if !_in_bounds(c, Σa, Σb)
        return Σa, Σb
    end
    Σc = _evaltrial(φ, c)
    _hz_update_inner(hzl, Σa, Σb, Σc, φ, φ0, ϵ)
end

# Inner update with pre-evaluated Σc: U1-U3 only. Used by secant^2 after Wolfe check.
function _hz_update_inner(hzl::HZAW, Σa, Σb, Σc::TrialBundle{T}, φ, φ0, ϵ) where {T}
    # verified against paper description [p. 123, CG_DESCENT_851]
    # === Step U1: Positive derivative (update upper bound) ===
    if Σc.dφ ≥ T(0)
        return Σa, Σc
    else # Σc.dφ < T(0)
        # === Step U2: Negative derivative with sufficient decrease ===
        if Σc.φ ≤ φ0 + ϵ * abs(φ0)
            return Σc, Σb
        end
        # === Step U3: Negative derivative without sufficient decrease ===
        Σā, Σb̄ = Σa, Σc
        Σa, Σb = _hz_update_U3(hzl, φ, φ0, Σā, Σb̄, ϵ)
        return Σa, Σb
    end
end

"""
    _hz_bracket

Find an interval satisfying the opposite slope condition starting from
[0, c] [pp. 123-124, CG_DESCENT_851].
"""
function _hz_bracket(hzl::HZAW, Σ0::TrialBundle{T}, Σc::TrialBundle{T}, φ, ρ, wolfesetup) where {T}
    # verified against paper description [pp. 123-124, CG_DESCENT_851]
    φ0 = Σ0.φ
    ϵ = hzl.ϵ
    # === Step B0: Initialize bracket search ===
    Σcj = Σc

    # Note, we know that dφ(0) < 0 since we accepted that the current step is in a
    # direction of descent.
    Σci = Σ0

    maxj = 100
    j = 0
    while j < maxj && Σcj.dφ < T(0)
        j += 1
        if Σcj.φ > φ0 + ϵ * abs(φ0)
            # === Step B2: Decreasing derivative without sufficient decrease ===
            # φ is decreasing at cj but function value is sufficiently larger than
            # φ0 so we must have passed a place with increasing φ, use U3 to update.
            Σa, Σb = _hz_update_U3(hzl, φ, φ0, Σ0, Σcj, ϵ)
            return Σa, Σb, false
        end

        # === Step B3: Decreasing derivative with sufficient decrease ===
        # Move lower bound up to cj, expand by factor ρ > 1
        Σci = Σcj

        cj = ρ * Σcj.p
        Σcj = _evaltrial(φ, cj)
        # Check if the new point satisfies Wolfe before continuing expansion
        if _is_converged(wolfesetup, Σcj)
            return Σcj, Σcj, true
        end
    end
    if j == maxj
        @warn "Failed to find a bracket satisfying the opposite slope condition after $maxj iterations."
    end

    # Implicitly Σcj.dφ ≥ T(0) since we exited the loop =>
    # === Step B1: Positive derivative found (opposite slope condition) ===
    return Σci, Σcj, false
end

function _hz_secant(Σa::TrialBundle{T}, Σb::TrialBundle{T}) where {T}
    # verified against paper description [p. 123, CG_DESCENT_851]
    # (a*dφb - b*dφa)/(dφb - dφa)
    # It has been observed that dφa can be very close to dφb,
    # so we avoid taking the difference
    a, dφa, b, dφb = Σa.p, Σa.dφ, Σb.p, Σb.dφ
    sec = a / (1 - dφa / dφb) + b / (1 - dφb / dφa)
    if isnan(sec)
        return (a * dφb - b * dφa) / (dφb - dφa)
    end
    return sec
end

function _hz_secant²(hzl::HZAW, φ, φ0, Σa::TrialBundle{T}, Σb::TrialBundle{T}, ϵ, wolfesetup) where {T}
    # verified against paper description [p. 123, CG_DESCENT_851]
    # === Step S1: First secant step ===
    c = _hz_secant(Σa, Σb)
    if !_in_bounds(c, Σa, Σb)
        return Σa, Σb, false
    end
    Σc = _evaltrial(φ, c)
    if _is_converged(wolfesetup, Σc)
        return Σc, Σc, true
    end
    # First update (U1-U3 with pre-evaluated Σc)
    ΣA, ΣB = _hz_update_inner(hzl, Σa, Σb, Σc, φ, φ0, ϵ)
    updated = false
    c̄ = c
    if c == ΣB.p # B == c
        # === Step S2: Second secant with new upper bound ===
        c̄ = _hz_secant(Σb, ΣB)
        updated = true
    elseif c == ΣA.p # A == c
        # === Step S3: Second secant with new lower bound ===
        c̄ = _hz_secant(Σa, ΣA)
        updated = true
    end

    # === Step S4 ===
    if !updated
        # === Step S4 (variant 2): Return without second secant ===
        return ΣA, ΣB, false
    end
    if !_in_bounds(c̄, ΣA, ΣB)
        return ΣA, ΣB, false
    end
    Σc̄ = _evaltrial(φ, c̄)
    if _is_converged(wolfesetup, Σc̄)
        return Σc̄, Σc̄, true
    end
    # === Step S4 (variant 1): Update with second secant point ===
    Σā, Σb̄ = _hz_update_inner(hzl, ΣA, ΣB, Σc̄, φ, φ0, ϵ)
    return Σā, Σb̄, false
end
