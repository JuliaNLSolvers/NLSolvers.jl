#===============================================================================
  TDTR (Two-Dimensional Trust Region) solves the trust region sub-problem

    min_p m(p) = ∇f'p + p'Hp/2   s.t. ||p|| ≤ Δ

  exactly, for two-dimensional problems only. The 2x2 spectral decomposition
  has a closed form, so the secular equation

    ||p(σ)||² = g̃₁²/(λ₁+σ)² + g̃₂²/(λ₂+σ)² = Δ²

  (g̃ = Q'∇f in the eigenbasis) reduces to a quartic in the multiplier σ that
  can either be solved in closed form (boundary = :quartic, Ferrari's method
  with a trigonometric resolvent branch, followed by one safeguarded Newton
  polish step) or by a safeguarded Newton iteration on the reciprocal secular
  equation 1/||p(σ)|| = 1/Δ (boundary = :newton), which is nearly linear in σ
  [MS]. The hard case is handled exactly: in two dimensions the boundary
  solution along the extra eigenvector direction is available in closed form,
  so no inverse iteration or LINPACK-style eigenvector estimate is needed.

  Unlike NWI and NTR, no factorizations or general eigensolves are performed,
  H is never mutated, and no memory is allocated, so the solver works for any
  Real element type (including BigFloat) and with immutable (static) arrays.

  [MS] Moré & Sorensen (1983), "Computing a trust region step",
       SIAM J. Sci. Stat. Comput.
===============================================================================#
"""
    TDTR(; boundary = :newton, abstol = 1e-10, maxiter = 50)

An exact trust region sub-problem solver for two-dimensional problems. The
2x2 model Hessian is diagonalized in closed form and the boundary multiplier
is found from the secular quartic, by a safeguarded Newton iteration on the
reciprocal secular equation (`boundary = :newton`, the default and the faster
of the two in benchmarks) or in closed form (`boundary = :quartic`, Ferrari's
method polished by one safeguarded Newton step, falling back to the iteration
when the closed form degenerates). The hard case is solved exactly.
Indefinite model Hessians are supported, and the `Direct`/`Inverse` form of
quasi-Newton schemes is respected. Throws an `ArgumentError` for problems
that are not two-dimensional.
"""
struct TDTR{Ta} <: NearlyExactTRSP
    boundary::Symbol
    abstol::Ta
    maxiter::Int
end
function TDTR(; boundary = :newton, abstol = 1e-10, maxiter = 50)
    if !(boundary === :quartic || boundary === :newton)
        throw(ArgumentError("boundary must be :quartic or :newton"))
    end
    TDTR(boundary, float(abstol), maxiter)
end
summary(::TDTR) = "Trust Region (2x2, closed form)"

_tdtr_entries(H::UniformScaling) = float(H.λ), float(zero(H.λ)), float(H.λ)
_tdtr_entries(H::Diagonal) = float(H.diag[1]), float(zero(H.diag[1])), float(H.diag[2])
_tdtr_entries(H::AbstractMatrix) =
    float(H[1, 1]), float((H[1, 2] + H[2, 1]) / 2), float(H[2, 2])

function _tdtr_setp(::InPlace, p, p1, p2)
    p[1] = p1
    p[2] = p2
    p
end
# false is a strong zero, so stale buffer contents cannot leak into the result;
# broadcasting against p keeps the container type (an SVector stays an SVector)
_tdtr_setp(::OutOfPlace, p, p1, p2) = false .* p .+ (p1, p2)

function (ms::TDTR)(∇f, H, Δ, p, scheme, mstyle; abstol = ms.abstol, maxiter = ms.maxiter)
    if length(∇f) != 2 || !(H isa UniformScaling || size(H) == (2, 2))
        throw(ArgumentError("TDTR is specialized to two-dimensional problems"))
    end
    h11, h12, h22 = _tdtr_entries(H)
    if !isa(scheme.approx, Direct)
        # H holds the inverse model matrix, invert it in closed form
        dt = h11 * h22 - h12 * h12
        h11, h12, h22 = h22 / dt, -h12 / dt, h11 / dt
    end
    h11, h12, h22, g1, g2, ΔT = promote(h11, h12, h22, float(∇f[1]), float(∇f[2]), float(Δ))

    λ1, λ2, c2, s2 = _tdtr_eigen(h11, h12, h22)
    # gradient in the eigenbasis, q₁ = (-s2, c2), q₂ = (c2, s2)
    gt1 = c2 * g2 - s2 * g1
    gt2 = c2 * g1 + s2 * g2

    pt1, pt2, σ, interior, hard_case, solved =
        _tdtr_solve(λ1, λ2, gt1, gt2, ΔT, ms.boundary, abstol, maxiter)

    p1 = c2 * pt2 - s2 * pt1
    p2 = c2 * pt1 + s2 * pt2
    p = _tdtr_setp(mstyle, p, p1, p2)

    m = gt1 * pt1 + gt2 * pt2 + (λ1 * pt1^2 + λ2 * pt2^2) / 2
    return (
        p = p,
        mz = m,
        interior = interior,
        λ = σ,
        hard_case = hard_case,
        solved = solved,
        Δ = Δ,
    )
end

# Closed-form spectral decomposition of a symmetric 2x2 matrix. Returns
# λ1 ≤ λ2 and (c2, s2) with eigenvectors q₂ = (c2, s2) and q₁ = (-s2, c2);
# the branch on the sign of the half-difference avoids cancellation.
function _tdtr_eigen(h11, h12, h22)
    mv = (h11 + h22) / 2
    dv = (h11 - h22) / 2
    r = hypot(dv, h12)
    λ1 = mv - r
    λ2 = mv + r
    if iszero(r)
        c2, s2 = one(r), zero(r)
    elseif dv >= 0
        nrm = sqrt(2 * r * (r + dv))
        c2, s2 = (r + dv) / nrm, h12 / nrm
    else
        nrm = sqrt(2 * r * (r - dv))
        c2, s2 = h12 / nrm, (r - dv) / nrm
    end
    return λ1, λ2, c2, s2
end

# Solve the diagonalized sub-problem: minimize g̃'p̃ + (λ₁p̃₁² + λ₂p̃₂²)/2 over
# ||p̃|| ≤ Δ. Returns p̃₁, p̃₂, σ, interior, hard_case, solved.
function _tdtr_solve(λ1, λ2, gt1, gt2, Δ, boundary, abstol, maxiter)
    T = typeof(λ1)
    z = zero(T)

    if iszero(gt1) && iszero(gt2)
        if λ1 >= z
            return z, z, z, true, false, true
        else
            # any boundary point along q₁ is a solution
            return Δ, z, -λ1, false, true, true
        end
    end

    if λ1 > z
        pN1 = -gt1 / λ1
        pN2 = -gt2 / λ2
        if pN1^2 + pN2^2 <= Δ^2
            return pN1, pN2, z, true, false, true
        end
    end

    # boundary solution: (λᵢ + σ)p̃ᵢ = -g̃ᵢ with ||p̃(σ)|| = Δ, σ ≥ max(0, -λ₁)
    if λ1 == λ2
        gn = hypot(gt1, gt2)
        σ = gn / Δ - λ1
        scale = Δ / gn
        return -gt1 * scale, -gt2 * scale, σ, false, false, true
    end

    if iszero(gt1)
        w = -gt2 / (λ2 - λ1)
        if λ1 <= z && abs(w) <= Δ
            # the hard case; the q₁ component is free to reach the boundary
            τ = sqrt(max(Δ^2 - w^2, z))
            return τ, w, -λ1, false, true, true
        end
        return z, -copysign(Δ, gt2), abs(gt2) / Δ - λ2, false, false, true
    end

    if iszero(gt2)
        return -copysign(Δ, gt1), z, abs(gt1) / Δ - λ1, false, false, true
    end

    # generic case: the unique root σ* of ||p̃(σ)|| = Δ in (max(0, -λ₁), ∞),
    # bracketed by the component-wise and full-gradient bounds
    σlo = max(z, -λ1 + abs(gt1) / Δ, abs(gt2) / Δ - λ2)
    σhi = max(σlo, hypot(gt1, gt2) / Δ - λ1)

    # second-order estimate of the root measured from -λ₁, accurate when the
    # root is close to -λ₁ (the near-hard regime, where the reciprocal secular
    # equation is flat on one side and Newton would fall back to bisection)
    σhard = T(NaN)
    if λ1 < z
        w0 = gt2 / (λ2 + σlo)
        τ20 = Δ^2 - w0^2
        if τ20 > z
            σhard = -λ1 + abs(gt1) / sqrt(τ20)
        end
    end

    solved = false
    σ = σlo
    if !isnan(σhard) && σhard + λ1 <= cbrt(eps(T)) * max(-λ1, σhard)
        # the root is unresolvably close to -λ₁: the boundary completion below
        # rebuilds p̃₁ from the boundary equation, so iterating on σ cannot
        # improve on the estimate
        σ = clamp(σhard, σlo, σhi)
        solved = true
    else
        if boundary === :quartic
            σq = _tdtr_secular_quartic(λ1, λ2, gt1, gt2, Δ, σlo, σhi)
            σ, solved = _tdtr_secular_newton(λ1, λ2, gt1, gt2, Δ, σlo, σhi, σq, abstol, 3)
        end
        if !solved
            σ0 = isnan(σhard) ? σlo + (σhi - σlo) / 2 : clamp(σhard, σlo, σhi)
            σ, solved =
                _tdtr_secular_newton(λ1, λ2, gt1, gt2, Δ, σlo, σhi, σ0, abstol, maxiter)
        end
    end
    a1 = λ1 + σ
    pt1 = iszero(a1) ? z : -gt1 / a1
    pt2 = -gt2 / (λ2 + σ)
    hard_case = false
    if λ1 < z && a1 <= cbrt(eps(T)) * max(-λ1, σ)
        # numerically hard case: the boundary root is so close to -λ₁ that the
        # rounding of σ leaves λ₁ + σ with too few digits to place p̃₁ on the
        # boundary by the division above. Set the q₁ component from the
        # boundary equation instead, as in the exact hard case; this changes
        # the stationarity residual by at most (λ₁+σ)Δ, at rounding level here
        pt1 = copysign(sqrt(max(Δ^2 - pt2^2, z)), pt1)
        hard_case = true
        solved = true
    end
    return pt1, pt2, σ, false, hard_case, solved
end

# Safeguarded Newton iteration on the reciprocal secular equation
# φ(σ) = 1/||p̃(σ)|| - 1/Δ, which is monotone and nearly linear on the
# bracket [σlo, σhi] [MS]. Falls back to bisection when a Newton step
# leaves the bracket.
function _tdtr_secular_newton(λ1, λ2, gt1, gt2, Δ, σlo, σhi, σstart, abstol, maxiter)
    T = typeof(σlo)
    σ = clamp(σstart, σlo, σhi)
    tol = max(T(abstol), 2 * eps(T)) * Δ
    solved = false
    for _ = 1:maxiter
        a1 = λ1 + σ
        a2 = λ2 + σ
        w1 = gt1 / a1
        w2 = gt2 / a2
        n2 = w1^2 + w2^2
        n = sqrt(n2)
        if abs(n - Δ) <= tol
            solved = true
            break
        end
        if n > Δ
            σlo = σ
        else
            σhi = σ
        end
        dn = -(w1^2 / a1 + w2^2 / a2) / n
        σnew = σ + (inv(n) - inv(Δ)) * n2 / dn
        if !(σlo < σnew < σhi)
            σnew = σlo + (σhi - σlo) / 2
        end
        if σnew == σ || σhi - σlo <= eps(T) * max(one(T), abs(σhi))
            # the bracket is resolved to machine precision
            solved = true
            break
        end
        σ = σnew
    end
    return σ, solved
end

# Closed-form solution of the secular quartic
#   (λ₁+σ)²(λ₂+σ)²Δ² - g̃₁²(λ₂+σ)² - g̃₂²(λ₁+σ)² = 0
# via Ferrari's method; the sought multiplier is the largest real root. The
# problem is normalized by S so all coefficients are O(1). Returns a point in
# [σlo, σhi] to be polished; degenerate arithmetic falls back to the midpoint.
function _tdtr_secular_quartic(λ1, λ2, gt1, gt2, Δ, σlo, σhi)
    T = typeof(σlo)
    S = max(σhi, abs(λ1), abs(λ2))
    b = λ1 / S
    d = λ2 / S
    a = (gt1 / (S * Δ))^2
    c = (gt2 / (S * Δ))^2

    B3 = 2 * (b + d)
    B2 = b^2 + 4 * b * d + d^2 - (a + c)
    B1 = 2 * b * d * (b + d) - 2 * (a * d + c * b)
    B0 = (b * d)^2 - (a * d^2 + c * b^2)

    # depressed quartic y⁴ + pd y² + qd y + rd with σ̂ = y - B3/4
    pd = B2 - 3 * B3^2 / 8
    qd = B1 - B3 * B2 / 2 + B3^3 / 8
    rd = B0 - B3 * B1 / 4 + B3^2 * B2 / 16 - 3 * B3^4 / 256

    ymax = T(-Inf)
    if iszero(qd)
        # biquadratic: y² solves a quadratic directly
        disc = pd^2 - 4 * rd
        if disc >= 0
            sd = sqrt(disc)
            for y2 in ((-pd + sd) / 2, (-pd - sd) / 2)
                if y2 >= 0
                    ymax = max(ymax, sqrt(y2))
                end
            end
        end
    else
        # Ferrari: y⁴ + pd y² + qd y + rd =
        #   (y² - √(2m) y + pd/2 + m + qd/(2√(2m))) ⋅
        #   (y² + √(2m) y + pd/2 + m - qd/(2√(2m)))
        # for any root m of the resolvent cubic; qd ≠ 0 guarantees one with m > 0
        mstar = _tdtr_max_real_root_cubic(pd, pd^2 / 4 - rd, -qd^2 / 8)
        mstar = max(mstar, zero(T))
        s2m = sqrt(2 * mstar)
        if s2m > 0
            t = qd / (2 * s2m)
            for (bq, cq) in ((-s2m, pd / 2 + mstar + t), (s2m, pd / 2 + mstar - t))
                nroots, r1, r2 = _tdtr_quad_roots(bq, cq)
                if nroots == 2
                    ymax = max(ymax, r1, r2)
                end
            end
        end
    end
    if !isfinite(ymax)
        return σlo + (σhi - σlo) / 2
    end
    return clamp(S * (ymax - B3 / 4), σlo, σhi)
end

# Largest real root of m³ + a2 m² + a1 m + a0. Three-real-root cases use the
# trigonometric form (the Cardano form would need complex intermediates);
# single-root cases use Cardano with the larger-magnitude cube root first to
# avoid cancellation.
function _tdtr_max_real_root_cubic(a2, a1, a0)
    T = typeof(a2)
    P = a1 - a2^2 / 3
    Q = a0 - a2 * a1 / 3 + 2 * a2^3 / 27
    off = a2 / 3
    disc = -(4 * P^3 + 27 * Q^2)
    if disc >= 0 && P < 0
        sp = sqrt(-P / 3)
        arg = clamp(3 * Q / (2 * P * sp), -one(T), one(T))
        t = 2 * sp * cos(acos(arg) / 3)
        return t - off
    else
        R = sqrt(max(Q^2 / 4 + P^3 / 27, zero(T)))
        u = cbrt(-Q / 2 - copysign(R, Q))
        t = iszero(u) ? u : u - P / (3 * u)
        return t - off
    end
end

# Real roots of y² + b y + c with the cancellation-free formula;
# returns (count, root, root)
function _tdtr_quad_roots(b, c)
    disc = b^2 - 4 * c
    if disc < 0
        return 0, zero(b), zero(b)
    end
    q = -(b + copysign(sqrt(disc), b)) / 2
    return 2, q, iszero(q) ? q : c / q
end
