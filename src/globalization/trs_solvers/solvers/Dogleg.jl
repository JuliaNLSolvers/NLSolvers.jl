# Compute B*v where B is the actual Hessian. For Direct form H=B so B*v = H*v.
# For Inverse form H=B⁻¹ so B*v = H\v. Falls back to v (B=I) if H is singular.
_hessian_product(scheme, H, v) = isa(scheme.approx, Direct) ? H * v : _safe_solve(H, v)
function _safe_solve(H, v)
    F = cholesky(Symmetric(H); check = false)
    issuccess(F) ? F \ v : v
end

# TODO add double dog leg and subspace dogleg
#===============================================================================
  Dogleg is a trust region sub-problem solver used to generate a cheap and crude
  approximation to the solution. If the Cauchy step is outside of the trust re-
  gion it will scale it down to have the length of the radius. If Cauchy step
  is in the interior, it will find the intersection between the Newton step and
  the Cauchy step. Since the trust region is a Euclidean Ball, it is simple to
  find this point on the trust region boundary.

  The Dogleg solver is only appropriate for positive definite Hessians.
===============================================================================#
"""
    Dogleg()

A trust region sub-problem solver that assumes positive definite hessians (exact
or quasi-Newton approximations such as BFGS or variants).
"""
struct Dogleg{T} <: TRSPSolver
    γ::T # unused, for double-dogleg
end
Dogleg() = Dogleg(nothing)

function (dogleg::Dogleg)(∇f, H, Δ, p, scheme, mstyle; abstol = 1e-10, maxiter = 50)
    T = eltype(p)
    n = length(∇f)

    # find the Cauchy point; assumes ∇f is not ≈ 0
    # For Direct form H is B (Hessian), for Inverse form H is B⁻¹,
    # so we need H\∇f to recover B*∇f.
    B∇f = _hessian_product(scheme, H, ∇f)
    d_cauchy = -∇f * norm(∇f)^2 / (∇f' * B∇f)

    # If it lies outside of the trust region, accept the Cauchy point and
    # move on
    norm_d_cauchy = norm(d_cauchy)
    if norm_d_cauchy ≥ Δ
        shrink = Δ / norm_d_cauchy # inv(Δ/norm_d_cauchy) puts it on the border

        p = _scale(mstyle, p, d_cauchy, shrink)
        interior = false
    else
        # Else, calculate (Quasi-)Newton step. If this is interior, then take the
        # step. Otherwise find where the dog-leg path crosses the trust region

        # find the (quasi-)Newton step
        p = find_direction!(p, H, nothing, ∇f, scheme)
        norm_p = norm(p)
        if norm_p ≤ Δ # fixme really need to add the 20% slack here (see TR book and NTR)
            if norm_p < Δ
                interior = true
            else
                interior = false
            end
        else
            # we now solve a quadratic to find the step size t from d_cauchy
            # towards p. We use a numerically stable way of doing this (see
            # any numerical analysis text book) See [NW, p. 75] for the expression
            # giving rise to the quadratic equation
            dot_cachy_p = dot(d_cauchy, p)

            # a is ||d_c - d_n||^2 expanded into scalar operations
            a = norm_d_cauchy^2 + norm_p^2 - 2 * dot_cachy_p
            b = -dot_cachy_p * norm_d_cauchy^2
            c = norm_d_cauchy^2 - Δ^2 # move the rhs over
            q = -(b + sign(b) * √(b^2 - 4 * a * c)) / 2

            # since we know that c is necessarily negative (since d_cauchy was
            # not at the border) the discriminant is positive, and there are two
            # roots - pick the positive one. There has to be one positive and one
            # negative.
            if b > 0
                # if b is positive, q is negative. Then if c is negative, we must
                # have that c / q is the positive root.
                t = c / q
            elseif b < 0
                # else, the other root must be positive
                t = q / a
            else
                t = T(0)
            end

            p, _ = move(mstyle, p, d_cauchy, p, p, t)
            interior = false
        end
    end
    # Model value m(p) = ∇f'p + p'Bp/2. For Inverse form, B = H⁻¹.
    Bp = _hessian_product(scheme, H, p)
    m = dot(∇f, p) + dot(p, Bp) / 2

    return (
        p = p,
        mz = m,
        interior = interior,
        λ = nothing,
        hard_case = nothing,
        solved = true,
        Δ = Δ,
    )
end
