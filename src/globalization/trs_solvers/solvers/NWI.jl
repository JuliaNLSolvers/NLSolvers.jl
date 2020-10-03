#=================================================================================
  NWI (Nocedal Wright Iterative)) is a trust region sub-problem solver that sol-
  ves the quadratic programming problem subject to the solution being in a
  Euclidean Ball. It is appropriate for all types of Hessians. It solves the
  problem by solving the secular equ  ation by a safeguarded Newton's method.
  The secular equation is simply a the inverse of the difference between the
  step length and the trust region radius. It accepts either the Newton step
  if this is interior and the Hessian is positive definite, or steps to the
  boundary (with some slack) in a clever way.

  NTR accepts all Hessians and is therefore well-suited for Newton's method for
  any vexity. It is expensive for very large problems, and uses the direct
  eigensolution. This could potentially be useful for problems with simple 
  structure of the eigenproblem, but I suspect NWI is superior in most cases.
=================================================================================#

struct NWI <: NearlyExactTRSP
end
summary(::NWI) = "Trust Region (Newton, eigen)"

"""
    initial_safeguards(B, h, g, Δ)

Returns a tuple of initial safeguarding values for λ. Newton's method might not
work well without these safeguards when the Hessian is not positive definite.
"""
function initial_safeguards(B, h, g, Δ)
    # equations are on p. 560 of [MORESORENSEN]
    T = eltype(g)
    λS = maximum(-h)

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

"""
    is_maybe_hard_case(QΛQ, Qt∇f)

Returns a tuple of a Bool, `hardcase` and an integer, `λidx`. `hardcase` is true
if the sub-problem is the "hard case", that is the case where the smallest
eigenvalue is negative. `λidx` is the index of the first eigenvalue not equal
to the smallest eigenvalue. `QΛQ.values` holds the eigenvalues of H sorted low
to high, `Qt∇f` is a vector of the inner products between the eigenvectors and the
gradient.
"""
function is_maybe_hard_case(QΛQ, Qt∇f::AbstractVector{T}) where T
    # If the solution to the trust region sub-problem is on the boundary of the
    # trust region, {w : ||w|| ≤ Δk}, then the solution is usually found by
    # finding a λ ≥ 0 such that ||(B + λI)⁻¹g|| = Δ and x'(B + λI)x > 0, x≠0.
    # However, in the hard case our strategy for iteratively finding such an
    # λ breaks down, because ||p(λ)|| does not go to ∞ as λ→-λⱼ for j such that
    # qₜ'g ≠ 0 (see section 4.3 in [N&W] for more details). Remember, that B
    # does *not* have to be pos. def. in the trust region based methods!

    # Get the eigenvalues
    Λ = QΛQ.values

    # Get number of eigen values
    λnum = length(Λ)

    # The hard case requires a negativ smallest eigenvalue.
    λmin = first(Λ) # eigenvalues are sorted
    if λmin >= T(0)
        return false, 1
    end

    # Assume hard case and verify
    λidx = 1
    hard_case = true
    for (Qt∇f_j, λ_j) in zip(Qt∇f, Λ)
        if abs(λmin - λ_j) > sqrt(eps(T))
            hard_case = true
            break
        else
            if abs(Qt∇f_j) > sqrt(eps(T))
                hard_case = false
                break
            end
        end
        λidx += 1
    end

    hard_case, λidx
end

# Equation 4.38 in N&W (2006)
calc_p!(p, Qt∇f, QΛQ, λ) = calc_p!(p, Qt∇f, QΛQ, λ, 1)

# Equation 4.45 in N&W (2006) since we allow for first_j > 1
function calc_p!(p, Qt∇f, QΛQ, λ::T, first_j) where T
    # Reset search direction to 0
    fill!(p, T(0))

    # Unpack eigenvalues and eigenvectors
    Λ = QΛQ.values
    Q = QΛQ.vectors
    for j = first_j:length(Λ)
        κ = Qt∇f[j] / (Λ[j] + λ)
        @. p = p - κ*Q[:, j]
    end
    p
end

"""
    solve_tr_subproblem!(∇f, H, Δ, s; abstol, maxiter)
Args:
    ∇f: The gradient
    H:  The Hessian
    Δ:  The trust region size, ||s|| <= Δ
    s: Memory allocated for the step size, updated in place
    abstol: The convergence abstol for root finding
    maxiter: The maximum number of root finding iterations

Returns:
    m - The numeric value of the quadratic minimization.
    interior - A boolean indicating whether the solution was interior
    lambda - The chosen regularizing quantity
    hard_case - Whether or not it was a "hard case" as described by N&W (2006)
    solved - Whether or not a solution was reached (as opposed to
      terminating early due to maxiter)
"""
function (ms::NWI)(∇f, H, Δ, p, scheme; abstol=1e-10, maxiter=50)
    T = eltype(p)
    n = length(∇f)
    H = H isa UniformScaling ? Diagonal(copy(∇f).*0 .+ 1) : H
    h = diag(H)

    # Note that currently the eigenvalues are only sorted if H is perfectly
    # symmetric.  (Julia issue #17093)
    if H isa Diagonal
        QΛQ = eigen(H)
    else
        QΛQ = eigen(Symmetric(H))
    end
    Q, Λ = QΛQ.vectors, QΛQ.values
    λmin, λmax = Λ[1], Λ[n]
    # Cache the inner products between the eigenvectors and the gradient.
    Qt∇f = Q' * ∇f

    # These values describe the outcome of the subproblem.  They will be
    # set below and returned at the end.
    solved = true

    # Potentially an unconstrained/interior solution. The smallest eigenvalue is
    # positive, so the Newton step, pN, is fine unless norm(pN, 2) > Δ.
    if λmin >= sqrt(eps(T))
        λ = T(0) # no amount of I is added yet
        p = calc_p!(p, Qt∇f, QΛQ, λ) # calculate the Newton step
        if norm(p, 2) ≤ Δ
            # No shrinkage is necessary: -(H \ ∇f) is the minimizer
            interior = true
            solved = true
            hard_case = false

            m = dot(∇f, p) + dot(p, H * p)/2

            return (p=p, mz=m, interior=interior, λ=λ, hard_case=hard_case, solved=solved, Δ=Δ)
        end
    end

    # Set interior flag
    interior = false

    # The hard case is when the gradient is orthogonal to all
    # eigenvectors associated with the lowest eigenvalue.
    maybe_hard_case, first_j = is_maybe_hard_case(QΛQ, Qt∇f)
    # Solutions smaller than this lower bound on lambda are not allowed:
    # as the shifted Hessian will not be PSD.
    λ_lb = -λmin
    λ = λ_lb
    
    # Verify that it is actually the hard case situation by calculating the
    # step with λ = λmin (it's currently λ_lb, verify that that is correct).
    if maybe_hard_case
        # The "hard case".
        # λ is taken to be -λmin and we only need to find a multiple of an
        # orthogonal eigenvector that lands the iterate on the boundary.

        # The old p is discarded, and replaced with one that takes into account
        # the first j such that λj ≠ λmin. Formula 4.45 in N&W (2006)
        pλ = calc_p!(p, Qt∇f, QΛQ, λ, first_j)

        # Check if the choice of λ leads to a solution inside the trust region.
        # If it does, then we construct the "hard case solution".
        if norm(pλ, 2) ≤ Δ
            hard_case = true
            solved = true

            tau = sqrt(Δ^2 - norm(pλ, 2)^2)

            @. p = -pλ + tau * Q[:, 1]

            m = dot(∇f, p) + dot(p, H * p)/2

            return (p=p, mz=m, interior=interior, λ=λ, hard_case=hard_case, solved=solved, Δ=Δ)
        end
    end
    # If this is reached, we cannot be in the hard case after all, and we
    # can use Newton's method to find a p such that norm(p, 2) = Δ.
    hard_case = false

    # Algorithim 4.3 of N&W (2006), with s insted of p_l for consistency
    # with Optim.jl

    solved = false
    # We cannot be in the λ = -λ₁ case, as the root is in (-λ₁, ∞) interval.
    λ = λ + sqrt(eps(T))
    isg = initial_safeguards(H, h, ∇f, Δ)
    λ = safeguard_λ(λ, isg)
    for iter in 1:maxiter
        λ_previous = λ
        for i = 1:n
            @inbounds H[i, i] = h[i] + λ
        end

        F = H isa Diagonal ? cholesky(H; check=false) : cholesky(Hermitian(H); check=false)
        if !issuccess(F)
            # We should not be here, but the lower bound might fail for
            # numerical reasons.
            if λ < λ_lb
                λ = T(1)/2 * (λ_previous - λ_lb) + λ_lb
            end
            continue
        end
        R = F.U
        p .= R \ (R' \ -∇f)
        q_l = R' \ p

        p_norm = norm(p, 2)
        λ_update = p_norm^2 * (p_norm - Δ) / (Δ * dot(q_l, q_l))
        λ += λ_update

        # Check that λ is not less than λ_lb, and if so, go
        # half the way to λ_lb. (This should be geometric mean)
        if λ < λ_lb
            λ = T(1)/2 * (λ_previous - λ_lb) + λ_lb
        end
        if abs(λ - λ_previous) ≤ abstol
            solved = true
            break
        end
    end
    for i = 1:n
        @inbounds H[i, i] = h[i]
    end
    m = dot(∇f, p) + dot(p, H * p)/2
    return (p=p, mz=m, interior=interior, λ=λ, hard_case=hard_case, solved=solved, Δ=Δ)
end
