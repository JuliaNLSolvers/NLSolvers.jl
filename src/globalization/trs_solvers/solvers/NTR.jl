#=================================================================================
  NTR is a trust region sub-problem solver that solves the quadratic programming
  problem subject to the solution being in a Euclidean Ball. It is appropriate
  for all types of Hessians. It solves the problem by solving the secular equ-
  ation by a safeguarded Newton's method. The secular equation is simply the
  inverse of the difference between the step length and the trust region radius.
  It accepts either the Newton step if this is interior and the Hessian is po-
  sitive definite, or steps to the boundary (with some slack) in a clever way.

  NTR accepts all Hessians and is therefore well-suited for Newton's method for
  any vexity. It is expensive for very large problems, but uses Cholesky fac-
  torizations in clever ways, so it should be less expensive than the version
  that calculates the full eigensolution.
=================================================================================#

# TODO allow passing in a lambda and previous cholesky if the
# solution was not accepted
# As described in Algorith, 7.3.4 in [CGTBOOK]
struct NTR <: TRSPSolver end
summary(::NTR) = "Trust Region (Newton, cholesky)"

function initial_λs(∇f, H, Δ)
    n = length(∇f)
    T = eltype(∇f)
    Hfrob = norm(H, 2)
    Hinf = norm(H, Inf)
    max_h = maximum(diag(H))

    norm_to_Δ = norm(∇f) / Δ

    sumabs_j = sum(abs, view(H, 2:n, 1))
    max_i_p = sumabs_j + H[1, 1]
    max_i_m = sumabs_j - H[1, 1]
    for j = 2:n
        sumabs_j = T(0)
        for i = 1:n-1
            i == j && continue
            sumabs_j += abs(H[i, j])
        end
        max_i_p = max(max_i_p, sumabs_j + H[j, j])
        max_i_m = max(max_i_m, sumabs_j - H[j, j])
    end
    λL = max(T(0), norm_to_Δ - min(max_i_p, Hfrob, Hinf), -max_h)
    λU = max(T(0), norm_to_Δ + min(max_i_m, Hfrob, Hinf))
    return λL, λU
end
# FIXME why is this not used?
#λ⁺_newton(λ, w, Δ) = λ + (s₂^2 / dot(w, w)) * (s₂ - Δ) / Δ
function (ms::NTR)(
    ∇f,
    H,
    Δ::T,
    s,
    scheme,
    mstyle,
    λ0 = 0;
    abstol = 1e-10,
    maxiter = 50,
    κeasy = T(1) / 10,
    κhard = T(2) / 10,
) where {T}
    # λ0 might not be 0 if we come from a failed TRS solve
    λ = T(λ0)
    θ = T(1) / 2
    n = length(∇f)
    h = H isa UniformScaling ? copy(∇f) .* 0 .+ 1 : diag(H)
    H = H isa UniformScaling ? Diagonal(copy(∇f) .* 0 .+ 1) : H

    # Check for interior convergence
    if λ == T(0)
        F = cholesky(Symmetric(H); check = false)
        s .= -∇f
        s .= F \ s
        s₂ = norm(s, 2)

        if issuccess(F) && s₂ < Δ
            H = update_H!(H, h)
            return tr_return(;
                λ = λ,
                ∇f = ∇f,
                H = H,
                s = s,
                interior = true,
                solved = true,
                hard_case = false,
                Δ = Δ,
            )
        end
    end

    # If the solution was not internal, start Newton's method on the
    # secular equation.
    isg = initial_safeguards(H, h, ∇f, Δ)
    λ = safeguard_λ(λ, isg)
    λL, λU = isg.L, isg.U

    for iter = 1:maxiter
        H = update_H!(H, h, λ)
        F = cholesky(Symmetric(H); check = false)
        in𝓖 = false
        #===========================================================================
         If F is successful, then H is positive definite, and we can safely look
         at the Newton step. If this is interior, we're done, if not, we're either
         in L or G. In L, the Newton step on the secular equation  stays in L and
         is safe to take. G requires more work.
        ===========================================================================#
        if issuccess(F)
            # H(λ) is PD, so we're in 𝓕

            # Algorithm 7.3.1 on p. 185 in [ConnGouldTointBook]
            # Step 1 was factorizing
            # Step 2
            s .= -∇f
            s .= F \ s

            # Check if step is approximately equal to the radius
            s₂ = norm(s, 2)
            if s₂ ≈ Δ
                H = update_H!(H, h)
                return tr_return(;
                    λ = λ,
                    ∇f = ∇f,
                    H = H,
                    s = s,
                    interior = false,
                    solved = true,
                    hard_case = false,
                    Δ = Δ,
                )
            end
            if s₂ < Δ # in 𝓖 because we're in 𝓕, but curve below Δ
                # we're in 𝓖 so λ is a new upper bound; λᴹ < λ
                in𝓖 = true
                λU = λ
            else # λ ∈ 𝓛
                # in 𝓛 λ is a *lower* bound instead
                λL = λ
            end

            # Step 3
            w = F.U' \ s

            # Step 4
            # Newton trial step
            λ⁺ = λ + ((s₂ - Δ) / Δ) * (s₂^2 / dot(w, w))
            if in𝓖
                w, u = λL_with_linpack(F)
                λL = max(λL, λ - dot(u, H * u))

                α, s_g, m_g = 𝓖_root(u, s, Δ, ∇f, H)
                # check hard case convergence (the κhard termination test in
                # section 7.3 of [ConnGouldTointBook]);
                # the test uses the uncorrected Newton step s, and s is only
                # replaced by the boundary-crossing step on success
                if α^2 * dot(u, H * u) ≤ κhard * (dot(s, H * s) + λ * Δ^2)
                    s .= s_g
                    H = update_H!(H, h)
                    return tr_return(;
                        λ = λ,
                        ∇f = ∇f,
                        H = H,
                        s = s,
                        interior = false,
                        solved = true,
                        hard_case = true,
                        Δ = Δ,
                        m = m_g,
                    )
                end
                # If not the hard case solution, try to factorize H(λ⁺)
                H = update_H!(H, h, λ⁺)
                F = cholesky(H; check = false)
                if issuccess(F) # Then we're in L, great! lemma 7.3.2
                    λ = λ⁺
                else # we landed in N, this is bad, so use bounds to approach L
                    λ = max(sqrt(λL * λU), λL + θ * (λU - λL))
                end
            else # in L, we can safely step
                λ = λ⁺
            end

            # check for convergence (the κeasy "easy case" termination criterion
            # in section 7.3 of [ConnGouldTointBook]; applies in 𝓛 as well as 𝓖)
            if abs(s₂ - Δ) ≤ κeasy * Δ
                H = update_H!(H, h)
                return tr_return(;
                    λ = λ,
                    ∇f = ∇f,
                    H = H,
                    s = s,
                    interior = false,
                    solved = true,
                    hard_case = false,
                    Δ = Δ,
                )
            end
        else # λ ∈ 𝓝, because the factorization failed (typo in CGT)
            # Use partial factorization to find δ and v such that
            # H(λ) + δ*e*e' = 0. All we can do here is to find a better
            # lower bound, we cannot apply the Newton step here.
            δ, v = λL_in_𝓝(H, F)
            λL = max(λL, λ + δ / dot(v, v)) # update lower bound
            λ = max(sqrt(λL * λU), λL + θ * (λU - λL)) # no converence possible, so step in bracket
        end
    end
    H = update_H!(H, h)
    tr_return(;
        λ = λ,
        ∇f = ∇f,
        H = H,
        s = s,
        interior = false,
        solved = false,
        hard_case = false,
        Δ = Δ,
    )
end

function λL_in_𝓝(H, F)
    T = eltype(F)
    n = first(size(F))
    δ = sum(abs2, view(F.factors, 1:(F.info-1), F.info)) - H[F.info, F.info]
    v = zeros(T, n)
    v[F.info] = 1
    for j = (F.info-1):-1:1
        vj = zero(T)
        for i = (j+1):F.info
            vj += F.factors[j, i] * v[i]
        end
        v[j] = -vj / F.factors[j, j]
    end
    return δ, v
end

function λL_with_linpack(F)
    T = eltype(F)
    n = first(size(F))
    U = F.U
    w = zeros(T, n)
    # LINPACK-style estimate of the direction of smallest singular value:
    # solve U'w = e where each eₖ = ±1 is chosen to maximize |wₖ|
    for k = 1:n
        num = dot(view(U, 1:(k-1), k), view(w, 1:(k-1)))
        ukk = U[k, k]
        w_p = (1 - num) / ukk
        w_m = (-1 - num) / ukk
        w[k] = abs(w_p) ≥ abs(w_m) ? w_p : w_m
    end
    sol = U \ w
    w, sol ./ norm(sol)
end

function 𝓖_root(u, s, Δ, ∇f, H)
    pa = sum(abs2, u)
    pb = 2 * dot(u, s)
    pd = sqrt(pb^2 - 4 * pa * (sum(abs2, s) - Δ^2))
    α₁ = (-pb + pd) / 2pa
    α₂ = (-pb - pd) / 2pa

    s₁ = s + α₁ * u
    m₁ = dot(∇f, s₁) + dot(s₁, H * s₁) / 2
    s₂ = s + α₂ * u
    m₂ = dot(∇f, s₂) + dot(s₂, H * s₂) / 2
    α, s, m = m₁ ≤ m₂ ? (α₁, s₁, m₁) : (α₂, s₂, m₂)
    α, s, m
end
