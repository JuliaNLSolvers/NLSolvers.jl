"""
  TCG is a trust region sub-problem solver that approximately solves the quadratic
  programming problem subject to the solution being in a Euclidean Ball. It is
  appropriate for all types of Hessians. It solves the problem by using a conjugate
  gradient approach to solving the unconstrained quadratic optimization problem. If
  an interior solution exists it will be found according to the usual CG theory. If the
  problem is unbounded (negative curvature) or the solution is outside of the trust region
  we step to the boundary.

  TCG accepts all Hessians and is therefore well-suited for Newton's method for
  any vexity. It is relatively cheap for (very) large problems.

  The implementation follows Algorithm 7.5.1 on p. 205 of [ConnGouldTointBook] and
  the algorithm on p. 628 in [Steihaug1983]
"""
struct TCG <: TRSPSolver
end
summary(::TCG) = "Steihaug-Toint Truncated CG"

function (ms::TCG)(∇f, H, Δ::T, s, scheme, λ0=0; abstol=1e-10, maxiter=min(5,length(s)), κeasy=T(1)/10, κhard=T(2)/10) where T
    M = I
    # We only know that TCG has its properties if we start with s = 0
    s .= T(0)
    # g is the gradient of the quadratic model with the step α*p
    if all(iszero, ∇f)
        return tr_return(;λ=NaN, ∇f=∇f, H=H, s=s, interior=true, solved=true, hard_case=false, Δ=Δ)
    end
    g = copy(∇f)
    v = M\g
    p = -v
    for i = 1:maxiter
        # Check for negative curvature
        κ = dot(p, H*p)

        c = dot(s, M*s)
        b′ = dot(s, M*p) # b/2
        a = dot(p, M*p)

        if κ <= 0
          	# This branch just catches that sigma can be nan if a is exactly 0.
        	  if !iszero(a)
	              σ = (-b′ + sqrt(b′^2 + a*(Δ^2 - c)))/a
                @. s = s + σ*p
            end
            return tr_return(;λ=NaN, ∇f=∇f, H=H, s=s, interior=false, solved=true, hard_case=false, Δ=Δ)
        end

        α = dot(g, v)/κ

        if a*α^2 + α*2*b′ + c >= Δ^2
            σ = (-b′ + sqrt(b′^2 + a*(Δ^2 - c)))/a
            @. s = s + σ*p
            return tr_return(;λ=NaN, ∇f=∇f, H=H, s=s, interior=false, solved=true, hard_case=false, Δ=Δ)
        end
        @. s = s + α*p
        den = dot(g, v)
        g .= g .+ α.*(H*p)
        v .= M\g
        β = dot(g, v)/den
        @. p = -v + β*p
    end
    tr_return(;λ=NaN, ∇f=∇f, H=H, s=s, interior=true, solved=true, hard_case=false, Δ=Δ)
end