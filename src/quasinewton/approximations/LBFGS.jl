# Nocedal (1980) based on 
# Specialized S and Y matrices are constructed and updated
struct TwoLoop end
struct CompactLimited end
struct LBFGS{TA,F,T,TP,Tskip}
    approx::TA
    type::F
    memory::T
    P::TP
    skip::Tskip
    function LBFGS(approx::TA, type::F, memory::T, P::TP, skip::Tskip) where {TA,F,T,TP,Tskip}
        if approx isa Direct && type isa TwoLoop
            throw(ArgumentError(
                "Direct L-BFGS with TwoLoop is not supported. " *
                "The two-loop recursion only works with the inverse Hessian approximation. " *
                "Use LBFGS(Inverse(), ...) or LBFGS() instead."
            ))
        end
        new{TA,F,T,TP,Tskip}(approx, type, memory, P, skip)
    end
end
summary(lbfgs::LBFGS{Inverse}) = "Inverse LBFGS"
summary(lbfgs::LBFGS{Direct}) = "Direct LBFGS"

hasprecon(::LBFGS{<:Inverse,<:Any,<:Any,<:Nothing}) = NoPrecon()
hasprecon(::LBFGS{<:Inverse,<:Any,<:Any,<:Any}) = HasPrecon()

LBFGS(m::Int) = LBFGS(Inverse(), TwoLoop(), m, nothing, NoPDSkip())
LBFGS(approx, m::Int) = LBFGS(approx, TwoLoop(), m, nothing, NoPDSkip())
LBFGS(approx, type, memory, P) = LBFGS(approx, type, memory, P, NoPDSkip())
LBFGS(; memory = 5, inverse = true, skip = NoPDSkip()) = LBFGS(inverse ? Inverse() : Direct(), TwoLoop(), memory, nothing, skip)
init_B(aproach::LBFGS, ::Nothing, x0) = nothing
"""
	q holds gradient at current state
	history is a named tuple of S and Y
	memory is the number of past (s,y) pairs we have
"""
function find_direction!(
    scheme::LBFGS{<:Inverse,<:TwoLoop},
    q,
    qnvars,
    memory,
    scaling,
    precon,
)

    S, Y = qnvars.S, qnvars.Y
    α, ρ = qnvars.α, qnvars.ρ
    d = qnvars.d

    # Backward pass
    @inbounds for i = memory:-1:1
        α[i] = ρ[i] * real(dot(S[i], q))
        q .-= α[i] .* Y[i]
    end
    # Copy scaled or preconditioned q into s for forward pass
    if memory > 0
        if scaling isa InitialScaling{<:ShannoPhua} && precon isa Nothing # we need a pair to scale
            k = scaling(S[memory], Y[memory])
            @. d = k * q
        elseif !(precon isa Nothing)
            mul!(d, precon, q)
        end
    else
        d .= q
    end
    # Forward pass
    @inbounds for i = 1:memory
        β = ρ[i] * real(dot(Y[i], d))
        d .+= S[i] .* (α[i] - β)
    end
    # Negate search direction
    negate(InPlace(), d)
end
function update_obj!(
    problem::OptimizationProblem,
    qnvars,
    α,
    x,
    ∇fx,
    z,
    ∇fz,
    current_memory::Integer,
    scheme::LBFGS{<:Inverse,<:TwoLoop},
    scale = nothing;
    skip_data = nothing,
)
    # Calculate final step vector and update the state
    fz, ∇fz = upto_gradient(problem, ∇fz, z)
    # add Project gradient

    # Quasi-Newton update
    qnvars = update!(scheme, qnvars, ∇fx, ∇fz, current_memory, skip_data)

    return fz, ∇fz, qnvars
end
@inbounds function update!(
    scheme::LBFGS{<:Inverse,<:TwoLoop},
    qnvars,
    ∇fx,
    ∇fz,
    current_memory,
    skip_data = nothing,
)
    S, Y, ρ = qnvars.S, qnvars.Y, qnvars.ρ
    d = qnvars.d  # holds α * d (the step vector) from the caller
    n = length(S)
    m = min(n, 1 + current_memory)

    # Compute y as a temporary before deciding whether to store the pair
    y_candidate = ∇fz - ∇fx

    # Check skip condition — if triggered, don't store this (s, y) pair
    if skip_data !== nothing && should_skip(scheme.skip, d, y_candidate, skip_data)
        return TwoLoopVars(d, S, Y, qnvars.α, ρ, current_memory)
    end

    # Rotate memory if full
    if current_memory == n
        s1, y1 = S[1], Y[1] # hoisting, no allocation
        for i = 2:n
            S[i-1] = S[i]
            Y[i-1] = Y[i]
            ρ[i-1] = ρ[i]
        end
        S[end] = s1
        Y[end] = y1
    end
    # Store the (s, y) pair
    @. S[m] = d
    @. Y[m] = y_candidate
    ρ[m] = 1 / dot(S[m], Y[m])
    TwoLoopVars(d, S, Y, qnvars.α, ρ, m)
end
