# Add a calculate_γ from Bertsekas.
# This is when an initial α = 1 far oversteps the first point that hits a boundary
# on the piece-wise linear projected search path. This could be done always, never
# or if last search required a lot of line search reductions.

"""
# ActiveBox
## Constructor
```julia
    ActiveBox(; factorize = cholesky, epsilon = 1e-8)
```

`factorize` is a function that factorizes the restricted Hessian, `epsilon` determines the threshold for whether a bound is approximately active or not, see eqn. (32) in [1].

## Description
ActiveBox second order for bound constrained convex optimization. It's an active set and allows for rapid exploration of the constraint face. It employs a modified Armijo-line search that takes the active set into account. Details can be found in [1].

## References
- 1) http://www.mit.edu/~dimitrib/ProjectedNewton.pdf
- 2) Iterative Methods for Optimization https://archive.siam.org/books/textbooks/fr18_book.pdf
"""
struct ActiveBox{F,T}
    factorize::F
    ϵ::T
end
ActiveBox(; factorize = cholesky, epsilon = nothing) = ActiveBox(factorize, epsilon)
summary(::ActiveBox) = "ActiveBox"
modelscheme(::ActiveBox) = Newton()
"""
    diagrestrict(x, c, i)

Returns the correct element of the Hessian according to the active set and the diagonal matrix described in [1].

[1] http://www.mit.edu/~dimitrib/ProjectedNewton.pdf
"""
function diagrestrict(x::T, ci, cj, i) where {T}
    if !(ci | cj)
        # If not binding, then return the value
        return x
    else
        # If binding, then return 1 if the diagonal or 0 otherwise
        T(i)
    end
end

function is_ϵ_active(x, lower, upper, ∇fx, ϵ∇f = eltype(x)(0))
    # it is requied that ϵ ⩽ min(U_i - L_i)/2 to uniquely choose
    # an underestimate of the inactive set or else there would be
    # two ways of defining 𝓐^ϵ.
    lowerbinding = x <= lower + ϵ∇f
    upperbinding = x >= upper - ϵ∇f

    pointing_down = ∇fx >= 0
    pointing_up = ∇fx <= 0

    lower_active = lowerbinding && pointing_down
    upper_active = upperbinding && pointing_up

    lower_active || upper_active
end
isbinding(i, j) = i & j

factorize(ab::ActiveBox, M) = ab.factorize(M)
function solve(
    problem::OptimizationProblem,
    x0,
    approach::ActiveBox,
    options::OptimizationOptions,
)
    B0 = false * x0 * x0' + I
    s0 = (x0, B0)
    _solve(problem, s0, approach, options)
end
function solve(
    problem::OptimizationProblem,
    s0::Tuple,
    approach::ActiveBox,
    options::OptimizationOptions,
)
    _solve(problem, s0, approach, options)
end

function _solve(
    prob::OptimizationProblem,
    s0::Tuple,
    scheme::ActiveBox,
    options::OptimizationOptions,
)
    t0 = time()
    x0, B0 = s0

    lower, upper = bounds(prob)
    if isnothing(scheme.ϵ)
        ϵbounds = minimum(Δvec(upper,lower))/2 # [1, pp. 100], [2, 5.41] 
    else
        ϵbounds = scheme.ϵ
    end

    check_bounded(x0, lower, upper)
    linesearch = ArmijoBertsekas()
    mstyle = OutOfPlace()

    objvars = prepare_variables(prob, scheme, x0, copy(x0), B0)
    f0, ∇f0 = objvars.fz, norm(objvars.∇fz, Inf) # use user norm
    fz, ∇fz = objvars.fz, objvars.∇fz # use user norm
    fx, ∇fx = fz, copy(∇fz)
    B = B0
    x, z = copy(x0), copy(x0)
    Tf = typeof(fz)
    is_first = false
    Ix = Diagonal(z .* 0 .+ 1)
    for iter = 1:options.maxiter
        x = copy(z)
        fx = copy(fz)
        ∇fx = copy(∇fz)
        ϵ = min(norm(clamp.(x .- ∇fx, lower, upper) .- x), ϵbounds) # Kelley 5.41 and just after (83) in [1]
        activeset = is_ϵ_active.(x, lower, upper, ∇fx, ϵ)
        Hhat = diagrestrict.(B, activeset, activeset', Ix)
        # Update current gradient and calculate the search direction
        HhatFact = factorize(scheme, Hhat)
        d = -(HhatFact \ ∇fx) # use find_direction here

        φ = (; prob, ∇fz, z, x, p = d, φ0 = fz, dφ0 = dot(∇fz, d))

        # Perform line search along d
        # Also returns final step vector and update the state
        α, f_α, ls_success, z = find_steplength(
            mstyle,
            linesearch,
            φ,
            Tf(1),
            ∇fz,
            activeset,
            lower,
            upper,
            x,
            d,
            ∇fx,
            activeset,
        )
        # # Calculate final step vector and update the state
        s = @. x - z

        # Update approximation
        fz, ∇fz, B, s, y =
            update_obj(prob.objective, s, ∇fx, z, ∇fz, B, Newton(), is_first, nothing)
        if norm(x .- clamp.(x .- ∇fz, lower, upper), Inf) < options.g_abstol
            return ConvergenceInfo(
                scheme,
                (
                    prob = prob,
                    B = B,
                    ρs = norm(Δvec(x,z)),
                    ρx = norm(x),
                    solution = z,
                    fx = fx,
                    minimum = fz,
                    ∇fz = ∇fz,
                    f0 = f0,
                    ∇f0 = ∇f0,
                    iter = iter,
                    time = time() - t0,
                ),
                options,
            )
        end
        if _check_callback(
            options.callback,
            (
                iter = iter,
                time = time()-t0,
                state = (x = x, z = z, fz = fz, ∇fz = ∇fz, B = B, activeset = activeset),
            ),
        )
            break
        end
    end
    iter = options.maxiter
    return ConvergenceInfo(
        scheme,
        (
            prob = prob,
            B = B,
            ρs = norm(Δvec(x,z)),
            ρx = norm(x),
            solution = z,
            fx = fx,
            minimum = fz,
            ∇fz = ∇fz,
            f0 = f0,
            ∇f0 = ∇f0,
            iter = iter,
            time = time() - t0,
        ),
        options,
    )
end

"""
# ArmijoBertsekas
## Constructor
```julia
    ArmijoBertsekas()
```
## Description
ArmijoBertsekas is the modified Armijo backtracking line search described in [1]. It takes into account whether an element of the gradient is active or not.

## References
[1] http://www.mit.edu/~dimitrib/ActiveBox.pdf
"""
struct ArmijoBertsekas{T1,T2,T3,TR} <: LineSearcher
    ratio::T1
    decrease::T1
    maxiter::T2
    interp::T3
    steprange::TR
    verbose::Bool
end
ArmijoBertsekas(;
    ratio = 0.5,
    decrease = 1e-4,
    maxiter = 50,
    steprange = (0.0, Inf),
    interp = FixedInterp(),
    verbose = false,
) = ArmijoBertsekas(ratio, decrease, maxiter, interp, steprange, verbose)

function find_steplength(
    mstyle,
    ls::ArmijoBertsekas,
    φ::T,
    λ,
    ∇fx,
    Ibool,
    lower,
    upper,
    x,
    p,
    g,
    activeset,
) where {T}
    #== unpack ==#
    φ0, dφ0 = φ.φ0, φ.dφ0
    Tf = typeof(φ0)
    ratio, decrease, maxiter, verbose =
        Tf(ls.ratio), Tf(ls.decrease), ls.maxiter, ls.verbose

    iter, α, β = 0, λ, λ # iteration variables
    x⁺ = box_retract.(lower, upper, x, p, α)
    f_α = (; ϕ = φ.prob.objective.f(x⁺))  # initial function value

    if verbose
        println("Entering line search with step size: ", λ)
        println("Initial value: ", φ0)
        println("Value at first step: ", f_α)
    end

    is_solved =
        isfinite(f_α.ϕ) &&
        f_α.ϕ <= φ0 - decrease * sum(Bertsekas_R(α),zip(x, x⁺, g, p, activeset))
    while !is_solved && iter <= maxiter
        iter += 1
        β, α = α, α / 2
        x⁺ .= box_retract.(lower, upper, x, p, α)
        f_α = (; ϕ = φ.prob.objective.f(x⁺))  # initial function value
        #        β, α, f_α = interpolate(ls.interp, x->φ, φ0, dφ0, α, f_α.ϕ, ratio)
        is_solved =
            isfinite(f_α.ϕ) &&
            f_α.ϕ <= φ0 - decrease * sum(Bertsekas_R(α),zip(x, x⁺, g, p, activeset))
    end

    ls_success = iter >= maxiter ? false : true

    if verbose
        !ls_success && println("maxiter exceeded in backtracking")
        println("Exiting line search with step size: ", α)
        println("Exiting line search with value: ", f_α)
    end
    return α, f_α, ls_success, x⁺
end

# The two cases of the acceptance sum in eq. (37) of [1], whose right-hand
# side is nonnegative: Bertsekas steps along x - αp with p = D∇f (eqs. (33),
# (34)), and here p = -D∇f, so the inactive term α*∂f*p flips sign.
struct Bertsekas_R{A}
    α::A
end
(b::Bertsekas_R{A})(data) = b(data...)
(b::Bertsekas_R{A})(x, x⁺, g, p, i) where A = i ? g * (x - x⁺) : -b.α * p * g
bertsekas_R(x, x⁺, g, p, α, i) = Bertsekas_R(α)(x, x⁺, g, p, i)


# defined univariately
# should be a "manifodl"
box_retract(lower, upper, x, p, α) = min(upper, max(lower, x + α * p))
