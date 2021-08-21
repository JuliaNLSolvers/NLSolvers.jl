"""
# SPG
## Constructor
```julia
    SPG(; memory=8)
```

## Description
SPG is the Spectral Projected Gradient method that combines the spectral (Barzilai and Borwein, BB) method with the Grippo-Lampariello-Lucidi (GLL) non-monotoneous line search. The GLL line search requires a memory length to be specified with a defualt in this implementation of 8. 

## References
[1] https://www.ime.usp.br/~egbirgin/publications/bmr.pdf
"""
struct SPG{T}
    memory::T
end
SPG(; memory=8) = SPG(; memory)
summary(::SPG) = "SPG"
modelscheme(::SPG) = BB()

function solve(prob::OptimizationProblem, x0, scheme::ActiveBox, options::OptimizationOptions)
    t0 = time()

    x0, λ = x0, one(eltype(x0))
    lower, upper = bounds(prob)


    !any(clamp.(x0, lower, upper) .!= x0) || error("Initial guess not in the feasible region")

    linesearch = GrippoLamparielloLucidi()
    mstyle = OutOfPlace()

    objvars = prepare_variables(prob, scheme, x0, copy(x0), λ)
    f0, ∇f0 = objvars.fz, norm(objvars.∇fz, Inf) # use user norm
    fz, ∇fz = objvars.fz, objvars.∇fz # use user norm
    fx, ∇fx = fz, copy(∇fz)
    B = B0
    x, z = copy(x0), copy(x0)
    Tf = typeof(fz)
    is_first=false
    Ix = Diagonal(z.*0 .+ 1)
    for iter = 1:options.maxiter
        x = copy(z)
        fx = copy(fz)
        ∇fx = copy(∇fz)

        ϵ = min(norm(clamp.(x.-∇fx, lower, upper).-x), ϵbounds) # Kelley 5.41 and just after (83) in [1]
        activeset = is_ϵ_active.(x, lower, upper, ∇fx, ϵ)

        Hhat = diagrestrict.(B, activeset, activeset', Ix)
        # Update current gradient and calculate the search direction
        d = clamp.(x.-Hhat\∇fx, lower, upper).-x # solve Bd = -∇fx  #use find_direction here
        φ = _lineobjective(mstyle, prob, ∇fz, z, x, d, fz, dot(∇fz, d))

        # Perform line search along d
        # Also returns final step vector and update the state
        α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tf(1), ∇fz, activeset, lower, upper, x, d, ∇fx, activeset)
        # # Calculate final step vector and update the state
        s = @. α * d
        z = @. x + s
        s = clamp.(z, lower, upper) - x
        z = x + s

        # Update approximation
        fz, ∇fz, B, s, y = update_obj(prob.objective, s, ∇fx, z, ∇fz, B, Newton(), is_first)
        if norm(x.-clamp.(x.-∇fz, lower, upper), Inf) < options.g_abstol
            return ConvergenceInfo(scheme, (prob=prob, B=B, ρs=norm(x.-z), ρx=norm(x), solution=z, fx=fx, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
        end
    end
  z, fz, options.maxiter
  return ConvergenceInfo(scheme, (prob=prob, B=B, ρs=norm(x.-z), ρx=norm(x), solution=z, fx=fx, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
end
