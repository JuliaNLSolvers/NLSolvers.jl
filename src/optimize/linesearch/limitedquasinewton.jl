struct TwoLoopVars{T, TR, M}
    d::T # search direction
    S::TR # change in x
    Y::TR # change in successive gradients
    α::M
    ρ::M
end
function TwoLoopVars(x, memory)
    d = copy(x)

    S = [copy(x) for i = 1:memory]
    Y = [copy(x) for i = 1:memory]
    α = similar(x, memory)
    ρ = similar(x, memory)
    TwoLoopVars(d, S, Y, α, ρ)
end
lbfgs_vars(method, x) = TwoLoopVars(x, method.memory)
function solve(problem::OptimizationProblem, x0, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, options::OptimizationOptions, cache=preallocate_qn_caches(mstyle(problem), x0))
    solve(problem, (x0, nothing), approach, options, cache)
end
function solve(problem::OptimizationProblem, s0::Tuple, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, options::OptimizationOptions, cache=preallocate_qn_caches(mstyle(problem), first(s0)))
    _solve(InPlace(), problem, s0, approach, options, cache)
end

function _solve(mstyle, problem::OptimizationProblem, s0::Tuple, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, options::OptimizationOptions, cache)
    t0 = time()

    #==============
         Setup
    ==============#
    x0, B0 = s0
    T = eltype(x0)
    # Remove B from all of this
    objvars = prepare_variables(problem, approach, x0, copy(x0), B0)

    qnvars = lbfgs_vars(modelscheme(approach), objvars.x)
    ∇f0 = norm(objvars.∇fz, Inf) 
    f0 = objvars.fz
    P = initial_preconditioner(approach, x0)

    #========================
         First iteration
    ========================#
    objvars, qnvars, P = iterate(mstyle, 1, qnvars, objvars, P, approach, problem, problem, options)
    iter = 1
    # Check for gradient convergence
    is_converged = converged(approach, objvars, ∇f0, options)
    while iter <= options.maxiter && !any(is_converged)
        iter += 1
        # take a step and update approximation
        objvars, qnvars, P = iterate(mstyle, iter, qnvars, objvars, P, approach, problem, problem, options, false)
        # Check for gradient convergence
        is_converged = converged(approach, objvars, ∇f0, options)
    end
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    return ConvergenceInfo(approach, (P=P,B=B, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
end
function iterate(mstyle::InPlace, iter::Integer, qnvars::TwoLoopVars, objvars, P, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, prob::OptimizationProblem, problem::OptimizationProblem, options::OptimizationOptions, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch, K = modelscheme(approach), algorithm(approach), approach.scaling # make grabber and get a better name than K
    current_memory = min(iter-1, scheme.memory)
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    Tf = typeof(fz)

    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Update preconditioner
    P = update_preconditioner(scheme, x, P) # returns nothing????
    # Update current gradient and calculate the search direction
    d = find_direction!(scheme, copy(∇fz), qnvars, current_memory, K, P) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, problem, ∇fz, z, x, d, fx, real(dot(∇fx, d))) # real is needed to convert complex dots to actually being real

    # Perform line search along d
    α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tf(1))

    s = current_memory == length(qnvars.S) ? qnvars.S[1] : qnvars.S[1+current_memory]
    @. s = α * d
    z = retract(problem, z, x, s)
 
    # Update approximation
    fz, ∇fz, qnvars = update_obj!(problem, qnvars, α, x, ∇fx, z, ∇fz, current_memory, scheme)
    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=nothing, Pg=nothing), qnvars, P
end

function iterate(mstyle::OutOfPlace, cache, objvars, P, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, prob::OptimizationProblem, problem::OptimizationProblem, options::OptimizationOptions, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    # Move nexts into currs
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    Tf = typeof(fz)
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)
    # Update preconditioner
    P = update_preconditioner(scheme, x, P)
    # Update current gradient and calculate the search direction
    d = find_direction(B, P, ∇fx, scheme) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, problem, problem, ∇fz, z, x, d, fx, dot(∇fx, d))

    # # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

    # # Calculate final step vector and update the state
    s = @. α * d
    z = retract(problem, z, x, s)

    # Update approximation
    fz, ∇fz, B = update_obj(problem, s, ∇fx, z, ∇fz, B, scheme, is_first)

    return x, fx, ∇fx, z, fz, ∇fz, B, P
end
