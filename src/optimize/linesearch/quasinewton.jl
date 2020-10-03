struct QNVars{T, Ty}
    d::T # search direction
    s::T # change in x
    y::Ty # change in successive gradients
end
function QNVars(x, g)
    QNVars(copy(g), copy(x), copy(x))
end
function preallocate_qn_caches(mstyle, x0)
    if mstyle === InPlace()
        # Maintain gradient and state pairs in QNVars
        cache = QNVars(x0, x0)
        return cache
    else
        return nothing
    end
end

function solve(problem::OptimizationProblem, x0, approach::LineSearch, options::OptimizationOptions, cache=preallocate_qn_caches(mstyle(problem), x0))
    _solve(mstyle(problem), problem, (x0, nothing), approach, options, cache)
end
function solve(problem::OptimizationProblem, s0::Tuple, approach::LineSearch, options::OptimizationOptions, cache=preallocate_qn_caches(mstyle(problem), first(s0)))
    _solve(mstyle(problem), problem, s0, approach, options, cache)
end

function _solve(mstyle, problem::OptimizationProblem, s0::Tuple, approach::LineSearch, options::OptimizationOptions, cache)
#    global_logger(options.logger)
    t0 = time()

    #==============
         Setup
    ==============#
    x0, B0 = s0
    T = eltype(x0)
    
    objvars = prepare_variables(problem, approach, x0, copy(x0), B0)
    P = initial_preconditioner(approach, x0)
    f0, ∇f0 = objvars.fz, norm(objvars.∇fz, Inf) # use user norm

    if any(initial_converged(approach, objvars, ∇f0, options))
        x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
        return ConvergenceInfo(approach, (P=P, B=B, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=0, time=time()-t0), options)
    end
    qnvars = QNVars(copy(objvars.∇fz), copy(objvars.∇fz), copy(objvars.∇fz))

    #==============================
             First iteration
    ==============================#
    objvars, P, qnvars = iterate(mstyle, qnvars, objvars, P, approach, problem, options)
    iter = 1
    # Check for gradient convergence
    is_converged = converged(approach, objvars, ∇f0, options)
    print_trace(approach, options, iter, t0, objvars)
    while iter < options.maxiter && !any(is_converged)
        iter += 1
        #==============================
                     iterate
        ==============================#
        objvars, P, qnvars = iterate(mstyle, qnvars, objvars, P, approach, problem, options, false)
        #==============================
                check convergence
        ==============================#
        is_converged = converged(approach, objvars, ∇f0, options)
        print_trace(approach, options, iter, t0, objvars)
    end
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    return ConvergenceInfo(approach, (P=P, B=B, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
end
function print_trace(approach::LineSearch, options, iter, t0, objvars)
    if !isa(options.logger, NullLogger) 
       println(@sprintf("iter: %d   time: %.4f   obj: %.4e   ||∇f||: %.4e    α: %.4e", iter, time()-t0, objvars.fz, norm(objvars.∇fz, Inf), objvars.α))
    end
end
function iterate(mstyle::InPlace, cache, objvars, P, approach::LineSearch, problem::OptimizationProblem, options::OptimizationOptions, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars

    Tf = typeof(fx)
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    y, d, s = cache.y, cache.d, cache.s

    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Update preconditioner
    P = update_preconditioner(scheme, x, P)
    # Update current gradient and calculate the search direction
    d = find_direction!(d, B, P, ∇fx, scheme) # solve Bd = -∇fx
    # real is needed to convert complex dots to actually being real
    φ = _lineobjective(mstyle, problem, ∇fz, z, x, d, fx, real(dot(∇fx, d)))

    # Perform line search along d
    # Also returns final step vector and update the state
    α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tf(1))

    
    @. s = α * d
    z = retract(problem, z, x, s)
 
    # Update approximation
    fz, ∇fz, B, s, y = update_obj!(problem, s, y, ∇fx, z, ∇fz, B, scheme, is_first)
    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=B, Pg=Pg, α=α), P, QNVars(d, s, y)
end
function print_trace(approach::LineSearch, options, iter, t0, objvars, Δ)
    if !isa(options.logger, NullLogger) 
        println(@sprintf("iter: %d   time: %f   f: %.4e   ||∇f||: %.4e    Δ: %.4e", iter, time()-t0, objvars.fz, norm(objvars.∇fz, Inf), Δ))
    end
end

function iterate(mstyle::OutOfPlace, cache, objvars, P, approach::LineSearch, problem::OptimizationProblem, options::OptimizationOptions, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    Tf = typeof(fx)
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    # Move nexts into currs
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)

    # Update preconditioner
    P = update_preconditioner(scheme, x, P)
    # Update current gradient and calculate the search direction
    d = find_direction(B, P, ∇fx, scheme) # solve Bd = -∇fx
    # real is needed to convert complex dots to actually being real
    φ = _lineobjective(mstyle, problem, ∇fz, z, x, d, fx, real(dot(∇fx, d)))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tf(1))

    # # Calculate final step vector and update the state
    s = @. α * d
    z = retract(problem, z, x, s)

    # Update approximation
    fz, ∇fz, B, s, y = update_obj(problem, s, ∇fx, z, ∇fz, B, scheme, is_first)

    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=B, Pg=Pg), P, QNVars(d, s, y)
end
