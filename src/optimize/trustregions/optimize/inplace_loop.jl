function solve(
    problem::OptimizationProblem,
    s0::Tuple,
    approach::TrustRegion,
    options::OptimizationOptions;
    initial_Δ = 20.0,
)
    x0, B0 = s0
    objvars = prepare_variables(problem, approach, copy(x0), copy(x0), B0)
    solve(problem, approach, options, objvars; initial_Δ)
end
function solve(
    problem::OptimizationProblem,
    approach::TrustRegion,
    options::OptimizationOptions,
    objvars;
    initial_Δ = 20.0,
)

    trs_outofplace_check(approach.spsolve,problem)
    t0 = time()
    T = eltype(objvars.z)
    Δmin = cbrt(eps(T))
    Δk = T(initial_Δ)
    f0, ∇f0 = objvars.fz, norm(objvars.∇fz, Inf) # use user norm

    if any(initial_converged(approach, objvars, ∇f0, options, false, Δk))
        return ConvergenceInfo(
            approach,
            (
                Δ = Δk,
                ρs = norm(objvars.x .- objvars.z),
                ρx = norm(objvars.x),
                solution = objvars.z,
                fx = objvars.fx,
                minimum = objvars.fz,
                ∇fz = objvars.∇fz,
                f0 = f0,
                ∇f0 = ∇f0,
                iter = 0,
                time = time() - t0,
            ),
            options,
        )
    end
    qnvars = QNVars(objvars.z, objvars.z)
    p = copy(objvars.x)

    objvars, Δkp1, reject, qnvars =
        iterate!(p, objvars, Δk, approach, problem, options, qnvars, false)

    iter = 1
    # Check for convergence
    is_converged = converged(approach, objvars, ∇f0, options, reject, Δkp1)
    while iter <= options.maxiter && !any(is_converged)
        iter += 1
        objvars, Δkp1, reject, qnvars =
            iterate!(p, objvars, Δkp1, approach, problem, options, qnvars, false)

        # Check for convergence
        is_converged = converged(approach, objvars, ∇f0, options, reject, Δkp1)
        print_trace(approach, options, iter, t0, objvars, Δkp1)
    end
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    return ConvergenceInfo(
        approach,
        (
            Δ = Δkp1,
            ρs = norm(x .- z),
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
function print_trace(approach::TrustRegion, options, iter, t0, objvars, Δ)
    if false
        println(
            @sprintf(
                "iter: %d   time: %f   f: %.4e   ||∇f||: %.4e    Δ: %.4e",
                iter,
                time() - t0,
                objvars.fz,
                norm(objvars.∇fz, Inf),
                Δ
            )
        )
    end
end

function iterate!(
    p,
    objvars,
    Δk,
    approach::TrustRegion,
    problem,
    options,
    qnvars,
    scale = nothing,
)
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    T = eltype(x)
    scheme, subproblemsolver = modelscheme(approach), algorithm(approach)
    y, d, s = qnvars.y, qnvars.d, qnvars.s
    fx = fz

    x = _copyto(mstyle(problem), x, z)
    ∇fx = _copyto(mstyle(problem), ∇fx, ∇fz)

    spr = subproblemsolver(∇fx, B, Δk, p, scheme, problem.mstyle; abstol = 1e-10)
    Δm = -spr.mz

    # Grab the model value, m. If m is zero, the solution, z, does not improve
    # the model value over x. If the model is not converged, but the optimal
    # step is inside the trust region and gives a zero improvement in the objec-
    # tive value, we may conclude that "something" is wrong. We might be at a
    # ridge (positive-indefinite case) for example, or the scaling of the model
    # is such that we cannot satisfy ||∇f|| < tol.
    if abs(spr.mz) < eps(spr.mz)
        # set flag to check for problems
    end

    z = retract(problem, z, x, spr.p)
    # Update before acceptance, to keep adding information about the hessian
    # even when the step is not "good" enough.

    y, d, s = qnvars.y, qnvars.d, qnvars.s
    fx = fz
    # Should build a good code for picking update model.

    fz, ∇fz, B, s, y = update_obj!(problem, spr.p, y, ∇fx, z, ∇fz, B, scheme, scale)

    # Δf is often called ared or Ared for actual reduction. I prefer "change in"
    # f, or Delta f.
    Δf = fx - fz

    # Calculate the ratio of actual improvement over predicted improvement.
    R = Δf / Δm

    Δkp1, reject_step = update_trust_region(spr, R, p)

    if reject_step
        z = _copyto(mstyle(problem), z, x)
        ∇fz = _copyto(mstyle(problem), ∇fz, ∇fx)
        fz = fx
        # This is correct because 
        s = _scale(mstyle(problem), s, s, 0) # z - x
        y = _scale(mstyle(problem), y, y, 0) # ∇fz - ∇fx
        # and will cause quasinewton updates to not update
        # this seems incorrect as it's already updated, should hold off here
    end
    return (x = x, fx = fx, ∇fx = ∇fx, z = z, fz = fz, ∇fz = ∇fz, B = B, Pg = nothing),
    Δkp1,
    reject_step,
    QNVars(d, s, y)
end

function update_trust_region(spr, R, p)
    T = eltype(p)
    # Choosing a parameter > 0 might be preferable here. See p. 4 of Yuans survey
    # We want to avoid cycles, but we also need something that takes very small
    # steps when convergence is hard to achieve.
    α = T(0) / 7 # acceptance ratio
    t2 = T(1) / 4
    t3 = t2 # could differ!
    t4 = T(1) / 2
    λ34 = T(0) / 2
    γ = T(2.5) # gamma for grow
    λγ = T(1) / 2 # distance along growing interval ∈ (0, 1]
    Δmax = T(10)^5 # restrict the largest step
    σ = T(1) / 4

    Δk = spr.Δ
    # We accept all steps larger than α ∈ [0, 1/4). See p. 415 of [SOREN] and
    # p.79 as well as  Theorem 4.5 and 4.6 of [N&W]. An α = 0 might cycle,
    # see p. 4 of [YUAN].
    if !(α <= R)
        if spr.interior
            # If you reject an interior solution, make sure that the next
            # delta is smaller than the current step. Otherwise you waste
            # steps reducing Δk by constant factors while each solution
            # will be the same.
            Δkp1 = σ * norm(p)
        else
            Δkp1 = λ34 * norm(p, 2) * t3 + (1 - λ34) * Δk * t4
        end
        reject_step = true
    else
        # While we accept also the steps in the case that α <= Δf < t2, we do not
        # trust it too much. As a result, we restrict the trust region radius. The
        # new trust region radius should be set to a radius Δkp1 ∈ [t3*||d||, t4*Δk].
        # We use the number λ34 ∈ [0, 1] to move along the interval. [N&W] sets
        # λ34 _= 1 and t4 = 1/4, see Algorithm 4.1 on p. 69.
        if R < t2
            Δkp1 = λ34 * norm(p, 2) * t3 + (1 - λ34) * Δk * t4
        else
            Δkp1 = min(λγ * Δk + (1 - λγ) * Δk * γ, Δmax)
        end
        reject_step = false
    end
    return Δkp1, reject_step
end
