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
    objvars::NamedTuple;
    initial_Δ = 20.0,
)
    if !(mstyle(problem) === InPlace()) && !(approach.spsolve == Dogleg())
        throw(
            ErrorException("solve() not defined for OutOfPlace() with TrustRegion solvers"),
        )
    end
    if approach.eval_f_first &&
       problem.objective isa ScalarObjective &&
       problem.objective.f === nothing
        throw(
            ArgumentError(
                "eval_f_first = true requires the objective to have a standalone f; supply ScalarObjective(f = ..., ...)",
            ),
        )
    end
    t0 = time()
    T = eltype(objvars.z)
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
    # Newton's model update replaces B with the Hessian at the trial point, so
    # a copy of the Hessian at the current iterate is kept to restore after a
    # rejected step.
    Bcache =
        modelscheme(approach) isa Newton && objvars.B !== nothing ? copy(objvars.B) :
        nothing

    objvars, Δkp1, reject, qnvars =
        iterate!(p, objvars, Δk, approach, problem, options, qnvars, Bcache, false)

    iter = 1
    callback_stopped = false
    # Check for convergence
    is_converged = converged(approach, objvars, ∇f0, options, reject, Δkp1)
    callback_stopped = _check_callback(options.callback, (iter=iter, time=time()-t0, state=(objvars..., Δ=Δkp1, rejected=reject)))
    while iter <= options.maxiter && !any(is_converged) && !callback_stopped
        iter += 1
        objvars, Δkp1, reject, qnvars =
            iterate!(p, objvars, Δkp1, approach, problem, options, qnvars, Bcache, false)

        # Check for convergence
        is_converged = converged(approach, objvars, ∇f0, options, reject, Δkp1)
        print_trace(approach, options, iter, t0, objvars, Δkp1)
        callback_stopped = _check_callback(options.callback, (iter=iter, time=time()-t0, state=(objvars..., Δ=Δkp1, rejected=reject)))
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
    Bcache,
    scale,
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

    z = retract(problem, z, x, spr.p)

    if approach.eval_f_first
        # Only the objective value is needed to decide acceptance; the gradient
        # (and Hessian for Newton) is evaluated on acceptance below.
        fz = value(problem.objective, z)
    else
        if scheme isa Newton && Bcache !== nothing
            # tr_trial_eval! replaces B with the Hessian at the trial point;
            # keep the Hessian at x so it can be restored on rejection
            Bcache = _copyto(mstyle(problem), Bcache, B)
        end
        fz, ∇fz, B = tr_trial_eval!(problem, z, ∇fz, B, scheme)
    end

    # Δf is often called ared or Ared for actual reduction. I prefer "change in"
    # f, or Delta f. Δm may be zero or negative when the sub-problem solver
    # cannot improve the model (a ridge in the positive-indefinite case, or a
    # scaling for which ||∇f|| < tol cannot be satisfied); tr_acceptance
    # handles that case explicitly.
    Δf = fx - fz
    R, accept = tr_acceptance(Δf, Δm, T(approach.Δupdate.η))
    Δkp1 = update_trust_region(spr, R, accept, p)

    if accept
        if approach.eval_f_first
            if scheme isa Newton
                fz, ∇fz, B = tr_trial_eval!(problem, z, ∇fz, B, scheme)
            else
                ∇fz = gradient_only(problem.objective, ∇fz, z)
            end
        end
        B, s, y = tr_update_approx!(y, spr.p, ∇fx, ∇fz, B, scheme, scale)
    else
        if !approach.eval_f_first && approach.update_reject
            # Rejected steps may still update the approximation: the trial
            # gradient carries curvature information whether or not the step
            # is taken (Nocedal & Wright, Algorithm 6.2). Newton is exempt
            # (see below); opt out with TrustRegion(update_reject = false).
            B, s, y = tr_update_approx!(y, spr.p, ∇fx, ∇fz, B, scheme, scale)
        end
        z = _copyto(mstyle(problem), z, x)
        fz = fx
        if !approach.eval_f_first
            ∇fz = _copyto(mstyle(problem), ∇fz, ∇fx)
            if scheme isa Newton && Bcache !== nothing
                # B holds the Hessian of the rejected trial point; restore the
                # Hessian at x so the next model is built from accepted state
                B = _restore_B(mstyle(problem), B, Bcache)
            end
        end
    end
    return (x = x, fx = fx, ∇fx = ∇fx, z = z, fz = fz, ∇fz = ∇fz, B = B, Pg = nothing),
    Δkp1,
    !accept,
    QNVars(d, s, y)
end

_restore_B(::InPlace, B, Bcache) = copyto!(B, Bcache)
_restore_B(::OutOfPlace, B, Bcache) = Bcache

# The acceptance decision. A step is accepted when the ratio R of actual
# reduction Δf to model reduction Δm is at least η. We accept all steps with
# R ≥ η for η ∈ (0, 1/4). See p. 415 of [SOREN] and p. 79 as well as Theorems
# 4.5 and 4.6 of [N&W]. η = 0 might cycle, see p. 4 of [YUAN], so BTR requires
# a positive η. Non-finite objective values and degenerate model decreases are
# handled explicitly rather than through NaN/Inf comparison semantics.
function tr_acceptance(Δf, Δm, η)
    R = Δf / Δm
    # The acceptance is deliberately based on the ratio alone, without
    # requiring a model decrease. When the sub-problem step predicts a model
    # increase (Δm < 0, possible for indefinite approximations) and the
    # objective also increases, the ratio is positive and the step may be
    # accepted although it moves uphill; this nonmonotone acceptance can
    # escape regions where the model is poor. Δm = 0 gives R = ±Inf (accept
    # or reject by the sign of Δf) and Δf = Δm = 0 gives NaN, which rejects,
    # as does a NaN objective value.
    return R, !isnan(R) && R >= η
end

function update_trust_region(spr, R, accept, p)
    T = eltype(p)
    t2 = T(1) / 4
    t3 = t2 # could differ!
    t4 = T(1) / 2
    λ34 = T(0) / 2
    γ = T(2.5) # gamma for grow
    λγ = T(1) / 2 # distance along growing interval ∈ (0, 1]
    Δmax = T(10)^5 # restrict the largest step
    σ = T(1) / 4

    Δk = spr.Δ
    # Shrinking picks a radius from the interval [t3*||p||, t4*Δk], with
    # λ34 ∈ [0, 1] interpolating between the endpoints as in [CGT]; [N&W]
    # Algorithm 4.1 on p. 69 is the λ34 = 1, t4 = 1/4 corner of this rule.
    # λ34 is currently fixed at 0, so the shrink is Δk/2.
    if !accept
        if spr.interior
            # If you reject an interior solution, make sure that the next
            # delta is smaller than the current step. Otherwise you waste
            # steps reducing Δk by constant factors while each solution
            # will be the same.
            Δkp1 = σ * norm(p)
        else
            Δkp1 = λ34 * norm(p, 2) * t3 + (1 - λ34) * Δk * t4
        end
    else
        # While we also accept the steps in the case that η <= R < t2, we do
        # not trust them too much, and the radius is restricted.
        if R < t2
            Δkp1 = λ34 * norm(p, 2) * t3 + (1 - λ34) * Δk * t4
        else
            Δkp1 = min(λγ * Δk + (1 - λγ) * Δk * γ, Δmax)
        end
    end
    return Δkp1
end
