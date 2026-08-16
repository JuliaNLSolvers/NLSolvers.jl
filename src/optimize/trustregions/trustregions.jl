abstract type TrustRegionUpdater end
"""
    TrustRegion(scheme, spsolve = NTR(); deltamin = nothing, eta = 1e-4, update_reject = true, eval_f_first = false)
    TrustRegion(; deltamin = nothing, eta = 1e-4, update_reject = true, eval_f_first = false)

A trust-region method that builds a local quadratic model from `scheme` (for
example `Newton()`, `BFGS()`, `SR1()`) and solves the constrained model problem
with the sub-problem solver `spsolve` (`NTR()`, `NWI()`, `Dogleg()`, `TCG()`).
The no-argument form uses `Newton()` with `NTR()`.

A trial step is accepted when the ratio of actual to model reduction satisfies
`R >= eta` (see [`BTR`](@ref)); accepted steps with `R < 1/4` shrink the region,
steps with `R >= 1/4` grow it, following Algorithm 4.1 of Nocedal & Wright (2nd
edition) and chapter 6 of Conn, Gould & Toint. A strictly positive `eta` avoids
the cycling that can occur at `eta = 0`, see Yuan's survey on trust-region
methods.

Keywords:

- `deltamin`: the radius floor of the stopping test; the solve stops once the
  radius falls below it. The default `nothing` resolves to the iterate's
  floating-point resolution (see [`BTR`](@ref)); pass `0` to disable the test.
- `eta`: the acceptance threshold described above.
- `update_reject`: if `true` (default), quasi-Newton schemes update the model
  approximation with the trial-point curvature pair even when the step is
  rejected; the trial gradient carries curvature information whether or not the
  step is taken (Nocedal & Wright, Algorithm 6.2). If `false`, the
  approximation is updated only on accepted steps. `Newton()` is unaffected:
  after a rejected step its Hessian is always restored to the current iterate's.
- `eval_f_first`: if `true`, only the objective value is computed at the trial
  point to decide acceptance, and the gradient (and Hessian for `Newton()`) is
  evaluated only when the step is accepted. This saves gradient evaluations
  when rejections are common and the gradient is expensive, but requires the
  objective to provide a standalone `f`, re-evaluates shared work for fused
  `fg`/`fgh` objectives on accepted steps, and implies that rejected steps never
  update the approximation (overriding `update_reject`).

The full set of radius-update constants (initial radius, growth and shrink
factors, the shrink interval, the radius cap) lives on [`BTR`](@ref); pass a
configured updater as the third argument, `TrustRegion(scheme, spsolve, BTR(...))`,
for full control. Sub-problem solver tolerances live on the solvers themselves,
e.g. `NTR(abstol = ...)`.
"""
struct TrustRegion{M,SP,D}
    scheme::M
    spsolve::SP
    Δupdate::D
    update_reject::Bool
    eval_f_first::Bool
end
TrustRegion(m, sp, Δupdate) = TrustRegion(m, sp, Δupdate, true, false)
summary(tr::TrustRegion) = "$(summary(modelscheme(tr))) with $(summary(algorithm(tr)))"
function initial_preconditioner(approach::TrustRegion, x)
    nothing
end
"""
    BTR(; deltamin = nothing, eta = 1e-4, delta0 = 20.0,
          t2 = 1/4, t3 = 1/4, t4 = 1/2, lambda34 = 0.0,
          gamma = 2.5, lambdagamma = 1/2, deltamax = 1e5, sigma = 1/4) <: TrustRegionUpdater

Basic trust region updater following, and named after [CGT]. It carries the
acceptance threshold and every constant of the radius-update rule:

- `eta`: the step-acceptance threshold. A trial step is accepted when the
  ratio of actual to predicted reduction is at least `eta`. It must be
  positive to rule out cycling (see p. 4 of Yuan's survey); values in
  `(0, 1/4)` are typical.
- `deltamin`: the radius floor of the stopping test. The solve stops once the
  radius falls below it. The default `nothing` resolves to the iterate's
  floating-point resolution, `eps(T) * max(1, ||x||_inf)`, below which no step
  can change the iterate; pass `0` to disable the test.
- `delta0`: the initial radius, used unless the `initial_Δ` keyword of `solve`
  overrides it.
- Accepted steps with ratio at least `t2` grow the radius by the factor
  `1 + lambdagamma * (gamma - 1)`, capped at `deltamax`.
- Accepted steps with ratio below `t2`, and rejected boundary steps, shrink
  the radius to a point in the interval `[t3 * ||p||, t4 * Δ]`, with
  `lambda34 ∈ [0, 1]` interpolating between the endpoints as in [CGT];
  Algorithm 4.1 on p. 69 of [N&W] is the `lambda34 = 1, t4 = 1/4` corner of
  this rule.
- Rejected interior steps shrink to `sigma * ||p||`, so retries are not
  wasted on reproducing the same interior solution.
"""
struct BTR{TΔ,T}
    Δmin::TΔ
    η::T
    Δ0::T
    t2::T
    t3::T
    t4::T
    λ34::T
    γ::T
    λγ::T
    Δmax::T
    σ::T
end
function BTR(;
    deltamin = nothing,
    eta = 1e-4,
    delta0 = 20.0,
    t2 = 1 / 4,
    t3 = 1 / 4,
    t4 = 1 / 2,
    lambda34 = 0.0,
    gamma = 2.5,
    lambdagamma = 1 / 2,
    deltamax = 1e5,
    sigma = 1 / 4,
)
    knobs = promote(
        float(eta),
        float(delta0),
        float(t2),
        float(t3),
        float(t4),
        float(lambda34),
        float(gamma),
        float(lambdagamma),
        float(deltamax),
        float(sigma),
    )
    BTR(deltamin, knobs...)
end
BTR(Δmin) = BTR(; deltamin = Δmin)
BTR(Δmin, η) = BTR(; deltamin = Δmin, eta = η)
TrustRegion(; deltamin = nothing, eta = 1e-4, update_reject = true, eval_f_first = false) =
    TrustRegion(Newton(), NTR(), BTR(deltamin, eta), update_reject, eval_f_first)
TrustRegion(
    m,
    sp = NTR();
    deltamin = nothing,
    eta = 1e-4,
    update_reject = true,
    eval_f_first = false,
) = TrustRegion(m, sp, BTR(deltamin, eta), update_reject, eval_f_first)
modelscheme(tr::TrustRegion) = tr.scheme
algorithm(tr::TrustRegion) = tr.spsolve

# annotate scheme here
solve(problem::OptimizationProblem, x0, scheme, options::OptimizationOptions) =
    solve(problem, (x0, nothing), TrustRegion(scheme, NWI()), options)

function solve(
    problem::OptimizationProblem,
    x0,
    scheme::Newton,
    options::OptimizationOptions,
)
    solve(problem, (x0, nothing), TrustRegion(scheme, NTR()), options)
end
function solve(
    problem::OptimizationProblem,
    x0,
    approach::TrustRegion,
    options::OptimizationOptions,
)
    solve(problem, (x0, nothing), approach, options)
end
include("optimize/inplace_loop.jl")
include("optimize/outofplace_loop.jl")
