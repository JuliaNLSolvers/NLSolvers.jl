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

- `deltamin`: if a number is given, the solve stops once the radius falls below
  it. The default `nothing` disables the test.
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
    BTR(; deltamin = nothing, eta = 1e-4) <: TrustRegionUpdater

Basic trust region updater following, and named after [CGT]. `eta` is the
step-acceptance threshold: a trial step is accepted when the ratio of actual to
predicted reduction is at least `eta`. It must be positive to rule out cycling
(see p. 4 of Yuan's survey); values in `(0, 1/4)` are typical. `deltamin` is
consulted by the convergence test only: when it is a number, the solve stops
once the trust-region radius falls below it, and the default `nothing` disables
the test.
"""
struct BTR{TΔ,Tη}
    Δmin::TΔ
    η::Tη
end
BTR(Δmin) = BTR(Δmin, 1e-4)
BTR(; deltamin = nothing, eta = 1e-4) = BTR(deltamin, eta)
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
