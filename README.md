# NLSolvers

[![Build Status](https://github.com/JuliaNLSolvers/NLSolvers.jl/workflows/CI/badge.svg)](https://github.com/JuliaNLSolvers/NLSolvers.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Code Coverage](http://codecov.io/github/JuliaNLSolvers/NLSolvers.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaNLSolvers/NLSolvers.jl?branch=master)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianlsolvers.github.io/NLSolvers.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianlsolvers.github.io/NLSolvers.jl/dev)

Optimization, curve fitting, and systems of nonlinear equations for Julia.

You bring the objective and its derivatives; NLSolvers does not generate
gradients or Hessians for you with AD. In return the solvers are generic over
the number and array types you use: plain numbers, `Array`s, `StaticArray`s,
and CPU-resident array types in general, in mutating and non-mutating styles.

See the [documentation](https://julianlsolvers.github.io/NLSolvers.jl/stable)
for the full manual.

## Installation

```julia
using Pkg
Pkg.add("NLSolvers")
```

## Minimizing a function

An objective and its derivatives are collected in a `ScalarObjective`, wrapped
in an `OptimizationProblem` that records the code style (`inplace`) and any
bounds, and handed to `solve` together with a starting point, a method, and
`OptimizationOptions`:

```julia
using NLSolvers

f(x) = x^4 + sin(x)
g(∇f, x) = 4x^3 + cos(x)
fg(∇f, x) = f(x), g(∇f, x)

objective = ScalarObjective(f = f, g = g, fg = fg)
prob = OptimizationProblem(objective; inplace = false) # scalar input, so not inplace
solve(prob, 0.3, LineSearch(BFGS()), OptimizationOptions())
```

```
Results of minimization

* Algorithm:
  Inverse BFGS with Approximate Wolfe Line Search (Hager & Zhang)

* Candidate solution:
  Final objective value:    -4.35e-01
  Final gradient norm:      4.62e-09

  Initial objective value:  3.04e-01
  Initial gradient norm:    1.06e+00

* Stopping criteria
  |x - x'|              = 2.36e-06 <= 0.00e+00 (false)
  |x - x'|/|x|          = 3.99e-06 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 1.32e-11 <= -Inf (false)
  |f(x) - f(x')|/|f(x)| = 3.04e-11 <= -Inf (false)
  |g(x)|                = 4.62e-09 <= 1.00e-08 (true)
  |g(x)|/|g(x₀)|        = 4.35e-09 <= 0.00e+00 (false)

* Work counters
  Seconds run:   2.12e-04
  Iterations:    7
```

With `inplace = true` the same problem takes mutating derivative functions and
array iterates, and cache arrays are updated in place; `StaticArrays` work with
either style. Second-order methods take the Hessian through the `h` and `fgh`
fields of `ScalarObjective`.

## Solving nonlinear equations

Systems of equations use a `VectorObjective` in an `NEqProblem`:

```julia
function F!(Fx, x)
    Fx[1] = 1 - x[1]
    Fx[2] = 10(x[2] - x[1]^2)
    return Fx
end
function J!(Jx, x)
    Jx[1, 1] = -1
    Jx[1, 2] = 0
    Jx[2, 1] = -20x[1]
    Jx[2, 2] = 10
    return Jx
end
FJ!(Fx, Jx, x) = (F!(Fx, x), J!(Jx, x))

vectorobj = NLSolvers.VectorObjective(F!, J!, FJ!, nothing)
prob = NEqProblem(vectorobj)
solve(prob, [5.0, 0.0], TrustRegion(Newton(), Dogleg()), NEqOptions())
```

Fixed-point iterations use `FixedPointProblem`, and curve fitting uses
`LeastSquaresObjective`; see the
[docs](https://julianlsolvers.github.io/NLSolvers.jl/stable) for both.

## Methods

Line search: `BFGS`, `LBFGS`, `DBFGS`, `DFP`, `SR1`, `GradientDescent`,
`Newton`, and `ConjugateGradient` with the usual family of update formulas.

Trust region: `TrustRegion(scheme, subsolver)` combines a model scheme
(`Newton`, `BFGS`, `SR1`, ...) with a subproblem solver: `NWI` and `NTR`
(nearly exact), `Dogleg`, `TCG` (Steihaug-Toint truncated CG), or `TDTR`
(exact and factorization-free, for two-dimensional problems only). The
acceptance threshold, radius-update constants, and subsolver tolerances are
all exposed as options.

Bounds: `ActiveBox` (projected Newton) and `ParticleSwarm`.

Without derivatives: `NelderMead`, `SimulatedAnnealing`, `PureRandomSearch`,
and `BrentMin` for univariate minimization.

First-order and spectral: `Adam`, `AdaMax`, `BB`, `DFSANE`.

Nonlinear equations and acceleration: Newton with line search or trust region,
`InexactNewton` (Krylov), `DFSANE`, and `Anderson` acceleration for fixed
points.

## Notes

- Newton-type methods accept a `linsolve` argument, and several methods accept
  nonlinear left preconditioners; see the
  [docs](https://julianlsolvers.github.io/NLSolvers.jl/stable) for the
  interfaces.
- `NWI` on `BigFloat` (or other generic number types) needs
  `using GenericLinearAlgebra` for the eigendecomposition.
- Spectral and momentum methods such as `DFSANE` can be chaotic: tiny
  floating-point differences across CPUs and compiler versions can grow into
  different iterates, so cross-software comparisons need care. See
  [this paper](https://link.springer.com/article/10.1007/s10915-011-9521-3).
