# NLSolvers
| **Source**  | **Build Status** |
|:-:|:-:|
| [![Source](https://img.shields.io/badge/GitHub-source-green.svg)](https://github.com/pkofod/NLSolvers.jl) |  [![Build Status](https://travis-ci.org/pkofod/NLSolvers.jl.svg?branch=master)](https://travis-ci.org/pkofod/NLSolvers.jl) | [![](https://badges.gitter.im/pkofod/NLSolvers.jl.svg)](https://gitter.im/pkofod/NLSolvers.jl) |
| [![Codecov branch](https://img.shields.io/codecov/c/github/pkofod/NLSolvers.jl/master.svg)](https://codecov.io/gh/pkofod/NLSolvers.jl) |[![Build Status](https://ci.appveyor.com/api/projects/status/prp8ygfp4rr9tafe?svg=true)](https://ci.appveyor.com/project/blegat/optim-jl) 

NLSolvers provides optimization, curve fitting, and equation solving functionalities for Julia.
The goal is to provide a set of robust and flexible methods that run fast. Currently, the package
does not try to implement any automatic generation of unspecified functions (gradients, Hessians,
Hessian-vector products) using AD.

NLSolvers.jl uses different problem types for different problems

- `OptimizationProblem` for optimization problems
- `NEqProblem` for non-linear equations problems
- `FixedPointProblem` for non-linear equations problems

## Optimization Problems
Take the following scalar objective (with scalar input)
```julia
using NLSolvers
function objective(x)
    fx = x^4 + sin(x)
end
function gradient(∇f, x)
    ∇f = 4x^3 + cos(x)
    return ∇f
end
objective_gradient(∇f, x) = objective(x), gradient(∇f, x)
function hessian(∇²f, x)
    ∇²f = 12x^2 - sin(x)
    return ∇²f
end
function objective_gradient_hessian(∇f, ∇²f, x)
    f, ∇f = objective_gradient(∇f, x)
    ∇²f = hessian(∇²f, x)
    return f, ∇f, ∇²f
end
scalarobj = ScalarObjective(f=objective,
                            g=gradient,
                            fg=objective_gradient,
                            fgh=objective_gradient_hessian,
                            h=hessian)
optprob = OptimizationProblem(scalarobj; inplace=false) # scalar input, so not inplace

solve(optprob, 0.3, LineSearch(Newton()), OptimizationOptions())
```
With output
```
Results of minimization

* Algorithm:
  Newton's method with default linsolve with backtracking (no interp)

* Candidate solution:
  Final objective value:    -4.35e-01
  Final gradient norm:      3.07e-12

  Initial objective value:  3.04e-01
  Initial gradient norm:    1.06e+00

* Convergence measures
  |x - x'|              = 6.39e-07 <= 0.00e+00 (false)
  |x - x'|/|x|          = 1.08e-06 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 9.71e-13 <= 0.00e+00 (false)
  |f(x) - f(x')|/|f(x)| = 2.23e-12 <= 0.00e+00 (false)
  |g(x)|                = 3.07e-12 <= 1.00e-08 (true)
  |g(x)|/|g(x₀)|        = 2.88e-12 <= 0.00e+00 (false)

* Work counters
  Seconds run:   7.15e-06
  Iterations:    6
```
The problem types are especially useful when manifolds, bounds, and other constraints enter the picture.

Let's take the same problem as above but write it with arrays and mutating code style. The inplace keyword argument to the `OptimizationProblem` is used to apply
the desired code paths internally. If set to true, cache arrays will be updated inplace and mutation is promised to be allowed for the input type(s). If set to
false, no operations will mutate or happen in place.
```
using NLSolvers
function objective_ip(x)
    fx = x[1]^4 + sin(x[1])
end
function gradient_ip(∇f, x)
    ∇f[1] = 4x[1]^3 + cos(x[1])
    return ∇f
end
objective_gradient_ip(∇f, x) = objective_ip(x), gradient_ip(∇f, x)
function hessian_ip(∇²f, x)
    ∇²f[1] = 12x[1]^2 - sin(x[1])
    return ∇²f
end
function objective_gradient_hessian_ip(∇f, ∇²f, x)
    f, ∇f = objective_gradient_ip(∇f, x)
    ∇²f = hessian_ip(∇²f, x)
    return f, ∇f, ∇²f
end
scalarobj_ip = ScalarObjective(f=objective_ip,
                            g=gradient_ip,
                            fg=objective_gradient_ip,
                            fgh=objective_gradient_hessian_ip,
                            h=hessian_ip)
optprob_ip = OptimizationProblem(scalarobj_ip; inplace=true)

solve(optprob_ip, [0.3], LineSearch(Newton()), OptimizationOptions())
```
which gives
```
Results of minimization

* Algorithm:
  Newton's method with default linsolve with backtracking (no interp)

* Candidate solution:
  Final objective value:    -4.35e-01
  Final gradient norm:      3.07e-12

  Initial objective value:  3.04e-01
  Initial gradient norm:    1.06e+00

* Convergence measures
  |x - x'|              = 6.39e-07 <= 0.00e+00 (false)
  |x - x'|/|x|          = 1.08e-06 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 9.71e-13 <= 0.00e+00 (false)
  |f(x) - f(x')|/|f(x)| = 2.23e-12 <= 0.00e+00 (false)
  |g(x)|                = 3.07e-12 <= 1.00e-08 (true)
  |g(x)|/|g(x₀)|        = 2.88e-12 <= 0.00e+00 (false)

* Work counters
  Seconds run:   1.10e-05
  Iterations:    6

```
as above. Another set of examples could be `SArray`'s and `MArray`'s from the `StaticArrays.jl` package. 

```
using StaticArrays
solve(optprob_ip, @MVector([0.3]), LineSearch(Newton()), OptimizationOptions())
```
which gives
```
Results of minimization

* Algorithm:
  Newton's method with default linsolve with backtracking (no interp)

* Candidate solution:
  Final objective value:    -4.35e-01
  Final gradient norm:      3.07e-12

  Initial objective value:  3.04e-01
  Initial gradient norm:    1.06e+00

* Convergence measures
  |x - x'|              = 6.39e-07 <= 0.00e+00 (false)
  |x - x'|/|x|          = 1.08e-06 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 9.71e-13 <= 0.00e+00 (false)
  |f(x) - f(x')|/|f(x)| = 2.23e-12 <= 0.00e+00 (false)
  |g(x)|                = 3.07e-12 <= 1.00e-08 (true)
  |g(x)|/|g(x₀)|        = 2.88e-12 <= 0.00e+00 (false)

* Work counters
  Seconds run:   5.68e-04
  Iterations:    6
```

So numbers, mutating array code and non-mutating array code is supported depending on the input to the problem type and initial `x` or state in general.

## BigFloat `eigen`
To use the `NWI` algorithm in a `TrustRegion` algorithm, it is necessary to first `using GenericLinearAlgebra`.

## Systems of Nonlinear Equations `NEqProblem`

To solve a system of non-linear equations you should use the `NEqProblem` type. First,
we have to define a `VectorObjective`. We can try to solve for the roots in the problem
defined by setting the gradient of the Rosenbrock test problem equal to zero.
```
function F_rosenbrock!(Fx, x)
    Fx[1] = 1 - x[1]
    Fx[2] = 10(x[2]-x[1]^2)
    return Fx
end
function J_rosenbrock!(Jx, x)
    Jx[1,1] = -1
    Jx[1,2] = 0
    Jx[2,1] = -20*x[1]
    Jx[2,2] = 10
    return Jx
end
function FJ_rosenbrock!(Fx, Jx, x)
    F_rosenbrock!(Fx, x)
    J_rosenbrock!(Jx, x)
    Fx, Jx
end
function Jvop_rosenbrock!(x)
    function JacV(Fv, v)
        Fv[1] = -1*v[1]
        Fv[2,] = -20*x[1]*v[1] + 10*v[2]
    end
    LinearMap(JacV, length(x))
end
vectorobj = NLSolvers.VectorObjective(F_rosenbrock!, J_rosenbrock!, FJ_rosenbrock!, Jvop_rosenbrock!)
```
and define a probem type that lets `solve` dispatch to the correct code
```
vectorprob = NEqProblem(vectorobj)
```
and we can solve using two variants of Newton's method. One that globalizes the
solve using a trust-region based method and one that uses a line search
```

julia> solve(vectorprob, [5.0, 0.0], TrustRegion(Newton(), Dogleg()), NEqOptions())
Results of solving non-linear equations

* Algorithm:
  Newton's method with default linsolve with Dogleg{Nothing}

* Candidate solution:
  Final residual 2-norm:      5.24e-14
  Final residual Inf-norm:    5.24e-14

  Initial residual 2-norm:    6.25e+04
  Initial residual Inf-norm:  2.50e+02

* Convergence measures
  |F(x')|               = 5.24e-14 <= 0.00e+00 (false)

* Work counters
  Seconds run:   1.91e-05
  Iterations:    2


julia> solve(vectorprob, [5.0, 0.0], LineSearch(Newton()), NEqOptions())
Results of solving non-linear equations

* Algorithm:
  Newton's method with default linsolve with backtracking (no interp)

* Candidate solution:
  Final residual 2-norm:      0.00e+00
  Final residual Inf-norm:    0.00e+00

  Initial residual 2-norm:    2.50e+02
  Initial residual Inf-norm:  2.50e+02

* Convergence measures
  |F(x')|               = 0.00e+00 <= 0.00e+00 (true)

* Work counters
  Seconds run:   1.00e-05
  Iterations:    2
```

## Univariate optimization

The only method that is exclusively univariate is Brent's method for function minimization `BrentMin`.

## Multivariate optimization
### Sampling based
- SimulatedAnnealing
- PureRandomSearch
- ParticleSwarm
### Direct search
- NelderMead
### Quasi-Newton Line search
- DBFGS
- BFGS
- SR1
- DFP
- GradientDescent
- LBFGS
### Conjugate Gradient Descent
- ConjugateGradient
### Newton Line Search
- Newton
### Gradient based (no line search)
- Adam
- AdaMax
### Acceleration methods
- Anderson
### Krylov
- InexactNewton
### BB style
- BB
- DFSANE

## Simple bounds (box)
- ParticleSwarm
- ActiveBox


## Custom solve
Newton methods generally accept a linsolve argument.

## Preconditioning
Several methods accept nonlinear (left-)preconditioners. A preconditioner is provided as a function that has two methods: `p(x)` and `p(x, P)` where the first prepares and returns the preconditioner and the second is the signature for updating the preconditioner. If the preconditioner is constant, both method
will simply return this preconditioner. A preconditioner is used in two contexts: in `ldiv!(pgr, factorize(P), gr)` that accepts a cache array for the preconditioned gradient `pgr`, the preconditioner `P`, and the gradient to be preconditioned `gr`, and in `mul!(x, P, y)`. For the out-of-place methods (`minimize` as opposed to `minimize!`) it is sufficient to have `\(P, gr)` and `*(P, y)` defined.

## Beware, chaotic gradient methods!
Some methods that might be labeled as acceleration, momentum, or spectral methods can exhibit chaotic behavior. Please keep this in mind if comparing things like `DFSANE` with similar implemenations in other software. It can give very different results given different compiler optimizations, CPU architectures, etc. See for example https://link.springer.com/article/10.1007/s10915-011-9521-3 .


## TODO
Documented in each type's docstring including LineSearch, BFGS, ....

Abstract arrays!!! :|
manifolds
Use user norms
Banded Jacobian
AD
nan hessian

line search should have a short curcuit for very small steps

# Next steps 
Mixed complementatiry
SAMIN, BOXES
Univariate!!
IP Newotn
Krylov Hessian
