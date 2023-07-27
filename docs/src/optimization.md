# Optimization
NLSolvers.jl implements several algorithms for non-linear optimization. The use-cases for non-linear optimization are quite diverse, and as such it is important for a Julia package for optimization to be flexible in the interface and the admissable types. Scalar optimization should not allocate a lot of buffers, static array types should not try to do in-place operations and allocate normal `Array`s, and so on. This section explains how to use the various algorithms in NLSolvers.jl and gives the user ideas about where it is possible to take advantage of parallel computing, special array types, and more.

## Univariate optimization
Brent's method for minimizing a scalar objective is implemented as the `BrentMin` method. To solve it, you need to provide an objective and bounds.
```
# Define objective function
brent_f(x) = sin(x)
# Define the objective wrapper
brent_scalar = ScalarObjective(; f = brent_f)
# Define the optimization problem using the objective wrapper and bounds as a tuple
brent_prob = OptimizationProblem(brent_scalar, (π/2, 2*π))
# Solve the problem using Brent's method for optimization
solve(brent_prob, BrentMin(), OptimizationOptions())
```

## Multivariate optimization
Many applications of optimization software deals with multivariate optimization and there are many methods in the literature and in software that deals with these types of questions. Below, we will show how to do multivariate optimization in NLSolvers.jl. Notice, that multivariate optimization is fundamentally different from multivalued optimization where the objective is a vector. Multivalued optimization is currently not supported in NLSolvers.jl

### Unconstrained optimization
If there are no constraints present, there are a lot of methods to choose from. They typically require an objective and an initial point. So-called gradient based methods, or first order methods, require gradients as well, and the second order methods require Hessian information as well. Some methods can even exploit Hessian-vector products.

To show how all the pieces fit together, we can try to find a local minimizer of the Himmelblau function.

```
function himmelblau!(x)
    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    return fx
end
function himmelblau_g!(∇f, x)
    ∇f[1] =
        4.0 * x[1]^3 + 4.0 * x[1] * x[2] - 44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    ∇f[2] =
        2.0 * x[1]^2 + 2.0 * x[2] - 22.0 + 4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    ∇f
end
function himmelblau_h!(∇²f, x)
    ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0
    ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
    ∇²f[2, 1] = ∇²f[1, 2]
    ∇²f[2, 2] = 2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0
    return ∇²f
end
```
The preceding functions define the objective, its gradient, and its Hessian. Notice, that the functions have to have the following input

- objective functions require the current state `x`. It must return the objective value.
- gradient functions require the gradient container `∇f` as the first argument and the state `x` as the second. It must return the gradient.
- Hessian functions require the Hessian container `∇²f` as the first argument and the state `x` as the second. It must return the Hessian.

Next, we wrap these up in the objective type

```
objective = ScalarObjective(
    f=himmelblau!,
    g=himmelblau_g!,
	h=himmelblau_h!,
    )
```

The `ScalarObjective` signifies that for any given input the objective returns a single number. This is contrary to non-linear systems of equations for example. We should also set an initial point to search from.

```
x0 = [3.0, 1.0]
```

Finally, we define the problem type.

```
prob = OptimizationProblem(objective)
```

Then, we solve it.

```
julia> results = solve(prob, x0, LineSearch(LBFGS()), OptimizationOptions())
Results of minimization

* Algorithm:
  Inverse LBFGS with backtracking (no interp)

* Candidate solution:
  Final objective value:    2.53e-25
  Final gradient norm:      3.87e-12

  Initial objective value:  1.00e+01
  Initial gradient norm:    1.80e+01

* Stopping criteria
  |x - x'|              = 1.94e-08 <= 0.00e+00 (false)
  |x - x'|/|x|          = 5.37e-09 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 1.29e-14 <= 0.00e+00 (false)
  |f(x) - f(x')|/|f(x)| = 1.00e+00 <= 0.00e+00 (false)
  |g(x)|                = 3.87e-12 <= 1.00e-08 (true)
  |g(x)|/|g(x₀)|        = 2.15e-13 <= 0.00e+00 (false)

* Work counters
  Seconds run:   1.79e-05
  Iterations:    11
```

Since we have Hessian information available, we can also use variants of Newton's method.

```
julia> x0 = [3.0, 1.0]
2-element Vector{Float64}:
 3.0
 1.0

julia> results = solve(prob, x0, TrustRegion(Newton()), OptimizationOptions())
Results of minimization

* Algorithm:
  Newton's method with default linsolve with Trust Region (Newton, cholesky)

* Candidate solution:
  Final objective value:    7.10e-30
  Final gradient norm:      2.84e-14

  Initial objective value:  1.00e+01
  Initial gradient norm:    1.80e+01

* Stopping criteria
  |x - x'|              = 2.73e-08 <= 0.00e+00 (false)
  |x - x'|/|x|          = 6.50e-09 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 3.01e-14 <= 0.00e+00 (false)
  |f(x) - f(x')|/|f(x)| = 1.00e+00 <= 0.00e+00 (false)
  |g(x)|                = 2.84e-14 <= 1.00e-08 (true)
  |g(x)|/|g(x₀)|        = 1.58e-15 <= 0.00e+00 (false)
  Δ                     = 7.63e+04 <= 0.00e+00 (false)

* Work counters
  Seconds run:   6.39e-05
  Iterations:    9
```

It is also possible to use methods that only use the objective to guide the search for an optimum. For example, it is possible to use the direct search method by Nelder and Mead.

```
julia> results = solve(prob, x0, NelderMead(), OptimizationOptions())
Results of minimization

* Algorithm:
  Nelder-Mead

* Candidate solution:
  Final objective value:    0.00e+00

  Initial objective value:  0.00e+00

* Stopping criteria
  √(Σ(yᵢ-ȳ)²)/n         = 9.58e-09 <= 1.00e-08 (true)

* Work counters
  Seconds run:   1.22e-02
  Iterations:    32
```

We can also use a method based on sampling candidate solutions. Adaptive Particle Swarm is one such method. This method requires bounds on the state variable, so let us define a new optimization problem.

```
prob_bounds = OptimizationProblem(objective, ([0.0,0.0], [3.0,4.0]))
```

Then, we can `solve` the problem using `ParticleSwarm`. We set the `maxiter` option, because the method has no real termination criteria, but will keep iterating until `maxiter` has been reached.

```
julia> results = solve(prob_bounds, x0, ParticleSwarm(), OptimizationOptions(maxiter=38))
Results of minimization

* Algorithm:
  Adaptive Particle Swarm

* Candidate solution:
  Final objective value:    8.46e-14

  Initial objective value:  1.00e+01

* Stopping criteria

* Work counters
  Seconds run:   2.49e-04
  Iterations:    38
```

## Box constrained problems
Box constraints, variable limits, and simple bounds are common names for variables where each individual parameter can have lower and upper bounds associated with them. We saw above how to enforce these bounds in the `ParticleSwarm` optimizer. There are other methods available. For example, `ActiveBounds` is a projected Newton's method. It was built for convex problem, so it can fail if the function is not locally convex

```
julia> solve(prob_bounds, [0.0, 1.8], ActiveBox(), OptimizationOptions())
ERROR: PosDefException: matrix is not positive definite; Cholesky factorization failed.
...
```

However, it can often work well if you specify a factorization with modifications if negative eigenvalues are detected, such as the one in PositiveFactorizations.jl.
```
julia> solve(prob_bounds, [0.0, 1.8], ActiveBox(factorize=NLSolvers.positive_factorize), OptimizationOptions())
Results of minimization

* Algorithm:
  ActiveBox

* Candidate solution:
  Final objective value:    3.16e-30
  Final gradient norm:      2.84e-14
  Final projected gradient norm:  7.11e-15

  Initial objective value:  9.88e+01
  Initial gradient norm:    4.55e+01

* Stopping criteria
  |x - x'|              = 1.02e-09 <= 0.00e+00 (false)
  |x - x'|/|x|          = 2.82e-10 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 1.42e-17 <= 0.00e+00 (false)
  |f(x) - f(x')|/|f(x)| = 1.00e+00 <= 0.00e+00 (false)
  |x - P(x - g(x))|     = 7.11e-15 <= 1.00e-08 (true)
  |g(x)|                = 2.84e-14 <= 1.00e-08 (true)
  |g(x)|/|g(x₀)|        = 6.25e-16 <= 0.00e+00 (false)

* Work counters
  Seconds run:   1.26e-04
  Iterations:    12

```

If the solution happens to end up at the boundary it will be printed as part of the show method for the results. The following example will lead to a solution at the boundary.
```
prob_bounds = OptimizationProblem(objective, ([0.0,0.0], [2.5,2.8]))
```
And has the following output.
```
julia> solve(prob_bounds, [0.0, 1.8], ActiveBox(factorize=NLSolvers.positive_factorize), OptimizationOptions())
Results of minimization

* Algorithm:
  ActiveBox

* Candidate solution:
  Final objective value:    6.57e+00
  Final gradient norm:      2.39e+01
  Final projected gradient norm:  9.74e-11

  Initial objective value:  9.88e+01
  Initial gradient norm:    4.55e+01

* Stopping criteria
  |x - x'|              = 1.90e-06 <= 0.00e+00 (false)
  |x - x'|/|x|          = 5.65e-07 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 8.07e-11 <= 0.00e+00 (false)
  |f(x) - f(x')|/|f(x)| = 1.23e-11 <= 0.00e+00 (false)
  |x - P(x - g(x))|     = 9.74e-11 <= 1.00e-08 (true)
  |g(x)|                = 2.39e+01 <= 1.00e-08 (false)
  |g(x)|/|g(x₀)|        = 5.26e-01 <= 0.00e+00 (false)

  !!! Solution is at the boundary !!!

* Work counters
  Seconds run:   8.49e-05
  Iterations:    9
```
Notice, that the final gradient norm is way above the default threshold, but the norm of the projected gradient is indeed small. There is also a message that the solution is at the boundary.