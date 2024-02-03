# Solving non-linear systems of equations
Non-linear systems of equations arise in many different applications. As for optimization, different situations will call for different inputs and optimizations: scalar, static, mutating, iterative solvers, etc. Below, we will cover some important algorithms and special cases.

## Scalar solving with 1st order derivatives
Assume that you have a residual system that you want to solve. This means that if your original system to solve is is `F(x) = K` for some real number `K`, you have made sure to provide a residual function `G(x) = F(x) - K` that we can solve for `G(x) = 0` instead. Then, you can use Newton's method for root finding as follows:


```
function F(x)
    x^2
end
    
function FJ(Jx, x)
    x^2, 2x
end

prob_obj = NLSolvers.ScalarObjective(
    f=F,
    fg=FJ,
)
    
prob = NEqProblem(prob_obj; inplace = false)

x0 = 0.3
res = solve(prob, x0, LineSearch(Newton(), Backtracking()))
```

with output

```
julia> res = solve(prob, x0, LineSearch(Newton(), Backtracking()))
Results of solving non-linear equations

* Algorithm:
  Newton's method with default linsolve with backtracking (no interp)

* Candidate solution:
  Final residual 2-norm:      5.36e-09
  Final residual Inf-norm:    5.36e-09

  Initial residual 2-norm:    9.00e-02
  Initial residual Inf-norm:  9.00e-02

* Stopping criteria
  |F(x')|               = 5.36e-09 <= 1.00e-08 (true)

* Work counters
  Seconds run:   1.41e-05
  Iterations:    12
```
The output reports initial and final residual norms. The convergence check is also reported, and some work counters report the time spent and the number of iterations. 

## Multivariate non-linear equation solving
Multivariate non-linear equation solving requires writing a `VectorObjective` instead of a `ScalarObjective` as above. The `VectorObjective` is 

```
using NLSolvers, ForwardDiff
function theta(x)
    if x[1] > 0
        return atan(x[2] / x[1]) / (2.0 * pi)
    else
        return (pi + atan(x[2] / x[1])) / (2.0 * pi)
    end
end

function F_powell!(Fx, x)
    if (x[1]^2 + x[2]^2 == 0)
        dtdx1 = 0
        dtdx2 = 0
    else
        dtdx1 = -x[2] / (2 * pi * (x[1]^2 + x[2]^2))
        dtdx2 = x[1] / (2 * pi * (x[1]^2 + x[2]^2))
    end
    Fx[1] =
        -2000.0 * (x[3] - 10.0 * theta(x)) * dtdx1 +
        200.0 * (sqrt(x[1]^2 + x[2]^2) - 1) * x[1] / sqrt(x[1]^2 + x[2]^2)
    Fx[2] =
        -2000.0 * (x[3] - 10.0 * theta(x)) * dtdx2 +
        200.0 * (sqrt(x[1]^2 + x[2]^2) - 1) * x[2] / sqrt(x[1]^2 + x[2]^2)
    Fx[3] = 200.0 * (x[3] - 10.0 * theta(x)) + 2.0 * x[3]
    Fx
end

function F_jacobian_powell!(Fx, Jx, x)
    ForwardDiff.jacobian!(Jx, F_powell!, Fx, x)
    Fx, Jx
end

prob_obj = VectorObjective(F=F_powell!, FJ=F_jacobian_powell!)
prob = NEqProblem(prob_obj)

x0 = [-1.0, 0.0, 0.0]
res = solve(prob, copy(x0), LineSearch(Newton(), Backtracking()))
```
with result
```
julia> res = solve(prob, copy(x0), LineSearch(Newton(), Backtracking()))
Results of solving non-linear equations

* Algorithm:
  Newton's method with default linsolve with backtracking (no interp)

* Candidate solution:
  Final residual 2-norm:      8.57e-16
  Final residual Inf-norm:    6.24e-16

  Initial residual 2-norm:    1.88e+03
  Initial residual Inf-norm:  1.59e+03

* Stopping criteria
  |F(x')|               = 6.24e-16 <= 1.00e-08 (true)

* Work counters
  Seconds run:   2.10e-04
  Iterations:    33
```
We again see initial and final residual norms. This time the 2- and Inf-norm are different because there is more than one element in the state to be optimized over. The final 2-norm also satisfies the required threshold, and the run-time and number of iterations are reported.