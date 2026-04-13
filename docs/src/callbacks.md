# Callbacks

NLSolvers.jl supports user-defined callbacks that are invoked after every iteration of any optimization solver. Callbacks let you monitor progress, log intermediate state, build a convergence trace, or stop the solver early based on custom criteria.

## Basic usage

A callback is any function (or callable object) that takes a single `info` argument and returns a `Bool`:

```julia
using NLSolvers

callback = info -> begin
    println("iter=$(info.iter), f=$(info.state.fz)")
    return false   # continue optimization
end

solve(prob, x0, LineSearch(BFGS()),
      OptimizationOptions(callback = callback))
```

Returning `true` stops the solver early; returning `false` lets it continue. The default `callback = nothing` disables the mechanism with zero overhead.

## What the callback receives

The `info` argument is a `NamedTuple` with three fields:

| Field | Type | Description |
|---|---|---|
| `iter` | `Int` | Current iteration count (1-based) |
| `time` | `Float64` | Elapsed seconds since `solve` started |
| `state` | `NamedTuple` | Solver-specific state (see below) |

## Solver-specific `state`

The contents of `info.state` depend on which solver is running. Use `haskey` if you write generic callbacks.

### Line search solvers (BFGS, DBFGS, DFP, SR1, L-BFGS, CG, Gradient Descent, Newton)

`state` is the internal `objvars` named tuple:

| Field | Description |
|---|---|
| `x` | Previous iterate |
| `fx` | Objective value at `x` |
| `∇fx` | Gradient at `x` |
| `z` | New iterate (after line search) |
| `fz` | Objective value at `z` |
| `∇fz` | Gradient at `z` |
| `B` | Current Hessian (or inverse) approximation; `nothing` for L-BFGS/CG |
| `Pg` | Preconditioned gradient if using a preconditioner; otherwise `nothing` |

### Trust region solvers

Same fields as line search, plus:

| Field | Description |
|---|---|
| `Δ` | Current trust region radius |
| `rejected` | `true` if the previous step was rejected by the trust region rule |

### Nelder-Mead

| Field | Description |
|---|---|
| `simplex_vector` | Vector of simplex vertices |
| `simplex_value` | Function values at each vertex |
| `x_centroid` | Centroid of the simplex (excluding the worst vertex) |
| `nm_obj` | Convergence metric — standard deviation of `simplex_value` |

### Simulated Annealing

| Field | Description |
|---|---|
| `x_best` | Best point found so far |
| `f_best` | Best objective value found so far |
| `x_now` | Current state of the chain |
| `f_now` | Objective value at `x_now` |
| `temperature` | Current temperature |

### Particle Swarm

| Field | Description |
|---|---|
| `X` | Current particle positions |
| `X_best` | Each particle's personal best |
| `Fs` | Function values at `X` |
| `Fs_best` | Function values at `X_best` |
| `x` | Global best particle |
| `best_f` | Global best objective value |
| `swarm_f` | Convergence metric for the swarm |

### Brent's method (univariate)

| Field | Description |
|---|---|
| `x` | Current best point |
| `fx` | Function value at `x` |
| `a`, `b` | Current bracketing interval `[a, b]` |
| `v`, `w` | Two previous iterates |
| `fv`, `fw` | Function values at `v` and `w` |

### Active Box (projected Newton)

| Field | Description |
|---|---|
| `x` | Previous iterate |
| `z` | New iterate |
| `fz` | Objective value at `z` |
| `∇fz` | Gradient at `z` |
| `B` | Hessian approximation |
| `activeset` | Boolean vector indicating active bound constraints |

## Examples

### Build a convergence trace

```julia
trace = Float64[]
solve(prob, x0, LineSearch(BFGS()),
      OptimizationOptions(
          callback = info -> (push!(trace, info.state.fz); false),
          maxiter = 100,
      ))
```

### Stop when the gradient is sufficiently small

```julia
gtol = 1e-6
solve(prob, x0, LineSearch(BFGS()),
      OptimizationOptions(
          callback = info -> norm(info.state.∇fz, Inf) < gtol,
          g_abstol = 0.0,  # disable built-in g-tolerance so callback wins
      ))
```

### Time-limited optimization

```julia
time_limit = 5.0  # seconds
solve(prob, x0, LineSearch(BFGS()),
      OptimizationOptions(callback = info -> info.time > time_limit))
```

### Save iterates for plotting later

Arrays in `info.state` (such as `z`, `∇fz`, `simplex_vector`) are aliases of live solver buffers — they will be overwritten on the next iteration. **Copy them if you need to retain them past the callback call.**

```julia
history = Vector{Vector{Float64}}()
solve(prob, x0, LineSearch(BFGS()),
      OptimizationOptions(
          callback = info -> (push!(history, copy(info.state.z)); false),
      ))
```

Scalar fields like `info.iter`, `info.time`, `info.state.fz` are values and do not need copying.

## Performance

The callback machinery has zero runtime overhead when `callback === nothing` (the default). The callback type is captured as a type parameter on `OptimizationOptions`, so the compiler eliminates the dispatch entirely. Passing a concrete callback function adds only the cost of calling that function once per iteration.
