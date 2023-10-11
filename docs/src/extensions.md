# Extensions to NLSolvers
Extensions are feature in Julia that allows package owners to provide functionality (through added methods to library functions) based on other packages. These packages can be useful for some users but cause increased load times for all users. Using extensions we've provided a few convenience functions that do not have to be loaded by all users, but only those who need it.

## ForwardDiff.jl
ForwardDiff.jl derivatives are available to users by using the `ScalarObjective(ForwardDiffAutoDiff(), x_seed; f=my_f)` constructor. Below is an example that shows that the hand-written and ForwardDiff derivatives agree.
```
using NLSolvers, ForwardDiff, Test
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
function himmelblau_fg!(∇f, x)
    himmelblau!(x), himmelblau_g!(∇f, x)
end
function himmelblau_h!(∇²f, x)
    ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0
    ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
    ∇²f[2, 1] = ∇²f[1, 2]
    ∇²f[2, 2] = 2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0
    return ∇²f
end
function himmelblau_fgh!(∇f, H, x)
    himmelblau!(x), himmelblau_g!(∇f, x), himmelblau_h!(H, x)
end

objective = ScalarObjective(
    f=himmelblau!,
    g=himmelblau_g!,
    fg=himmelblau_fg!,
	h=himmelblau_h!,
    fgh=himmelblau_fgh!,
    )

forward_objective = ScalarObjective(Experimental.ForwardDiffAutoDiff(), [3.0,3.0]; f=himmelblau!)

# Test gradient
G_forward = zeros(2)
forward_objective.g(G_forward, [0.1,0.2])
G = zeros(2)
objective.g(G, [0.1,0.2])
using Test
@test G  ≈ G_forward

# Test joint f and g evaluation
G_forward = zeros(2)
fG_forward, _ = forward_objective.fg(G_forward, [0.1,0.2])

G = zeros(2)
f, _ = objective.fg(G, [0.1,0.2])

@test G  ≈ G_forward
@test f  ≈ fG_forward

# Test Hessian evaluation
H_forward = zeros(2,2)
forward_objective.h(H_forward, [0.1,0.2])

H = zeros(2,2)
objective.h(H, [0.1,0.2])

@test H  ≈ H_forward

# Test joint f, G, H evaluation
H_forward = zeros(2,2)
G_forward = zeros(2)
f_forward, _, _ = forward_objective.fgh(G_forward, H_forward, [0.1,0.2])

G = zeros(2)
H = zeros(2,2)
f, _, _= objective.fgh(G, H, [0.1,0.2])

@test f  ≈ f_forward
@test G  ≈ G_forward
@test H  ≈ H_forward

# Test hessian-vector calculations
# test hv
x0 = [0.1,0.2]
hv = x0*0
forward_objective.hv(hv, x0, 2*ones(2))

objective.h(H, x0)
@test H*(2.0*ones(2)) ≈ hv
```

## AbstractDifferentiation
```
using NLSolvers, ForwardDiff, Test
import AbstractDifferentiation as AD
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
function himmelblau_fg!(∇f, x)
    himmelblau!(x), himmelblau_g!(∇f, x)
end
function himmelblau_h!(∇²f, x)
    ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0
    ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
    ∇²f[2, 1] = ∇²f[1, 2]
    ∇²f[2, 2] = 2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0
    return ∇²f
end
function himmelblau_fgh!(∇f, H, x)
    himmelblau!(x), himmelblau_g!(∇f, x), himmelblau_h!(H, x)
end

objective = ScalarObjective(
    f=himmelblau!,
    g=himmelblau_g!,
    fg=himmelblau_fg!,
	h=himmelblau_h!,
    fgh=himmelblau_fgh!,
    )

forward_objective = ScalarObjective(AD.ForwardDiffBackend(), [3.0,3.0]; f=himmelblau!)

# Test gradient
G_forward = zeros(2)
forward_objective.g(G_forward, [0.1,0.2])
G = zeros(2)
objective.g(G, [0.1,0.2])
using Test
@test G  ≈ G_forward

# Test joint f and g evaluation
G_forward = zeros(2)
fG_forward, _ = forward_objective.fg(G_forward, [0.1,0.2])

G = zeros(2)
f, _ = objective.fg(G, [0.1,0.2])

@test G  ≈ G_forward
@test f  ≈ fG_forward

# Test Hessian evaluation
H_forward = zeros(2,2)
forward_objective.h(H_forward, [0.1,0.2])

H = zeros(2,2)
objective.h(H, [0.1,0.2])

@test H  ≈ H_forward

# Test joint f, G, H evaluation
H_forward = zeros(2,2)
G_forward = zeros(2)
f_forward, _, _ = forward_objective.fgh(G_forward, H_forward, [0.1,0.2])

G = zeros(2)
H = zeros(2,2)
f, _, _= objective.fgh(G, H, [0.1,0.2])

@test f  ≈ f_forward
@test G  ≈ G_forward
@test H  ≈ H_forward

# Test hessian-vector calculations
# test hv
x0 = [0.1,0.2]
hv = x0*0
forward_objective.hv(hv, x0, 2*ones(2))

objective.h(H, x0)
@test H*(2.0*ones(2)) ≈ hv
```

## SparseDiffTools.jl
SparseDiffTools.jl allows you to exploit sparsity in finite difference and forward automatic differentiation. 

```
using NLSolvers, ForwardDiff, Test, SparseDiffTools
exponential!(x) = exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
sparseforward_objective = ScalarObjective(Experimental.SparseForwardDiff(), [3.0,3.0]; f=exponential!)
forward_objective = ScalarObjective(Experimental.ForwardDiffAutoDiff(), [3.0,3.0]; f=exponential!)
x=rand(2)
sparseforward_objective.h(rand(2,2), x)
forward_objective.h(rand(2,2), x)
wd = [0.0,0.0]
ws = [0.0,0.0]
x0 = [0.1,0.2]
hv = x0*0
v = 2*ones(2)
forward_objective.hv(wd, x, v)

sparseforward_objective.hv(ws, x, v)

```