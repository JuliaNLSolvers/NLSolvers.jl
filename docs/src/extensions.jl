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

no_derivatives = ScalarObjective(
    
    )

augged = ScalarObjective(ForwardDiffAutoDiff(), [3.0,3.0]; f=himmelblau!)
auGGed = zeros(2)
augged.g(auGGed, [0.1,0.2])

G = zeros(2)
objective.g(G, [0.1,0.2])
using Test
@test G  ≈ auGGed

auGGed = zeros(2)
fauGGed, _ = augged.fg(auGGed, [0.1,0.2])

G = zeros(2)
f, _ = objective.fg(G, [0.1,0.2])

@test G  ≈ auGGed
@test f  ≈ fauGGed


HauGGed = zeros(2,2)
augged.h(HauGGed, [0.1,0.2])

H = zeros(2,2)
objective.h(H, [0.1,0.2])

@test H  ≈ HauGGed




HauGGed = zeros(2,2)
auGGed = zeros(2)
fauGGed, _, _ = augged.fgh(auGGed, HauGGed, [0.1,0.2])

G = zeros(2)
H = zeros(2,2)
f, _, _= objective.fgh(G, H, [0.1,0.2])

@test f  ≈ fauGGed
@test G  ≈ auGGed
@test H  ≈ HauGGed

# test hv
x0 =[0.1,0.2]
hv = x0*0
augged.hv(hv, x0, 2*ones(2))
#=2-element Vector{Float64}:
 -66.0
 -34.0
=#

objective.h(H, x0)
@test H*(2.0*ones(2)) ≈ hv
```