using NLSolvers
NLE_PROBS = Dict()


# Rosenbrock
# Source: MINPACK 
NLE_PROBS["rosenbrock"] = Dict()
NLE_PROBS["rosenbrock"]["array"] = Dict()
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

NLE_PROBS["rosenbrock"]["array"]["x0"] = [-1.2, 1.0]
NLE_PROBS["rosenbrock"]["array"]["mutating"] = NLSolvers.VectorObjective(F_rosenbrock!, J_rosenbrock!, FJ_rosenbrock!, Jvop_rosenbrock!)


function F_powell_singular!(x::Vector, Fx::Vector, Jx::Union{Nothing, Matrix}=nothing)
    if !(Fx isa Nothing)
        Fx[1] = x[1] + 10x[2]
        Fx[2] = sqrt(5)*(x[3] - x[4])
        Fx[3] = (x[2] - 2x[3])^2
        Fx[4] = sqrt(10)*(x[1] - x[4])^2
    end
    if !(Jx isa Nothing)
        fill!(Jx, 0)
        Jx[1,1] = 1
        Jx[1,2] = 10
        Jx[2,3] = sqrt(5)
        Jx[2,4] = -Jx[2,3]
        Jx[3,2] = 2(x[2] - 2x[3])
        Jx[3,3] = -2Jx[3,2]
        Jx[4,1] = 2sqrt(10)*(x[1] - x[4])
        Jx[4,4] = -Jx[4,1]
    end
    objective_return(Fx, Jx)
end

function Jvop_powell_singular(x)
    function JacV(Fv, v)
        Fv[1] = v[1]+10*v[2]
        Fv[2] = (v[3]-v[4])*sqrt(5)
        xx23 = 2*x[2]-4*x[3]
        Fv[3] = v[2]*xx23-v[3]*xx23*2
        xx41 = 2*sqrt(10)*(x[1]-x[4])
        Fv[4] = v[1]*xx41 - v[4]*xx41
    end
    LinearMap(JacV, length(x))
end    

#=
NLE_PROBS["quantile"] = Dict()
NLE_PROBS["quantile"]["number"] = Dict()
@inline quantile_f(Fx, x) = log(max(x, 0.000001))
@inline quantile_j(Jx, x) = 1.0/x
@inline function quantile_fj(Fx, Jx, x)
    Fx = quantile_f(Fx, x)
    Jx = quantile_j(Jx, x)
    Fx, Jx
end
NLE_PROBS["quantile"]["number"]["x0"] = 0.5
NLE_PROBS["quantile"]["number"]["mutating"] = quantobj
function f()
    quantile_f(Fx, x) = log(max(x, 0.000001))
    quantile_j(Jx, x) = 1.0/x
    function quantile_fj(Fx, Jx, x)
       Fx = quantile_f(Fx, x)
       Jx = quantile_j(Jx, x)
       Fx, Jx
    end
    quantobj = NLSolvers.VectorObjective(quantile_f, quantile_j, quantile_fj, nothing)

    quantproblem = NEqProblem(quantobj, nothing, NLSolvers.Euclidean(0), NLSolvers.OutOfPlace())

    method = LineSearch(Newton())
    options = NEqOptions()
    state = NLSolvers.init(quantproblem, method, 3.0)
    @time res = solve(quantproblem, 0.4, method, options, state)
    @time res = solve(quantproblem, 0.4, method, options, state)
end

quantile_f(Fx, x) = log(max(x, 0.000001))
quantile_j(Jx, x) = 1.0/x
function quantile_fj(Fx, Jx, x)
   Fx = quantile_f(Fx, x)
   Jx = quantile_j(Jx, x)
   Fx, Jx
end
const quantobj = NLSolvers.VectorObjective(quantile_f, quantile_j, quantile_fj, nothing)

const quantproblem = NEqProblem(quantobj, nothing, NLSolvers.Euclidean(0), NLSolvers.OutOfPlace())

method = LineSearch(Newton())
options = NEqOptions()
state = NLSolvers.init(quantproblem, method, 3.0)
@time res = solve(quantproblem, 0.4, method, options, state)
@time res = solve(quantproblem, 0.4, method, options, state)
=#