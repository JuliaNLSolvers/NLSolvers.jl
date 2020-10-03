using StaticArrays, NLSolvers, LinearAlgebra
OPT_PROBS = Dict()

#### Himmelblau

OPT_PROBS["himmelblau"] = Dict()
OPT_PROBS["himmelblau"]["array"] = Dict()

function himmelblau!(x)
    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    return fx
end
function himmelblau_batched_f!(X)
    F = map(himmelblau!, X)
    return F
end
function himmelblau_batched_f!(F, X)
    map!(himmelblau!, F, X)
    return F
end
function himmelblau_g!(∇f, x)
    ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
    44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
    4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    ∇f
end

function himmelblau_fg!(∇f, x)
    ∇f = himmelblau_g!(∇f, x)
    fx = himmelblau!(x)
    return fx, ∇f
end
function himmelblau_fgh!(∇f, ∇²f, x)
    ∇²f = himmelblau_h!(∇²f, x)
    fx, ∇f = himmelblau_fg!(∇f, x)
    return fx, ∇f, ∇²f
end
function himmelblau_h!(∇²f, x)
    ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0
    ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
    ∇²f[2, 1] = ∇²f[1, 2]
    ∇²f[2, 2] = 2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0
    return ∇²f
end
function himmelblau_hv!(hv, x, v)
    hv[1] = (12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0)*v[1] + (4.0 * x[1] + 4.0 * x[2])*v[2]
    hv[2] =  (4.0 * x[1] + 4.0 * x[2])*v[1] + (2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0)*v[2]
    return hv
end

OPT_PROBS["himmelblau"]["array"]["x0"] = [3.0, 1.0]
OPT_PROBS["himmelblau"]["array"]["mutating"] = ScalarObjective(himmelblau!, himmelblau_g!, himmelblau_fg!, himmelblau_fgh!, himmelblau_h!, himmelblau_hv!, himmelblau_batched_f!, nothing)

OPT_PROBS["himmelblau"]["staticarray"] = Dict()

function himmelblau_f(x)
    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    return fx
end

function himmelblau_fg(∇f, x)
    ∇f = himmelblau_g(∇f, x)
    fx = himmelblau_f(x)
    return fx, ∇f
end

function himmelblau_g(∇f, x)
    ∇f1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
    44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    ∇f2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
    4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    ∇f = @SVector([∇f1, ∇f2])
    return ∇f
end

function himmelblau_fgh(∇f, ∇²f, x)
    ∇²f11 = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
    ∇²f12 = 4.0 * x[1] + 4.0 * x[2]
    ∇²f21 = 4.0 * x[1] + 4.0 * x[2]
    ∇²f22 = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
    ∇²f = @SMatrix([∇²f11 ∇²f12; ∇²f21 ∇²f22])
    fx, ∇f = himmelblau_fg(∇f, x)

    return fx, ∇f, ∇²f
end


sl = @SVector([3.0,1.0])
state0newton = (@SVector([2.0,2.0]), I+sl*sl')
OPT_PROBS["himmelblau"]["staticarray"]["x0"] = sl
OPT_PROBS["himmelblau"]["staticarray"]["state0"] = state0newton
OPT_PROBS["himmelblau"]["staticarray"]["static"] = ScalarObjective(himmelblau_f, himmelblau_g, himmelblau_fg, himmelblau_fgh, nothing, nothing, nothing, nothing)

### Exponential

exponential!(x) = exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
function exponential_g!(g, x)
    g[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
    g[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
    return g
end
function exponential_h!(H, x)
    H[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
    H[1, 2] = 0.0
    H[2, 1] = 0.0
    H[2, 2] = 2.0 * exp((3.0 - x[2])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
    return H
end
function exponential_hv!(Hv, x, v)
    Hv[1, 1] = (2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9))*v[1]
    Hv[2, 2] = (2.0 * exp((3.0 - x[2])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19))*v[2]
    return Hv
end
function exponential_fg!(g, x)
    fx = exponential!(x)
    g = exponential_g!(g, x)
    return fx, g
end
function exponential_fgh!(g, H, x)
    fx, g = exponential_fg!(g, x)
    H = exponential_h!(H, x)
    return fx, g, H
end

OPT_PROBS["exponential"] = Dict()
OPT_PROBS["exponential"]["array"] = Dict()
# Byttet om på x og H
OPT_PROBS["exponential"]["array"]["x0"] = [0.0, 0.0]
OPT_PROBS["exponential"]["array"]["mutating"] = ScalarObjective(exponential!, exponential_g!, exponential_fg!, exponential_fgh!, exponential_h!, exponential_hv!, nothing, nothing)

OPT_PROBS["laplacian"] = Dict()
OPT_PROBS["laplacian"]["array"] = Dict()
# Byttet om på x og H
OPT_PROBS["laplacian"]["array"]["x0(n)"] = n->zeros(n)

plap(U; n=length(U)) = (n-1) * sum((0.1 .+ diff(U).^2).^2) - sum(U) / (n-1)
plap1(U; n=length(U), dU = diff(U), dW = 4 .* (0.1 .+ dU.^2) .* dU) =
(n - 1) .* ([0.0; dW] .- [dW; 0.0]) .- ones(n) / (n-1)
precond(x::Vector) = precond(length(x))
precond(n::Number) = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1)) * (n+1)
_f(x) = plap([0;x;0])
function _fg(∇f, x)
    fx = _f(x)
    copyto!(∇f, (plap1([0;x;0]))[2:end-1])
    fx, ∇f
end

OPT_PROBS["laplacian"]["array"]["mutating"] = OptimizationProblem(ScalarObjective(_f, nothing, _fg, nothing, nothing, nothing, nothing, nothing))









function theta(x)
    if x[1] > 0
        return atan(x[2] / x[1]) / (2.0 * pi)
    else
        return (pi + atan(x[2] / x[1])) / (2.0 * pi)
    end
end

function fletcher_powell_f(x)
    theta_x = theta(x)
    fx = 100.0 * ((x[3] - 10.0 * theta_x)^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2
    return fx
end
function fletcher_powell_g(∇f, x)
    T = eltype(x)
    theta_x = theta(x)

    if x[1]^2 + x[2]^2 == 0
        dtdx1 = T(0)
        dtdx2 = T(0)
    else
        dtdx1 = - x[2] / ( T(2) * pi * ( x[1]^2 + x[2]^2 ) )
        dtdx2 =   x[1] / ( T(2) * pi * ( x[1]^2 + x[2]^2 ) )
    end
    ∇f1 = -2000.0*(x[3]-10.0*theta_x)*dtdx1 +
        200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 )
    ∇f2 = -2000.0*(x[3]-10.0*theta_x)*dtdx2 +
        200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 )
    ∇f3 =  200.0*(x[3]-10.0*theta_x) + 2.0*x[3];
    ∇f = @SVector[∇f1, ∇f2, ∇f3]

    return ∇f
end
function fletcher_powell_fg(∇f, x)
    T = eltype(x)
    theta_x = theta(x)

    if x[1]^2 + x[2]^2 == 0
        dtdx1 = T(0)
        dtdx2 = T(0)
    else
        dtdx1 = - x[2] / ( T(2) * pi * ( x[1]^2 + x[2]^2 ) )
        dtdx2 =   x[1] / ( T(2) * pi * ( x[1]^2 + x[2]^2 ) )
    end
    ∇f1 = -2000.0*(x[3]-10.0*theta_x)*dtdx1 +
        200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 )
    ∇f2 = -2000.0*(x[3]-10.0*theta_x)*dtdx2 +
        200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 )
    ∇f3 =  200.0*(x[3]-10.0*theta_x) + 2.0*x[3];
    ∇f = @SVector[∇f1, ∇f2, ∇f3]

    fx = 100.0 * ((x[3] - 10.0 * theta_x)^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

    return fx, ∇f
end
const sv3 = @SVector[0.1,0.0,0.0]

const fp_static = ScalarObjective(fletcher_powell_f, fletcher_powell_g, fletcher_powell_fg, nothing, nothing, nothing, nothing, nothing)

const state0 = (@SVector[-0.5, 0.0, 0.0], I+sv3*sv3')
OPT_PROBS["fletcher_powell"] = Dict()
OPT_PROBS["fletcher_powell"]["staticarray"] = Dict()
# Byttet om på x og H
OPT_PROBS["fletcher_powell"]["staticarray"]["x0"] = state0
OPT_PROBS["fletcher_powell"]["staticarray"]["static"] = OptimizationProblem(fp_static; inplace=false)
