using ForwardDiff

# create the full scalarobjective

#=
struct ScalarObjective{Tf, Tg, Tfg, Tfgh, Th, Thv, Tbf, P}
    f::Tf
    g::Tg
    fg::Tfg
    fgh::Tfgh
    h::Th
    hv::Thv
    batched_f::Tbf
    param::P
end
=#
abstract type AutoAD end
struct ForwadDiffAD <: AutoAD end
import NLSolvers: ScalarObjective
function ScalarObjective(
    autodiff::ForwadDiffAD,
    f,
    x,
    g = copy(x),
    h = x * x';
    param = nothing,
)
    chunksize = 4
    # g
    gradcfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{chunksize}())
    grad = (res, z) -> ForwardDiff.gradient!(res, f, z, gradcfg, Val{false}())

    # fg
    function fg(G, x)
        G = grad(G, x)
        f(x), G
    end

    # h
    hesscfg = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{chunksize}())
    hess = (res, z) -> ForwardDiff.hessian!(res, f, z, hesscfg, Val{false}())

    # fgh
    function fgh(G, H, x)
        G = grad(G, x)
        H = hess(H, x)
        f(x), G, H
    end

    # hv

    return ScalarObjective(
        f,
        grad,
        fg,
        fgh,
        hess,
        nothing, #hv,
        nothing,
        param,
    )
end


OPT_PROBS["himmelblau"]["array"]["mutating2"] =
    ScalarObjective(ForwadDiffAD(), himmelblau!, x0)
f2 = OPT_PROBS["himmelblau"]["array"]["mutating2"]
prob2 = OptimizationProblem(f)

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, NelderMead(), OptimizationOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob2, x0, NelderMead(), OptimizationOptions())

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, BFGS(), OptimizationOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob2, x0, BFGS(), OptimizationOptions())

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, Newton(), OptimizationOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob2, x0, Newton(), OptimizationOptions())


x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(Newton()), OptimizationOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob2, x0, TrustRegion(Newton()), OptimizationOptions())


x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(Newton()), OptimizationOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob2, x0, LineSearch(Newton()), OptimizationOptions())
