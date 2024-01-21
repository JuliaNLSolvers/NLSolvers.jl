# Non-linear Least Squares

Non-linear Least Squares problems arise in many statistical problems across applications in economics, physics, pharmacometrics, and more. This package does not provide any statistical analysis tools, but it does provide the numerical procedures necessary to obtain the estimates.

## Scalar

#ScalarLsqObjective
#VectorLsqObjective

@. model(x, p) = p[1] * exp(-x * p[2])
xdata = range(0, stop = 10, length = 20)
ydata = model(xdata, [1.0 2.0]) + 0.01 * randn(length(xdata))

function f_lsq(x)
    mod = model(xdata, x)
    return sum(abs2, mod .- ydata) / 2
end
x0 = copy(p0)
function g!(G, x)
    ForwardDiff.gradient!(G, f_lsq, x)
    return G
end
function h!(H, x)
    ForwardDiff.hessian!(H, f_lsq, x)
    return H
end
function fg(G, x)
    fx = f_lsq(x)
    g!(G, x)
    return fx, G
end
function fgh!(G, H, x)
    fx = f_lsq(x)
    g!(G, x)
    h!(H, x)
    return fx, G, H
end
obj = ScalarObjective(f_lsq, g!, fg, fgh!, h!, nothing, nothing, nothing)
prob = OptimizationProblem(obj, ([1.5, 0.0], [3.0, 3.0]))

p0 = [1.5, 0.5]

res = [solve(prob, copy(p0), LineSearch(BFGS()), OptimizationOptions()),
solve(prob, copy(p0), NelderMead(), OptimizationOptions()),
solve(prob, copy(p0), SimulatedAnnealing(), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(BFGS()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(SR1()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(DFP()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(DBFGS()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(Newton(), Dogleg()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(Newton(), NTR()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(Newton(), NWI()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(SR1(), Dogleg()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(SR1(), NTR()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(SR1(), NWI()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(DFP(inverse=false), Dogleg()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(DFP(inverse=false), NTR()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(DFP(inverse=false), NWI()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(DBFGS(inverse=false), Dogleg()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(BFGS(inverse=false), NTR()), OptimizationOptions()),
solve(prob, copy(p0), TrustRegion(BFGS(inverse=false), NWI()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(ConjugateGradient(update=HS()), HZAW()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(ConjugateGradient(update=CD()), HZAW()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(ConjugateGradient(update=HZ()), HZAW()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(ConjugateGradient(update=FR()), HZAW()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(ConjugateGradient(update=PRP()), HZAW()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(ConjugateGradient(update=VPRP()), HZAW()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(ConjugateGradient(update=LS()), HZAW()), OptimizationOptions()),
solve(prob, copy(p0), LineSearch(ConjugateGradient(update=DY()), HZAW()), OptimizationOptions())]


res_elapsed = [(@elapsed solve(prob, copy(p0), LineSearch(BFGS()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), NelderMead(), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), SimulatedAnnealing(), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(BFGS()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(SR1()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(DFP()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(DBFGS()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(Newton(), Dogleg()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(Newton(), NTR()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(Newton(), NWI()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(SR1(), Dogleg()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(SR1(), NTR()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(SR1(), NWI()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(DFP(inverse=false), Dogleg()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(DFP(inverse=false), NTR()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(DFP(inverse=false), NWI()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(DBFGS(inverse=false), Dogleg()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(BFGS(inverse=false), NTR()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), TrustRegion(BFGS(inverse=false), NWI()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(ConjugateGradient(update=HS()), HZAW()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(ConjugateGradient(update=CD()), HZAW()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(ConjugateGradient(update=HZ()), HZAW()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(ConjugateGradient(update=FR()), HZAW()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(ConjugateGradient(update=PRP()), HZAW()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(ConjugateGradient(update=VPRP()), HZAW()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(ConjugateGradient(update=LS()), HZAW()), OptimizationOptions())),
(@elapsed solve(prob, copy(p0), LineSearch(ConjugateGradient(update=DY()), HZAW()), OptimizationOptions()))]
res_allocated = [(@allocated solve(prob, copy(p0), LineSearch(BFGS()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), NelderMead(), OptimizationOptions())),
(@allocated solve(prob, copy(p0), SimulatedAnnealing(), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(BFGS()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(SR1()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(DFP()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(DBFGS()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(Newton(), Dogleg()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(Newton(), NTR()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(Newton(), NWI()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(SR1(), Dogleg()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(SR1(), NTR()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(SR1(), NWI()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(DFP(), Dogleg()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(DFP(), NTR()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(DFP(), NWI()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(DBFGS(), Dogleg()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(BFGS(), NTR()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), TrustRegion(BFGS(), NWI()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(ConjugateGradient(update=HS()), HZAW()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(ConjugateGradient(update=CD()), HZAW()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(ConjugateGradient(update=HZ()), HZAW()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(ConjugateGradient(update=FR()), HZAW()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(ConjugateGradient(update=PRP()), HZAW()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(ConjugateGradient(update=VPRP()), HZAW()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(ConjugateGradient(update=LS()), HZAW()), OptimizationOptions())),
(@allocated solve(prob, copy(p0), LineSearch(ConjugateGradient(update=DY()), HZAW()), OptimizationOptions()))]


res_bounds = [solve(prob, copy(p0), ParticleSwarm(), OptimizationOptions()),
solve(prob, copy(p0), ActiveBox(), OptimizationOptions()),]

res_minimum = map(x->x.info.minimum, res)
res_bounds_minimum = map(x->x.info.minimum, res_bounds)