using ForwardDiff
using NLSolvers

#ScalarLsqObjective
#VectorLsqObjective

@. model(x, p) = p[1]*exp(-x*p[2])
xdata = range(0, stop=10, length=20)
ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]

function f(x)
    mod = model(xdata, x)
    return sum(abs2, mod.-ydata)/2
end
x0 = copy(p0)
function g!(G, x)
    ForwardDiff.gradient!(G, f, x)
    return G
end
function h!(H, x)
    ForwardDiff.hessian!(H, f, x)
    return H
end
function fg(G, x)
    fx = f(x)
    g!(G, x)
    return fx, G
end
function fgh!(G, H, x)
    fx = f(x)
    g!(G, x)
    h!(H, x)
    return fx, G, H
end
obj = ScalarObjective(f, g!, fg, fgh!, h!, nothing, nothing, nothing)
prob = OptimizationProblem(obj, ([0.0,0.0],[3.0,3.0]))
res = solve(prob, copy(p0), LineSearch(BFGS()), OptimizationOptions())
res = solve(prob, copy(p0), NelderMead(), OptimizationOptions())
res = solve(prob, copy(p0), SimulatedAnnealing(), OptimizationOptions())
res = solve(prob, copy(p0), ParticleSwarm(), OptimizationOptions())
res = solve(prob, copy(p0), ActiveBox(), OptimizationOptions())
res = solve(prob, copy(p0), LineSearch(BFGS()), OptimizationOptions())
res = solve(prob, copy(p0), LineSearch(SR1()), OptimizationOptions())
res = solve(prob, copy(p0), LineSearch(DFP()), OptimizationOptions())
res = solve(prob, copy(p0), LineSearch(DBFGS()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(Newton(), Dogleg()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(Newton(), NTR()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(Newton(), NWI()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(SR1(), Dogleg()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(SR1(), NTR()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(SR1(), NWI()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(DFP(), Dogleg()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(DFP(), NTR()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(DFP(), NWI()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(DBFGS(), Dogleg()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(BFGS(), NTR()), OptimizationOptions())
res = solve(prob, copy(p0), TrustRegion(BFGS(), NWI()), OptimizationOptions())

function F!(Fx, x)
    Fx .= model(xdata, x)
    return Fx
end
function J!(Jx, x)
    # include this in the wrapper...
    ForwardDiff.jacobian!(Jx, F!, copy(ydata), x)
    return Jx
end
function FJ!(Fx, Jx, x)
    F!(Fx, x)
    J!(Jx, x)
    Fx, Jx
end

x0 = [-1.2, 1.0]

# Use y-data to setup Fx (and Jx) correctly
vectorobj = NLSolvers.LeastSquares(copy(x0), copy(ydata), ydata*x0', F!, FJ!, ydata)

prob=LeastSquaresProblem(vectorobj, ([0.0,0.0],[1.0,1.0]))
res = solve(prob, copy(p0), LineSearch(BFGS()), LeastSquaresOptions(40))
res = solve(prob, copy(p0), LineSearch(SR1()), LeastSquaresOptions(40))
res = solve(prob, copy(p0), LineSearch(DFP()), LeastSquaresOptions(40))
res = solve(prob, copy(p0), LineSearch(DBFGS()), LeastSquaresOptions(40))
res = solve(prob, copy(p0), TrustRegion(BFGS()), LeastSquaresOptions(40))
res = solve(prob, copy(p0), TrustRegion(BFGS()), LeastSquaresOptions(40))
res = solve(prob, copy(p0), TrustRegion(BFGS()), LeastSquaresOptions(40))
res = solve(prob, copy(p0), Adam(), LeastSquaresOptions(40))

function LeastSquaresModel(model, xdata, ydata)
    function f(x)
        function squared_error(xy)
            abs2(model(xy[1], x) - xy[2])
        end
        mapreduce(squared_error, +, zip(eachrow(xdata), ydata))
    end
    function g!(G, x)
        ForwardDiff.gradient!(G, f, x)
        return G
    end
    function h!(H, x)
        ForwardDiff.hessian!(H, f, x)
        return H
    end
    function fg(G, x)
        fx = f(x)
        g!(G, x)
        return fx, G
    end
    function fgh!(G, H, x)
        fx = f(x)
        g!(G, x)
        h!(H, x)
        return fx, G, H
    end
    obj = ScalarObjective(f, g!, fg, fgh!, h!, nothing, nothing, nothing)        
end

unimodel(x, p) = p[1]*exp(-x[1]*p[2])
xdata = range(0, stop=10, length=20)
ydata = unimodel.(xdata, Ref([1.0 2.0])) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]
obj = LeastSquaresModel(unimodel, xdata, ydata)
prob = OptimizationProblem(obj, ([0.0,0.0],[3.0,3.0]))
res = solve(prob, copy(p0), LineSearch(BFGS()), OptimizationOptions())
res = solve(prob, copy(p0), NelderMead(), OptimizationOptions())
res = solve(prob, copy(p0), SimulatedAnnealing(), OptimizationOptions())
res = solve(prob, copy(p0), ParticleSwarm(), OptimizationOptions())

# can do this based on model or model and derivative of model

function f(x)
    mod = model(xdata, x)
    return sum(abs2, mod.-ydata)/2
end
x0 = copy(p0)
function g!(G, x)
    ForwardDiff.gradient!(G, f, x)
    return G
end
function h!(H, x)
    ForwardDiff.hessian!(H, f, x)
    return H
end
function fg(G, x)
    fx = f(x)
    g!(G, x)
    return fx, G
end
function fgh!(G, H, x)
    fx = f(x)
    g!(G, x)
    h!(H, x)
    return fx, G, H
end
obj = ScalarObjective(f, g!, fg, fgh!, h!, nothing, nothing, nothing)
