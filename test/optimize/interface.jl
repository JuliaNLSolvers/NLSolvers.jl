using NLSolvers
using LinearAlgebra
using SparseDiffTools
using SparseArrays
using IterativeSolvers
using ForwardDiff
using Test

@testset "optimization interface" begin
# TODO
# Make a more efficient MeritObjective that returns something that acts as the actual thing if requested (mostly for debug)
# but can also be efficiently used to get cauchy and newton
#
# Make DOGLEG work also with BFGS (why is convergence so slow?)
# # Look into what caches are created
# # Why does SR1 give inf above?
#
# Mutated first r last (easy to make fallback for nonmutating)
# NelderMead
# time limit not enforced in @show solve(NelderMead)
# no convergence crit either
#
# APSO
# no @show solve
# wrong return type (no ConvergenceInfo)
#
# PureRandom search. wrong return type and move bounds to problem
#
# Does not have a ! method, this should be documented. Maybe add it for consistency?
# If the sampler is empty and there are bounds, draw uniformly there in stead of specifying lb and ub in PureRandomSearch
#
# really need a QNmodel for model vars that creates nothing or don't populate fields of a named tuple for Newton for example
# LineObjective and  LineObjective! should just dispatch on the caceh being nothing or not
#
# ConjgtaeGraduent with HZAW fails because it overwrites Py into P∇fz which seems to alias P∇fz. That alias needs to be checked
# and a CGModelVars type should allocate Py where appropriate - could y be overwitten with Py and then recalcualte y afterwards?
#
#
# ADAM needs @show solve and AdaMax
# 
# TODO: LineObjetive doesn't need ! when we have problem in there and mstyle

#### OPTIMIZATION
f = OPT_PROBS["himmelblau"]["array"]["mutating"]
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
prob = OptimizationProblem(f)
prob_oop = OptimizationProblem(f; inplace=false)
prob_bounds = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]))
prob_bounds_oop = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]); inplace=false)
prob_on_bounds = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]))
prob_on_bounds_oop = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]); inplace=false)

res = solve(prob, x0, NelderMead(), OptimizationOptions())
@test all(x0 .== [3.0,2.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_oop, x0, NelderMead(), OptimizationOptions())
@test all(x0 .== [3.0,1.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_bounds, x0, APSO(), OptimizationOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"]).+1
res = solve(prob_on_bounds_oop, x0, ActiveBox(), OptimizationOptions())
@test_broken all(x0 .== [3.0,1.0])
xbounds = [ 3.5, 1.6165968467447174]
@test res.info.minimum == NLSolvers.value(prob_on_bounds, xbounds)

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_bounds, x0, SimulatedAnnealing(), OptimizationOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-1

solve(prob, PureRandomSearch(lb=[0.0,0.0], ub=[4.0,4.0]), OptimizationOptions())
solve(prob_oop, PureRandomSearch(lb=[0.0,0.0], ub=[4.0,4.0]), OptimizationOptions())

solve(prob, [0.0,0.0], SimulatedAnnealing(), OptimizationOptions())
solve(prob_oop, [0.0,0.0], SimulatedAnnealing(), OptimizationOptions())


#x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
#@show solve(prob_bounds, x0, SIMAN(), OptimizationOptions())
#@test all(x0 .== [3.0,1.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(), OptimizationOptions())
#@test all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(SR1()), OptimizationOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(DFP()), OptimizationOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(BFGS()), OptimizationOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(LBFGS()), OptimizationOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(LBFGS(), HZAW()), OptimizationOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(LBFGS(), Backtracking()), OptimizationOptions())
@test res.info.minimum < 1e-12

#@test all(x0 .== [3.0,2.0])
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(ConjugateGradient(), HZAW()), OptimizationOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_oop, x0, LineSearch(ConjugateGradient(), HZAW()), OptimizationOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), OptimizationOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_oop, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), OptimizationOptions())
@test res.info.minimum < 1e-12

# Stalls at [3, 1] with default @show solve
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(Newton()), OptimizationOptions())
@test res.info.minimum < 1e-12

#@test all(x0 .== [3.0,2.0])
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(Newton(), HZAW()), OptimizationOptions())
@test res.info.minimum < 1e-12
#@test all(x0 .== [3.0,2.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(), OptimizationOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(DBFGS(), Dogleg()), OptimizationOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(BFGS(), Dogleg()), OptimizationOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(SR1(), NTR()), OptimizationOptions())
@test res.info.minimum < 1e-16

# not PSD
#x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
#res = solve(prob, x0, TrustRegion(Newton(), Dogleg()), OptimizationOptions())
#@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(BFGS(), Dogleg()), OptimizationOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(DBFGS(), Dogleg()), OptimizationOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, Adam(), OptimizationOptions(maxiter=20000))
@test res.info.minimum < 1e-16

## Notice that prob is only used for value so this should be extremely generic! It does need a comparison though.
res = solve(prob, PureRandomSearch(lb=[0.0,0.0], ub=[4.0,4.0]), OptimizationOptions())

f = OPT_PROBS["exponential"]["array"]["mutating"]
x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
prob = OptimizationProblem(f)
prob_bounds = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]))
prob_on_bounds = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]))

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(DBFGS(), Dogleg()), OptimizationOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(BFGS(), Dogleg()), OptimizationOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(Newton(), Dogleg()), OptimizationOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(Newton(), NTR()), OptimizationOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(Newton(), NWI()), OptimizationOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(SR1(), NTR()), OptimizationOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(SR1(Inverse()), NTR()), OptimizationOptions())
@test res.info.minimum == 2.0
end

const statictest_s0 = OPT_PROBS["himmelblau"]["staticarray"]["state0"]
const statictest_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["staticarray"]["static"]; inplace=false)
@testset "staticopt" begin
    res = solve(statictest_prob, statictest_s0, LineSearch(Newton()), OptimizationOptions())
    @allocated solve(statictest_prob, statictest_s0, LineSearch(Newton()), OptimizationOptions())
    alloc = @allocated solve(statictest_prob, statictest_s0, LineSearch(Newton()), OptimizationOptions())
    @test alloc == 0

    _res = solve(statictest_prob, statictest_s0, LineSearch(Newton()), OptimizationOptions())
    _alloc = @allocated solve(statictest_prob, statictest_s0, LineSearch(Newton()), OptimizationOptions())
    @test _alloc == 0
    @test norm(_res.info.∇fz, Inf) < 1e-8

    _res = solve(statictest_prob, statictest_s0, LineSearch(Newton(), Backtracking()), OptimizationOptions())
    _alloc = @allocated solve(statictest_prob, statictest_s0, LineSearch(Newton(), Backtracking()), OptimizationOptions())
    @test _alloc == 0
    @test norm(_res.info.∇fz, Inf) < 1e-8
end

@testset "newton" begin
    test_x0 = [2.0, 2.0]
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=true)
    res = solve(test_prob, copy(test_x0), LineSearch(Newton()), OptimizationOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8

    test_x0 = [2.0, 2.0]
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=false)
    res = solve(test_prob, test_x0, LineSearch(Newton()), OptimizationOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8
end
@testset "Newton linsolve" begin
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=true)
    res_lu = solve(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(d, B, g)->ldiv!(d, lu(B), g))), OptimizationOptions())
    @test norm(res_lu.info.∇fz, Inf) < 1e-8
    res_qr = solve(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(d, B, g)->ldiv!(d, qr(B), g))), OptimizationOptions())
    @test norm(res_qr.info.∇fz, Inf) < 1e-8
  
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=false)
    res_qr = solve(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(B, g)->qr(B)\g)), OptimizationOptions())
    @test norm(res_qr.info.∇fz, Inf) < 1e-8
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=false)
    res_lu = solve(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(B, g)->lu(B)\g)), OptimizationOptions())
    @test norm(res_lu.info.∇fz, Inf) < 1e-8
end















const static_x0 = OPT_PROBS["fletcher_powell"]["staticarray"]["x0"][1]
const static_prob_qn = OPT_PROBS["fletcher_powell"]["staticarray"]["static"]
@testset "no alloc static" begin

    @testset "no alloc" begin
        @allocated solve(static_prob_qn, static_x0, LineSearch(BFGS(Inverse()), Backtracking()), OptimizationOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(BFGS(Inverse()), Backtracking()), OptimizationOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(BFGS(Direct()), Backtracking()), OptimizationOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(BFGS(Direct()), Backtracking()), OptimizationOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(DFP(Inverse()), Backtracking()), OptimizationOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(DFP(Inverse()), Backtracking()), OptimizationOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(DFP(Direct()), Backtracking()), OptimizationOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(DFP(Direct()), Backtracking()), OptimizationOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(SR1(Inverse()), Backtracking()), OptimizationOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(SR1(Inverse()), Backtracking()), OptimizationOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(SR1(Direct()), Backtracking()), OptimizationOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(SR1(Direct()), Backtracking()), OptimizationOptions())
        @test _alloc == 0
    end
end

Random.seed!(4568532)
solve(static_prob_qn, rand(3), Adam(), OptimizationOptions(maxiter=1000))
solve(static_prob_qn, rand(3), AdaMax(), OptimizationOptions(maxiter=1000))




@testset "bound newton" begin
    f = OPT_PROBS["himmelblau"]["array"]["mutating"]
    x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
    prob = OptimizationProblem(f)
    prob_oop = OptimizationProblem(f; inplace=false)
    prob_bounds = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]))
    prob_bounds_oop = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]); inplace=false)
    prob_on_bounds = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]))
    prob_on_bounds_oop = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]); inplace=false)

    start = [3.7,2.0]

    res_unc = solve(prob_bounds, copy(start), LineSearch(Newton(), Backtracking()), OptimizationOptions())
    @test res_unc.info.minimizer ≈ [3.0, 2.0]
    res_con = solve(prob_bounds, copy(start), ActiveBox(), OptimizationOptions())
    @test res_con.info.minimizer ≈ [3.0, 2.0]
    res_unc = solve(prob_on_bounds, copy(start), LineSearch(Newton(), Backtracking()), OptimizationOptions())
    @test res_unc.info.minimizer ≈ [3.0, 2.0]
    res_con = solve(prob_on_bounds, copy(start), ActiveBox(), OptimizationOptions())
    @test res_con.info.minimizer ≈ [3.5, 1.6165968467448326]
end

function fourth_f(x)
    fx = x^4 + sin(x)
    return fx
end
function fourth_fg(∇f, x)
    ∇f = 4x^3 + cos(x)

    fx = x^4 + sin(x)
    return fx, ∇f
end

function fourth_fgh(∇f, ∇²fx, x)
    ∇²f = 12x^2 - sin(x)
    ∇f = 4x^3 + cos(x)

    fx = x^4 + sin(x)
    return fx, ∇f, ∇²f
end

const scalar_prob_oop = OptimizationProblem(ScalarObjective(fourth_f, nothing, fourth_fg, fourth_fgh, nothing, nothing, nothing, nothing); inplace=false)
@testset "scalar no-alloc" begin
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(SR1(Direct())), OptimizationOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(SR1(Direct())), OptimizationOptions())
    @test _alloc == 0

    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(BFGS(Direct())), OptimizationOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(BFGS(Direct())), OptimizationOptions())
    @test _alloc == 0

    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(DFP(Direct())), OptimizationOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(DFP(Direct())), OptimizationOptions())
    @test _alloc == 0

    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), OptimizationOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), OptimizationOptions())
    @test _alloc == 0

    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), OptimizationOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), OptimizationOptions())
    @test _alloc == 0
end


using DoubleFloats
@testset "Test double floats" begin
	function fdouble(x)
	    fx = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
	    return fx
	end
	function fgdouble(G, x)
	    fx = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

        G1 = -2 * (1 - x[1]) - 400 * (x[2] - x[1]^2) * x[1]
        G2 = 200 * (x[2] - x[1]^2)
        G = [G1,G2]
	    return fx, G
	end
	f_obj = OptimizationProblem(ScalarObjective(fdouble, nothing, fgdouble, nothing, nothing, nothing, nothing, nothing))
	res = res = solve(f_obj, Double64.([1,2]), LineSearch(GradientDescent(Inverse())), OptimizationOptions(;g_abstol=1e-32, maxiter=100000))
    @test res.info.minimum < 1e-45
    res = res = solve(f_obj, Double64.([1,2]), LineSearch(BFGS(Inverse())), OptimizationOptions(;g_abstol=1e-32))
    @test res.info.minimum < 1e-45
	res = res = solve(f_obj, Double64.([1,2]), LineSearch(DFP(Inverse())), OptimizationOptions(;g_abstol=1e-32))
    @test res.info.minimum < 1e-45
	res = res = solve(f_obj, Double64.([1,2]), LineSearch(SR1(Inverse())), OptimizationOptions(;g_abstol=1e-32))
    @test res.info.minimum < 1e-45
end


function myfun(x::T) where T
    fx = T(x^4 + sin(x))
    return fx
end
function myfun(∇f, x::T) where T
    ∇f = T(4*x^3 + cos(x))
    fx = myfun(x)
    fx, ∇f
end
function myfun(∇f, ∇²f, x::T) where T<:Real
    ∇²f = T(12*x^2 - sin(x))
    fx, ∇f = myfun(∇f, x)
    T(fx), ∇f, ∇²f
end
@testset "scalar return types" begin
    for T in (Float16, Float32, Float64, Rational{BigInt}, Double32, Double64)
        if T == Rational{BigInt}
            options = OptimizationOptions()
        else
            options = OptimizationOptions(g_abstol=eps(T), g_reltol=T(0))
        end
        for M in (SR1, BFGS, DFP, Newton)
            if M == Newton
                obj = OptimizationProblem(ScalarObjective(myfun, nothing, myfun, myfun, nothing, nothing, nothing, nothing); inplace=false)
                res = solve(obj, T(3.1), LineSearch(M()), options)
                @test all(isa.([res.info.minimum, res.info.∇fz, res.info.minimizer], T))
            else
                obj = OptimizationProblem(ScalarObjective(myfun, nothing, myfun, myfun, nothing, nothing, nothing, nothing); inplace=false)
                res = solve(obj, T(3.1), LineSearch(M(Direct())), options)
                @test all(isa.([res.info.minimum, res.info.∇fz, res.info.minimizer], T))
                res = solve(obj, T(3.1), LineSearch(M(Inverse())), options)
                @test all(isa.([res.info.minimum, res.info.∇fz, res.info.minimizer], T))
            end
        end
    end
end



using GeometryTypes
@testset "GeometryTypes" begin
    function fu(G, x)
        fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

        if !(G == nothing)
            G1 = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
            G2 = 200.0 * (x[2] - x[1]^2)
            gx = Point(G1,G2)

            return fx, gx
        else
            return fx
        end
    end
    fu(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    f_obj = OptimizationProblem(ScalarObjective(fu, nothing, fu, nothing, nothing, nothing, nothing, nothing); inplace=false)
    res = solve(f_obj, Point(1.3,1.3), LineSearch(GradientDescent(Inverse())), OptimizationOptions())
    @test res.info.minimum < 1e-9
    res = solve(f_obj, Point(1.3,1.3), LineSearch(BFGS(Inverse())), OptimizationOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3,1.3), LineSearch(DFP(Inverse())), OptimizationOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3,1.3), LineSearch(SR1(Inverse())), OptimizationOptions())
    @test res.info.minimum < 1e-10
    
    res = solve(f_obj, Point(1.3,1.3), LineSearch(GradientDescent(Direct())), OptimizationOptions())
    @test res.info.minimum < 1e-9
    res = solve(f_obj, Point(1.3,1.3), LineSearch(BFGS(Direct())), OptimizationOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3,1.3), LineSearch(DFP(Direct())), OptimizationOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3,1.3), LineSearch(SR1(Direct())), OptimizationOptions())
    @test res.info.minimum < 1e-10
end




@testset "quadratics" begin
    A = rand(2,2)
    A = abs.(A)
    A = Symmetric(A*A')
    x = rand(2)
    b = A*x


    quadf(x) = -dot(b, x) + dot(x, A*x)/2
    function quadfg(G, x)
        G.=A*x-b
        quadf(x), G
    end
    function quadfgh(G, H, x)
        H .= A
        f, G = quadfg(G, x)
        f, G, H
    end

    quadprob = OptimizationProblem(ScalarObjective(quadf, nothing, quadfg, quadfgh, nothing, nothing, nothing, nothing); inplace=true)

    for approx in (GradientDescent(), BFGS(Inverse()), BFGS(Direct()), DBFGS(), SR1(Inverse()), SR1(Direct()), DFP(), Newton(), BB(), LBFGS()) # CBB
        lsres =  solve(quadprob, zeros(2), LineSearch(approx, Backtracking()), OptimizationOptions(maxiter=20000))
        println(rpad(summary(approx), 40), "   ||   $(rpad(lsres.info.iter, 5))   ||   $(lsres.info.∇fz)")
    end
end
@testset "batched" begin
    

end

@testset "MArray" begin
    f = OPT_PROBS["himmelblau"]["array"]["mutating"]
    x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
    prob = OptimizationProblem(f)

    x0m = @MVector [-1.0, 0.0, 0.0]
    x0 = [-1.0, 0.0, 0.0]
    @time res = solve(prob, copy(x0m), ConjugateGradient(update=VPRP()), OptimizationOptions());
    @time res = solve(prob, copy(x0), ConjugateGradient(update=VPRP()), OptimizationOptions());
    # workaround for https://github.com/JuliaArrays/StaticArrays.jl/issues/828
    @time res = solve(prob, (copy(x0m), MArray(I+x0m*x0m')), LineSearch(BFGS()), OptimizationOptions());
    @time res = solve(prob, copy(x0), LineSearch(BFGS()), OptimizationOptions());
    @time res = solve(prob, (copy(x0m), MArray(I+x0m*x0m')), LineSearch(SR1()), OptimizationOptions());
    @time res = solve(prob, copy(x0), LineSearch(SR1()), OptimizationOptions());
#    @time res = solve(prob, (copy(x0m), MArray(I+x0m*x0m')), TrustRegion(DBFGS(), Dogleg()), OptimizationOptions());
    @time res = solve(prob, copy(x0), TrustRegion(DBFGS(), Dogleg()), OptimizationOptions());
end




# using NLSolvers, StaticArrays, Test
# @testset "mixed optimization problems" begin
# function theta(x)
#    if x[1] > 0
#        return atan(x[2] / x[1]) / (2.0 * pi)
#    else
#        return (pi + atan(x[2] / x[1])) / (2.0 * pi)
#    end
# end
# f(x) = 100.0 * ((x[3] - 10.0 * theta(x))^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

# function f∇f!(x, ∇f)
#     if !(∇f==nothing)
#         if ( x[1]^2 + x[2]^2 == 0 )
#             dtdx1 = 0;
#             dtdx2 = 0;
#         else
#             dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
#             dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
#         end
#         ∇f[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
#             200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
#         ∇f[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
#             200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
#         ∇f[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
#     end

#     fx = f(x)
#     objective_return(fx, ∇f)
# end

# function f∇f(x, ∇f)
#     if !(∇f == nothing)
#         ∇f = similar(x)
#     end
#     fx, ∇f = f∇f!(gx, x)
#     objective_return(fx, ∇f)
# end
# function f∇fs(x, ∇f)
#     if !(∇f == nothing)
#         if ( x[1]^2 + x[2]^2 == 0 )
#             dtdx1 = 0;
#             dtdx2 = 0;
#         else
#             dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
#             dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
#         end

#         s1 = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
#             200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 )
#         s2 = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
#             200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 )
#         s3 = 200.0*(x[3]-10.0*theta(x)) + 2.0*x[3]
#         ∇f = @SVector [s1, s2, s3]
#     end
#     objective_return(f(x), ∇f)
# end
# obj_inplace = OnceDiffed(f∇f!)
# obj_outofplace = OnceDiffed(f∇f)
# obj_static = OnceDiffed(f∇fs)

# x0 = [-1.0, 0.0, 0.0]
# xopt = [1.0, 0.0, 0.0]
# x0s = @SVector [-1.0, 0.0, 0.0]

# println("Starting  from: ", x0)
# println("Targeting     : ", xopt)


# res = minimize(obj_inplace, copy(x0), NelderMead(), OptimizationOptions())
# print("NN  $(summary(NelderMead()))         ")
# @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.nm_obj, Inf), res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), SimulatedAnnealing(), OptimizationOptions())
# print("NN  $(summary(SimulatedAnnealing()))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=HZ()), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient(update=HZ())))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=CD()), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient(update=CD())))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=HS()), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient(update=HS())))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=FR()), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient(update=FR())))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=PRP()), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient(update=PRP(;plus=false))))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=PRP(plus=true)), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient(update=PRP(;plus=true))))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=VPRP()), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient(update=VPRP())))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=LS()), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient(update=LS())))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=DY()), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient(update=DY())))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)

# for _method in (GradientDescent, LBFGS, BFGS, DBFGS, DFP, SR1)
# # for _method in (GradientDescent, BFGS, DBFGS, DFP, SR1)
#     methodtxt = summary(_method()) 
#     for m in (Inverse(), Direct())
#         mtxt = m isa Inverse ? "(inverse): " : "(direct):  "
#         if _method == LBFGS && m isa Inverse
#             res = minimize!(obj_inplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN! $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#         elseif _method !== LBFGS
#             res = minimize(obj_inplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN  $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             res = minimize!(obj_inplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN! $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             res = minimize(obj_static, x0s, LineSearch(_method(m)), OptimizationOptions())
#             print("NN  $_method(S) $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             if m isa Direct && !(_method == GradientDescent)
#                 println("Trust region: NWI")
#                 res = minimize!(obj_inplace, copy(x0), TrustRegion(_method(Direct()), NWI()), OptimizationOptions())
#                 print("NN! $_method    $mtxt")
#                 @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#                 println("Trust region: NTR")
#                 res = minimize!(obj_inplace, copy(x0), TrustRegion(_method(Direct()), NTR()), OptimizationOptions())
#                 print("NN! $_method    $mtxt")
#                 @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             end
#         end
#     end
# end
# println()

# res = minimize(obj_inplace, x0, LineSearch(BFGS(Inverse())), OptimizationOptions())
# @test res.info.iter == 30
# @printf("NN  BFGS    (inverse): %2.2e  %2.2e %2.2e %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
# res = minimize!(obj_inplace, copy(x0), LineSearch(BFGS(Inverse()), Backtracking()), OptimizationOptions())
# @test res.info.iter == 30
# @printf("NN! BFGS    (inverse): %2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
# res = minimize!(obj_inplace, copy(x0), LineSearch(BFGS(Inverse()), Backtracking(interp=FFQuadInterp())), OptimizationOptions())
# @test res.info.iter == 30
# @printf("NN! BFGS    (inverse, quad): %2.2e %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
# res = minimize(obj_static, x0s, LineSearch(BFGS(Inverse())), OptimizationOptions())
# @test res.info.iter == 30
# @printf("NN  BFGS(S) (inverse): %2.2e %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
# res = minimize(obj_static, x0s, LineSearch(BFGS(Inverse()), Backtracking(interp=FFQuadInterp())), OptimizationOptions())
# @test res.info.iter == 30
# @printf("NN  BFGS(S) (inverse, quad): %2.2e %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)

# x0 = rand(3)
# x0s = SVector{3}(x0)
# println("\nFrom a random point: ", x0)
# res = minimize(obj_inplace, copy(x0), NelderMead(), OptimizationOptions())
# print("NN  $(summary(NelderMead()))         ")
# @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.nm_obj, Inf), res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), SimulatedAnnealing(), OptimizationOptions())
# print("NN  $(summary(SimulatedAnnealing()))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
# res = minimize(obj_inplace, copy(x0), ConjugateGradient(), OptimizationOptions())
# print("NN  $(summary(ConjugateGradient()))         ")
# @printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)

# for _method in (GradientDescent, LBFGS, BFGS, DBFGS, DFP, SR1)
#     methodtxt = summary(_method()) 
#     for m in (Inverse(), Direct())
#         mtxt = m isa Inverse ? "(inverse): " : "(direct):  "
#         if _method == LBFGS && m isa Inverse
#             res = minimize!(obj_inplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN! $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#         elseif _method !== LBFGS
#             res = minimize(obj_inplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN  $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             res = minimize!(obj_inplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN! $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             res = minimize(obj_static, x0s, LineSearch(_method(m)), OptimizationOptions())
#             print("NN  $_method(S) $mtxt")
#             @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             if m isa Direct && !(_method == GradientDescent)
#                 println("Trust region: NWI")
#                 res = minimize!(obj_inplace, copy(x0), TrustRegion(_method(Direct()), NWI()), OptimizationOptions())
#                 print("NN! $_method    $mtxt")
#                 @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#                 println("Trust region: NTR")
#                 res = minimize!(obj_inplace, copy(x0), TrustRegion(_method(Direct()), NTR()), OptimizationOptions())
#                 print("NN! $_method    $mtxt")
#                 @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             end
#         end
#     end
# end
# println()

# function himmelblau!(x, ∇f)
#     if !(∇f == nothing)
#         ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
#             44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
#         ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
#             4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
#     end

#     fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
#     objective_return(fx, ∇f)
# end


# function himmelblaus(x, ∇f)
#     fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
#     if !(∇f == nothing)
#         ∇f1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
#             44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
#         ∇f2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
#             4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
#         ∇f = @SVector([∇f1, ∇f2])
#     end
#     objective_return(fx, ∇f)
# end

# function himmelblau(x, ∇f)
#     g = ∇f == nothing ? ∇f : similar(x)

#     return himmelblau!(x, g)
# end

# him_inplace = OnceDiffed(himmelblau!)
# him_static = OnceDiffed(himmelblaus)
# him_outofplace = OnceDiffed(himmelblau)

# println("\nHimmelblau function")
# x0 = [3.0, 1.0]
# x0s = SVector{2}(x0)
# minimizers = [[3.0,2.0],[-2.805118,3.131312],[-3.779310,-3.283186],[3.584428,-1.848126]]
# for _method in (GradientDescent, LBFGS, BFGS, DBFGS, DFP, SR1)
# # for _method in (GradientDescent, BFGS, DBFGS, DFP, SR1)
#     methodtxt = summary(_method()) 
#     for m in (Inverse(), Direct())
#         mtxt = m isa Inverse ? "(inverse): " : "(direct):  "
#         if _method == LBFGS && m isa Inverse
#             res = minimize!(him_inplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN! $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#         elseif _method !== LBFGS
#             res = minimize(him_outofplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN  $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             res = minimize!(him_inplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN! $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             res = minimize(him_static, x0s, LineSearch(_method(m)), OptimizationOptions())
#             print("NN  $_method(S) $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             if m isa Direct && !(_method == GradientDescent)
#                 println("Trust region: NWI")
#                 res = minimize!(him_inplace, copy(x0), TrustRegion(_method(Direct()), NWI()), OptimizationOptions())
#                 print("NN! $_method    $mtxt")
#                 @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#                 println("Trust region: NTR")
#                 res = minimize!(him_inplace, copy(x0), TrustRegion(_method(Direct()), NTR()), OptimizationOptions())
#                 print("NN! $_method    $mtxt")
#                 @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             end
#         end
#     end
# end
# println()
# println()
# xrand = rand(2)
# xrands = SVector{2}(xrand)
# println("\nFrom a random point: ", xrand)

# for _method in (GradientDescent, LBFGS, BFGS, DBFGS, SR1)
# # for _method in (GradientDescent, BFGS, DBFGS, DFP, SR1)
#     methodtxt = summary(_method()) 
#     for m in (Inverse(), Direct())
#         mtxt = m isa Inverse ? "(inverse): " : "(direct):  "
#         if _method == LBFGS && m isa Inverse
#             res = minimize!(him_inplace, copy(x0), LineSearch(_method(m)), OptimizationOptions())
#             print("NN! $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#         elseif _method !== LBFGS
#             res = minimize(him_outofplace, copy(xrand), LineSearch(_method(m)), OptimizationOptions())
#             print("NN  $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             res = minimize!(him_inplace, copy(xrand), LineSearch(_method(m)), OptimizationOptions())
#             print("NN! $_method    $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             res = minimize(him_static, xrands, LineSearch(_method(m)), OptimizationOptions())
#             print("NN  $_method(S) $mtxt")
#             @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             if m isa Direct && !(_method == GradientDescent)
#                 println("Trust region: NWI")
#                 res = minimize!(him_inplace, copy(xrand), TrustRegion(_method(Direct()), NWI()), OptimizationOptions())
#                 print("NN! $_method    $mtxt")
#                 @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#                 println("Trust region: NTR")
#                 res = minimize!(him_inplace, copy(xrand), TrustRegion(_method(Direct()), NTR()), OptimizationOptions())
#                 print("NN! $_method    $mtxt")
#                 @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
#             end
#         end
#     end
# end
# println()

# end
















#=
using NLSolvers
function f∇f!(x, ∇f)
   if !(∇f==nothing)
       if ( x[1]^2 + x[2]^2 == 0 )
           dtdx1 = 0;
           dtdx2 = 0;
       else
           dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
           dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
       end
       ∇f[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
           200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
       ∇f[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
           200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
       ∇f[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
   end

   fx = f(x)
   return ∇f==nothing ? fx : (fx, ∇f)
end

f(x) = 100.0 * ((x[3] - 10.0 * theta(x))^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2
function theta(x)
  if x[1] > 0
      return atan(x[2] / x[1]) / (2.0 * pi)
  else
      return (pi + atan(x[2] / x[1])) / (2.0 * pi)
  end
end

#= test solve interface =#
obj! = OnceDiffed(f∇f!)
nm_prob! = MinProblem(obj=obj!)
solve(nm_prob!, rand(3), NelderMead(), OptimizationOptions())


solve(obj!, -rand(3)*9 .- 3, NLSolvers.NelderMead(), OptimizationOptions())
V = [[1.0,1.0,1.0], [0.0,1.0,1.0],[0.40,0.0,0.0],[-1.0,2.0,.03]]
F = obj!.(V)

splx = NLSolvers.ValuedSimplex(V, F)

solve(obj!, splx, NLSolvers.NelderMead(), OptimizationOptions())
solve(nm_prob!, splx, NelderMead(), OptimizationOptions())


function powell(x, ∇f)
    fx = (x[1] + 10.0 * x[2])^2 + 5.0 * (x[3] - x[4])^2 +
        (x[2] - 2.0 * x[3])^4 + 10.0 * (x[1] - x[4])^4

    if !isa(∇f, Nothing)
        ∇f[1] = 2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3
        ∇f[2] = 20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3
        ∇f[3] = 10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3
        ∇f[4] = -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3
        return ∇f, fx
    end
    return fx
end

obj_powell = OnceDiffed(powell)
# Define vertices in V
x0 = [1.0,1.0,1.0,1.0]
V = [copy(x0)]
for i = 1:4
    push!(V, x0+39*rand(4))
end
V
F = obj_powell.(V)
splx = NLSolvers.ValuedSimplex(V, F)
@time solve(obj_powell, splx, NLSolvers.NelderMead(), OptimizationOptions(maxiter=3000))
@time solve(obj_powell, copy(x0), NLSolvers.NelderMead(), OptimizationOptions(maxiter=3000))
@allocated solve(obj_powell, splx, NLSolvers.NelderMead(), OptimizationOptions(maxiter=3000))
solve(obj_powell, copy(x0), NLSolvers.NelderMead(), OptimizationOptions(maxiter=3000))


function extros!(x, storage)
   n = length(x)
   jodd = 1:2:n-1
   jeven = 2:2:n
   xt = similar(x)
   @. xt[jodd] = 10.0 * (x[jeven] - x[jodd]^2)
   @. xt[jeven] = 1.0 - x[jodd]

   if !isa(storage, Nothing)
       @. storage[jodd] = -20.0 * x[jodd] * xt[jodd] - xt[jeven]
       @. storage[jeven] = 10.0 * xt[jodd]
       return 0.5*sum(abs2, xt), storage
   end
   return 0.5*sum(abs2, xt)
end
x0=rand(300)
V = [copy(x0)]
for i = 1:300
    push!(V, x0+39*rand(length(x0)))
end
V
extros_obj! = OnceDiffed(extros!)
F = extros_obj!.(V)
splx = NLSolvers.ValuedSimplex(V, F)
@time solve(extros_obj!, splx, NLSolvers.NelderMead(), OptimizationOptions(maxiter=3000))
=#