using NLSolvers
using LinearAlgebra
using SparseDiffTools
using SparseArrays
using IterativeSolvers
using ForwardDiff
using Test
using GenericLinearAlgebra
isdefined(Main, :TestProblems) || include(joinpath(@__DIR__, "testproblems.jl"))

@testset "optimization interface" begin
    # TODO
    # Make a more efficient MeritObjective that returns something that acts as the actual thing if requested (mostly for debug)
    # but can also be efficiently used to get cauchy and newton
    #
    # Make DOGLEG work also with BFGS (why is convergence so slow?)
    # # Look into what caches are created
    #
    # NelderMead
    # time limit not enforced in @show solve(NelderMead)
    # no convergence crit either
    #
    # ParticleSwarm
    # no @show solve
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

    # Todo Normed residuals doesn't have batched. Try ParticleSwarm on equations
    #### OPTIMIZATION
    f = TestProblems.himmelblau.inplace
    x0 = TestProblems.himmelblau.x0()
    prob = OptimizationProblem(f)
    prob_oop = OptimizationProblem(f; inplace = false)
    prob_bounds = OptimizationProblem(obj = f, bounds = ([-5.0, -9.0], [13.0, 4.0]))
    prob_bounds_oop =
        OptimizationProblem(obj = f, bounds = ([-5.0, -9.0], [13.0, 4.0]); inplace = false)
    prob_on_bounds = OptimizationProblem(obj = f, bounds = ([3.5, -9.0], [13.0, 4.0]))
    prob_on_bounds_oop =
        OptimizationProblem(obj = f, bounds = ([3.5, -9.0], [13.0, 4.0]); inplace = false)

    res = solve(prob, x0, NelderMead(), OptimizationOptions())
    @test all(x0 .== [3.0, 2.0])
    @test res.info.minimum == 0.0
    @test all(solution(res) .== [3.0, 2.0])

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob_oop, x0, NelderMead(), OptimizationOptions())
    @test all(x0 .== [3.0, 1.0])
    @test res.info.minimum == 0.0

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob_bounds, x0, ParticleSwarm(), OptimizationOptions())
    @test_broken all(x0 .== [3.0, 2.0])
    @test res.info.minimum == 0.0

    x0 = TestProblems.himmelblau.x0() .+ 1
    res = solve(prob_on_bounds_oop, x0, ActiveBox(), OptimizationOptions())
    @test_broken all(x0 .== [3.0, 1.0])
    xbounds = [3.5, 1.616596846883819]
    @test res.info.minimum == NLSolvers.value(prob_on_bounds, xbounds)

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob_bounds, x0, SimulatedAnnealing(), OptimizationOptions())
    @test_broken all(x0 .== [3.0, 2.0])
    @test res.info.minimum < 1e-1

    solve(prob, PureRandomSearch(lb = [0.0, 0.0], ub = [4.0, 4.0]), OptimizationOptions())
    solve(
        prob_oop,
        PureRandomSearch(lb = [0.0, 0.0], ub = [4.0, 4.0]),
        OptimizationOptions(),
    )

    solve(prob, [0.0, 0.0], SimulatedAnnealing(), OptimizationOptions())
    solve(prob_oop, [0.0, 0.0], SimulatedAnnealing(), OptimizationOptions())


    #x0 = TestProblems.himmelblau.x0()
    #@show solve(prob_bounds, x0, SIMAN(), OptimizationOptions())
    #@test all(x0 .== [3.0,1.0])

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(), OptimizationOptions())
    #@test all(x0 .== [3.0,2.0])
    @test res.info.minimum < 1e-12

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(SR1()), OptimizationOptions())
    @test res.info.minimum < 1e-12

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(DFP()), OptimizationOptions())
    @test res.info.minimum < 1e-12

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(BFGS()), OptimizationOptions())
    @test res.info.minimum < 1e-12

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(LBFGS()), OptimizationOptions())
    @test res.info.minimum < 1e-12

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(LBFGS(), HZAW()), OptimizationOptions())
    @test res.info.minimum < 1e-12

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(LBFGS(), Backtracking()), OptimizationOptions())
    @test res.info.minimum < 1e-12

    #@test all(x0 .== [3.0,2.0])
    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(ConjugateGradient(), HZAW()), OptimizationOptions())
    @test res.info.minimum < 1e-12

    x0 = TestProblems.himmelblau.x0()
    res =
        solve(prob_oop, x0, LineSearch(ConjugateGradient(), HZAW()), OptimizationOptions())
    @test res.info.minimum < 1e-12

    x0 = TestProblems.himmelblau.x0()
    res = solve(
        prob,
        x0,
        LineSearch(ConjugateGradient(update = HS()), HZAW()),
        OptimizationOptions(),
    )
    @test res.info.minimum < 1e-12

    x0 = TestProblems.himmelblau.x0()
    res = solve(
        prob_oop,
        x0,
        LineSearch(ConjugateGradient(update = HS()), HZAW()),
        OptimizationOptions(),
    )
    @test res.info.minimum < 1e-12

    # Stalls at [3, 1] with default @show solve
    x0 = 1.0 .+ TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(Newton()), OptimizationOptions())
    @test res.info.minimum < 1e-12

    #@test all(x0 .== [3.0,2.0])
    x0 = 1.0 .+ TestProblems.himmelblau.x0()
    res = solve(prob, x0, LineSearch(Newton(), HZAW()), OptimizationOptions())
    @test res.info.minimum < 1e-12
    #@test all(x0 .== [3.0,2.0])

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, TrustRegion(), OptimizationOptions())
    @test_broken all(x0 .== [3.0, 2.0])
    @test res.info.minimum < 1e-16

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, TrustRegion(DBFGS(), Dogleg()), OptimizationOptions())
    @test res.info.minimum < 1e-16

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, TrustRegion(BFGS(), Dogleg()), OptimizationOptions())
    @test res.info.minimum < 1e-16

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, TrustRegion(SR1(), NTR()), OptimizationOptions())
    @test res.info.minimum < 1e-16

    # not PSD
    #x0 = TestProblems.himmelblau.x0()
    #res = solve(prob, x0, TrustRegion(Newton(), Dogleg()), OptimizationOptions())
    #@test res.info.minimum < 1e-16

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, TrustRegion(BFGS(), Dogleg()), OptimizationOptions())
    @test res.info.minimum < 1e-16

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, TrustRegion(DBFGS(), Dogleg()), OptimizationOptions())
    @test res.info.minimum < 1e-16

    x0 = TestProblems.himmelblau.x0()
    res = solve(prob, x0, Adam(), OptimizationOptions(maxiter = 20000))
    @test res.info.minimum < 1e-16

    ## Notice that prob is only used for value so this should be extremely generic! It does need a comparison though.
    res = solve(
        prob,
        PureRandomSearch(lb = [0.0, 0.0], ub = [4.0, 4.0]),
        OptimizationOptions(),
    )

    f = TestProblems.exponential.inplace
    x0 = TestProblems.exponential.x0()
    prob = OptimizationProblem(f)
    prob_bounds = OptimizationProblem(obj = f, bounds = ([-5.0, -9.0], [13.0, 4.0]))
    prob_on_bounds = OptimizationProblem(obj = f, bounds = ([3.5, -9.0], [13.0, 4.0]))

    x0 = TestProblems.exponential.x0()
    res = solve(prob, x0, TrustRegion(DBFGS(), Dogleg()), OptimizationOptions())
    @test res.info.minimum == TestProblems.exponential.minimum

    x0 = TestProblems.exponential.x0()
    res = solve(prob, x0, TrustRegion(BFGS(), Dogleg()), OptimizationOptions())
    @test res.info.minimum == TestProblems.exponential.minimum

    x0 = TestProblems.exponential.x0()
    res = solve(prob, x0, TrustRegion(Newton(), Dogleg()), OptimizationOptions())
    @test res.info.minimum == TestProblems.exponential.minimum

    x0 = TestProblems.exponential.x0()
    res = solve(prob, x0, TrustRegion(Newton(), NTR()), OptimizationOptions())
    @test res.info.minimum == TestProblems.exponential.minimum

    x0 = TestProblems.exponential.x0()
    res = solve(prob, x0, TrustRegion(Newton(), NWI()), OptimizationOptions())
    @test res.info.minimum == TestProblems.exponential.minimum

    x0 = big.(TestProblems.exponential.x0())
    res = solve(prob, x0, TrustRegion(Newton(), NWI()), OptimizationOptions())

    #=
    x0 = TestProblems.exponential.x0()
    res = solve(prob, x0, TrustRegion(SR1(; scaling = OrenLeuenberger), NTR()), OptimizationOptions())
    @test_broken res.info.minimum == TestProblems.exponential.minimum  # SR1 Direct safeguard (Nocedal & Wright eq. 6.26) skips updates needed here
    =#

    x0 = TestProblems.exponential.x0()
    res = solve(prob, x0, TrustRegion(SR1(Inverse()), NTR()), OptimizationOptions())
    @test res.info.minimum == TestProblems.exponential.minimum
end

const brent_f(x) = sign(x)
const brent_scalar = ScalarObjective(; f = brent_f)
const brent_prob = OptimizationProblem(brent_scalar, (-2.0, 2.0))
@testset "univariate nonalloc" begin
    @allocated solve(brent_prob, BrentMin(), OptimizationOptions())
    alloc = @allocated solve(brent_prob, BrentMin(), OptimizationOptions())
    @test alloc == 0
end

const f3(x) = abs(x)
const obj3 = ScalarObjective(; f = f3)
const prob3 = OptimizationProblem(obj3, (-10.1, 9.0))
@testset "brentmin" begin
    f(x) = (5.0 + x)^2.0
    obj = ScalarObjective(; f)
    prob = OptimizationProblem(obj, (-10.1, 9.0))

    f2(x) = abs(x)
    obj2 = ScalarObjective(; f = f2)
    prob2 = OptimizationProblem(obj2, (-10.1, 9.0))

    @test all(abs.(solve(prob, BrentMin(), OptimizationOptions()).info.minimum) .< 1e-8)
    @test all(abs.(solve(prob2, BrentMin(), OptimizationOptions()).info.minimum) .< 1e-8)
    @test all(abs.(solve(prob3, BrentMin(), OptimizationOptions()).info.minimum) .< 1e-8)
    #@allocated solve(prob3, BrentMin(), OptimizationOptions()) == 0
    #@test @allocated solve(prob3, BrentMin(), OptimizationOptions()) == 0

    # Test the "evaluate_bounds"
    for f in [x -> sign(x), x -> -sign(x)]
        for x in [(-2.0, 2.0), (-1.0, 2.0), (-2.0, 1.0)]
            obj = ScalarObjective(; f)
            prob = OptimizationProblem(obj, x)
            result = solve(prob, BrentMin(), OptimizationOptions())
            @test result.info.minimum == -1.0
        end
    end
end


const statictest_s0 = TestProblems.himmelblau.state0
const statictest_prob = OptimizationProblem(TestProblems.himmelblau.static; inplace = false)
@testset "staticopt" begin
    res = solve(statictest_prob, statictest_s0, LineSearch(Newton()), OptimizationOptions())
    @allocated solve(
        statictest_prob,
        statictest_s0,
        LineSearch(Newton()),
        OptimizationOptions(),
    )
    alloc = @allocated solve(
        statictest_prob,
        statictest_s0,
        LineSearch(Newton()),
        OptimizationOptions(),
    )
    @test alloc == 0

    _res =
        solve(statictest_prob, statictest_s0, LineSearch(Newton()), OptimizationOptions())
    _alloc = @allocated solve(
        statictest_prob,
        statictest_s0,
        LineSearch(Newton()),
        OptimizationOptions(),
    )
    @test _alloc == 0
    @test norm(_res.info.∇fz, Inf) < 1e-8

    _res = solve(
        statictest_prob,
        statictest_s0,
        LineSearch(Newton(), Backtracking()),
        OptimizationOptions(),
    )
    _alloc = @allocated solve(
        statictest_prob,
        statictest_s0,
        LineSearch(Newton(), Backtracking()),
        OptimizationOptions(),
    )
    @test _alloc == 0
    @test norm(_res.info.∇fz, Inf) < 1e-8
end

@testset "newton" begin
    test_x0 = [2.0, 2.0]
    test_prob = OptimizationProblem(TestProblems.himmelblau.inplace; inplace = true)
    res = solve(test_prob, copy(test_x0), LineSearch(Newton()), OptimizationOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8

    test_x0 = [2.0, 2.0]
    test_prob = OptimizationProblem(TestProblems.himmelblau.inplace; inplace = false)
    res = solve(test_prob, test_x0, LineSearch(Newton()), OptimizationOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8


    test_x0 = [2.0, 2.0]
    test_prob = OptimizationProblem(TestProblems.himmelblau.inplace; inplace = true)
    res = solve(
        test_prob,
        copy(test_x0),
        LineSearch(Newton(; linsolve = positive_linsolve)),
        OptimizationOptions(),
    )
    @test norm(res.info.∇fz, Inf) < 1e-8

    test_x0 = [2.0, 2.0]
    test_prob = OptimizationProblem(TestProblems.himmelblau.inplace; inplace = false)
    res = solve(
        test_prob,
        test_x0,
        LineSearch(Newton(; linsolve = positive_linsolve)),
        OptimizationOptions(),
    )
    @test norm(res.info.∇fz, Inf) < 1e-8
end
@testset "Newton linsolve" begin
    test_prob = OptimizationProblem(TestProblems.himmelblau.inplace; inplace = true)
    res_lu = solve(
        test_prob,
        (copy([2.0, 2.0]), [0.0 0.0; 0.0 0.0]),
        LineSearch(Newton(; linsolve = (d, B, g) -> ldiv!(d, lu(B), g))),
        OptimizationOptions(),
    )
    @test norm(res_lu.info.∇fz, Inf) < 1e-8
    res_qr = solve(
        test_prob,
        (copy([2.0, 2.0]), [0.0 0.0; 0.0 0.0]),
        LineSearch(Newton(; linsolve = (d, B, g) -> ldiv!(d, qr(B), g))),
        OptimizationOptions(),
    )
    @test norm(res_qr.info.∇fz, Inf) < 1e-8

    test_prob = OptimizationProblem(TestProblems.himmelblau.inplace; inplace = false)
    res_qr = solve(
        test_prob,
        (copy([2.0, 2.0]), [0.0 0.0; 0.0 0.0]),
        LineSearch(Newton(; linsolve = (B, g) -> qr(B) \ g)),
        OptimizationOptions(),
    )
    @test norm(res_qr.info.∇fz, Inf) < 1e-8
    test_prob = OptimizationProblem(TestProblems.himmelblau.inplace; inplace = false)
    res_lu = solve(
        test_prob,
        (copy([2.0, 2.0]), [0.0 0.0; 0.0 0.0]),
        LineSearch(Newton(; linsolve = (B, g) -> lu(B) \ g)),
        OptimizationOptions(),
    )
    @test norm(res_lu.info.∇fz, Inf) < 1e-8
end















const static_x0 = TestProblems.fletcher_powell.state0[1]
const static_prob_qn = TestProblems.fletcher_powell.static_problem
@testset "no alloc static" begin

    @testset "no alloc" begin
        @allocated solve(
            static_prob_qn,
            static_x0,
            LineSearch(BFGS(Inverse()), Backtracking()),
            OptimizationOptions(),
        )
        _alloc = @allocated solve(
            static_prob_qn,
            static_x0,
            LineSearch(BFGS(Inverse()), Backtracking()),
            OptimizationOptions(),
        )
        @test _alloc == 0

        solve(
            static_prob_qn,
            static_x0,
            LineSearch(BFGS(Direct()), Backtracking()),
            OptimizationOptions(),
        )
        _alloc = @allocated solve(
            static_prob_qn,
            static_x0,
            LineSearch(BFGS(Direct()), Backtracking()),
            OptimizationOptions(),
        )
        @test _alloc == 0

        solve(
            static_prob_qn,
            static_x0,
            LineSearch(DFP(Inverse()), Backtracking()),
            OptimizationOptions(),
        )
        _alloc = @allocated solve(
            static_prob_qn,
            static_x0,
            LineSearch(DFP(Inverse()), Backtracking()),
            OptimizationOptions(),
        )
        @test _alloc == 0

        solve(
            static_prob_qn,
            static_x0,
            LineSearch(DFP(Direct()), Backtracking()),
            OptimizationOptions(),
        )
        _alloc = @allocated solve(
            static_prob_qn,
            static_x0,
            LineSearch(DFP(Direct()), Backtracking()),
            OptimizationOptions(),
        )
        @test _alloc == 0

        solve(
            static_prob_qn,
            static_x0,
            LineSearch(SR1(Inverse()), Backtracking()),
            OptimizationOptions(),
        )
        _alloc = @allocated solve(
            static_prob_qn,
            static_x0,
            LineSearch(SR1(Inverse()), Backtracking()),
            OptimizationOptions(),
        )
        @test _alloc == 0

        solve(
            static_prob_qn,
            static_x0,
            LineSearch(SR1(Direct()), Backtracking()),
            OptimizationOptions(),
        )
        _alloc = @allocated solve(
            static_prob_qn,
            static_x0,
            LineSearch(SR1(Direct()), Backtracking()),
            OptimizationOptions(),
        )
        @test _alloc == 0
    end
end

Random.seed!(4568532)
solve(static_prob_qn, rand(3), Adam(), OptimizationOptions(maxiter = 1000))
solve(static_prob_qn, rand(3), AdaMax(), OptimizationOptions(maxiter = 1000))




@testset "bound newton" begin
    f = TestProblems.himmelblau.inplace
    x0 = TestProblems.himmelblau.x0()
    prob = OptimizationProblem(f)
    prob_oop = OptimizationProblem(f; inplace = false)
    prob_bounds = OptimizationProblem(obj = f, bounds = ([-5.0, -9.0], [13.0, 4.0]))
    prob_bounds_oop =
        OptimizationProblem(obj = f, bounds = ([-5.0, -9.0], [13.0, 4.0]); inplace = false)
    prob_on_bounds = OptimizationProblem(obj = f, bounds = ([3.5, -9.0], [13.0, 4.0]))
    prob_on_bounds_oop =
        OptimizationProblem(obj = f, bounds = ([3.5, -9.0], [13.0, 4.0]); inplace = false)

    start = [3.7, 2.0]

    res_unc = solve(
        prob_bounds,
        copy(start),
        LineSearch(Newton(), Backtracking()),
        OptimizationOptions(),
    )
    @test res_unc.info.solution ≈ [3.0, 2.0]
    res_con = solve(prob_bounds, copy(start), ActiveBox(), OptimizationOptions())
    @test res_con.info.solution ≈ [3.0, 2.0]
    res_unc = solve(
        prob_on_bounds,
        copy(start),
        LineSearch(Newton(), Backtracking()),
        OptimizationOptions(),
    )
    @test res_unc.info.solution ≈ [3.0, 2.0]
    res_con = solve(prob_on_bounds, copy(start), ActiveBox(), OptimizationOptions())
    @test res_con.info.solution ≈ [3.5, 1.6165968467448326]

    res_con_matrix = solve(
        prob_on_bounds,
        (copy(start), [1.0 0.0; 0.0 1.0]),
        ActiveBox(),
        OptimizationOptions(),
    )
    @test res_con_matrix.info.B isa Matrix
    res_con_mmatrix = solve(
        prob_on_bounds,
        (copy(start), @MMatrix([1.0 0.0; 0.0 1.0])),
        ActiveBox(),
        OptimizationOptions(),
    )
    @test res_con_mmatrix.info.B isa MMatrix
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

const scalar_prob_oop = OptimizationProblem(
    ScalarObjective(
        fourth_f,
        nothing,
        fourth_fg,
        fourth_fgh,
        nothing,
        nothing,
        nothing,
        nothing,
    );
    inplace = false,
)
@testset "scalar no-alloc" begin
    _alloc = @allocated solve(
        scalar_prob_oop,
        4.0,
        LineSearch(SR1(Direct())),
        OptimizationOptions(),
    )
    _alloc = @allocated solve(
        scalar_prob_oop,
        4.0,
        LineSearch(SR1(Direct())),
        OptimizationOptions(),
    )
    @test _alloc == 0

    _alloc = @allocated solve(
        scalar_prob_oop,
        4.0,
        LineSearch(BFGS(Direct())),
        OptimizationOptions(),
    )
    _alloc = @allocated solve(
        scalar_prob_oop,
        4.0,
        LineSearch(BFGS(Direct())),
        OptimizationOptions(),
    )
    @test _alloc == 0

    _alloc = @allocated solve(
        scalar_prob_oop,
        4.0,
        LineSearch(DFP(Direct())),
        OptimizationOptions(),
    )
    _alloc = @allocated solve(
        scalar_prob_oop,
        4.0,
        LineSearch(DFP(Direct())),
        OptimizationOptions(),
    )
    @test _alloc == 0

    _alloc =
        @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), OptimizationOptions())
    _alloc =
        @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), OptimizationOptions())
    @test _alloc == 0

    _alloc =
        @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), OptimizationOptions())
    _alloc =
        @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), OptimizationOptions())
    @test _alloc == 0
end


using DoubleFloats
@testset "Test double floats" begin
    f_obj = OptimizationProblem(TestProblems.rosenbrock.outofplace)
    res =
        res = solve(
            f_obj,
            Double64.([1, 2]),
            LineSearch(GradientDescent(Inverse())),
            OptimizationOptions(; g_abstol = 1e-32, maxiter = 100000),
        )
    @test res.info.minimum < 1e-45
    res =
        res = solve(
            f_obj,
            Double64.([1, 2]),
            LineSearch(BFGS(Inverse())),
            OptimizationOptions(; g_abstol = 1e-32),
        )
    @test res.info.minimum < 1e-45
    res =
        res = solve(
            f_obj,
            Double64.([1, 2]),
            LineSearch(DFP(; inverse = true, scaling = OrenLuenberger())),
            OptimizationOptions(; g_abstol = 1e-32),
        )
    @test res.info.minimum < 1e-45
    res =
        res = solve(
            f_obj,
            Double64.([1, 2]),
            LineSearch(SR1(Inverse())),
            OptimizationOptions(; g_abstol = 1e-32),
        )
    @test res.info.minimum < 1e-45
end


function myfun(x::T) where {T}
    fx = T(x^4 + sin(x))
    return fx
end
function myfun(∇f, x::T) where {T}
    ∇f = T(4 * x^3 + cos(x))
    fx = myfun(x)
    fx, ∇f
end
function myfun(∇f, ∇²f, x::T) where {T<:Real}
    ∇²f = T(12 * x^2 - sin(x))
    fx, ∇f = myfun(∇f, x)
    T(fx), ∇f, ∇²f
end
@testset "scalar return types" begin
    for T in (Float16, Float32, Float64, Rational{BigInt}, Double32, Double64)
        if T == Rational{BigInt}
            options = OptimizationOptions()
        else
            options = OptimizationOptions(g_abstol = eps(T), g_reltol = T(0))
        end
        for M in (SR1, BFGS, DFP, Newton)
            if M == Newton
                obj = OptimizationProblem(
                    ScalarObjective(
                        myfun,
                        nothing,
                        myfun,
                        myfun,
                        nothing,
                        nothing,
                        nothing,
                        nothing,
                    );
                    inplace = false,
                )
                res = solve(obj, T(3.1), LineSearch(M()), options)
                @test all(isa.([res.info.minimum, res.info.∇fz, res.info.solution], T))
            else
                obj = OptimizationProblem(
                    ScalarObjective(
                        myfun,
                        nothing,
                        myfun,
                        myfun,
                        nothing,
                        nothing,
                        nothing,
                        nothing,
                    );
                    inplace = false,
                )
                res = solve(obj, T(3.1), LineSearch(M(Direct())), options)
                @test all(isa.([res.info.minimum, res.info.∇fz, res.info.solution], T))
                res = solve(obj, T(3.1), LineSearch(M(Inverse())), options)
                @test all(isa.([res.info.minimum, res.info.∇fz, res.info.solution], T))
            end
        end
    end
end







@testset "quadratics" begin
    A = rand(2, 2)
    A = abs.(A)
    A = Symmetric(A * A')
    x = rand(2)
    b = A * x


    quadf(x) = -dot(b, x) + dot(x, A * x) / 2
    function quadfg(G, x)
        G .= A * x - b
        quadf(x), G
    end
    function quadfgh(G, H, x)
        H .= A
        f, G = quadfg(G, x)
        f, G, H
    end

    quadprob = OptimizationProblem(
        ScalarObjective(
            quadf,
            nothing,
            quadfg,
            quadfgh,
            nothing,
            nothing,
            nothing,
            nothing,
        );
        inplace = true,
    )

    for approx in (
        GradientDescent(),
        BFGS(Inverse()),
        BFGS(Direct()),
        DBFGS(),
        SR1(Inverse()),
        SR1(Direct()),
        DFP(),
        Newton(),
        BB(),
        LBFGS(),
    ) # CBB
        lsres = solve(
            quadprob,
            zeros(2),
            LineSearch(approx, Backtracking()),
            OptimizationOptions(maxiter = 20000),
        )
        println(
            rpad(summary(approx), 40),
            "   ||   $(rpad(lsres.info.iter, 5))   ||   $(lsres.info.∇fz)",
        )
    end
end
@testset "batched" begin end

@testset "MArray" begin
    f = TestProblems.himmelblau.inplace
    x0 = TestProblems.himmelblau.x0()
    prob = OptimizationProblem(f)

    x0m = @MVector [-1.0, 0.0, 0.0]
    x0 = [-1.0, 0.0, 0.0]
    @time res =
        solve(prob, copy(x0m), ConjugateGradient(update = VPRP()), OptimizationOptions())
    @time res =
        solve(prob, copy(x0), ConjugateGradient(update = VPRP()), OptimizationOptions())
    # workaround for https://github.com/JuliaArrays/StaticArrays.jl/issues/828
    @time res = solve(
        prob,
        (copy(x0m), MArray(I + x0m * x0m')),
        LineSearch(BFGS()),
        OptimizationOptions(),
    )
    @time res = solve(prob, copy(x0), LineSearch(BFGS()), OptimizationOptions())
    @time res = solve(
        prob,
        (copy(x0m), MArray(I + x0m * x0m')),
        LineSearch(SR1()),
        OptimizationOptions(),
    )
    @time res = solve(prob, copy(x0), LineSearch(SR1()), OptimizationOptions())
    #    @time res = solve(prob, (copy(x0m), MArray(I+x0m*x0m')), TrustRegion(DBFGS(), Dogleg()), OptimizationOptions());
    @time res = solve(prob, copy(x0), TrustRegion(DBFGS(), Dogleg()), OptimizationOptions())
end
