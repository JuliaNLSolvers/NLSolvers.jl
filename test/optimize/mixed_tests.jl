# Sweeps the gradient-based methods over the Fletcher-Powell helical valley
# and Himmelblau's function in in-place, out-of-place, and StaticArrays form.
using NLSolvers, StaticArrays, Test
using LinearAlgebra: norm
import Random
isdefined(Main, :TestProblems) || include(joinpath(@__DIR__, "testproblems.jl"))

Random.seed!(886444)

# One method through a line search in-place, out-of-place, and with
# StaticArrays; Direct approximations also through both nearly-exact
# trust-region solvers. Asserts each run reaches fmin within ftol.
function qn_method_runs(problem, scheme, m, x0, x0s, fmin, ftol)
    prob = OptimizationProblem(problem.inplace)
    prob_oop = OptimizationProblem(problem.inplace; inplace = false)
    prob_static = OptimizationProblem(problem.static; inplace = false)
    res = solve(prob, copy(x0), LineSearch(scheme), OptimizationOptions())
    @test res.info.minimum <= fmin + ftol
    res = solve(prob_oop, copy(x0), LineSearch(scheme), OptimizationOptions())
    @test res.info.minimum <= fmin + ftol
    res = solve(prob_static, x0s, LineSearch(scheme), OptimizationOptions())
    @test res.info.minimum <= fmin + ftol
    if m isa Direct
        for sp in (NWI(), NTR())
            res = solve(prob, copy(x0), TrustRegion(scheme, sp), OptimizationOptions())
            @test res.info.minimum <= fmin + ftol
        end
    end
end

function qn_method_sweep(problem, x0, x0s, fmin, ftol)
    @testset "LBFGS" begin
        prob = OptimizationProblem(problem.inplace)
        res = solve(prob, copy(x0), LineSearch(LBFGS()), OptimizationOptions())
        @test res.info.minimum <= fmin + ftol
    end
    for _method in (GradientDescent, BFGS, DBFGS, DFP)
        for m in (Inverse(), Direct())
            @testset "$_method $(typeof(m))" begin
                qn_method_runs(problem, _method(m), m, x0, x0s, fmin, ftol)
            end
        end
    end
end

@testset "mixed optimization problems" begin
    @testset "Fletcher-Powell helical valley" begin
        fp = TestProblems.fletcher_powell
        prob = OptimizationProblem(fp.inplace)
        x0 = fp.x0()
        x0s = @SVector [-1.0, 0.0, 0.0]

        @testset "conjugate gradient update $(typeof(update))" for update in (
            HZ(),
            CD(),
            FR(),
            PRP(plus = false),
            PRP(plus = true),
            VPRP(),
            DY(),
        )
            res = solve(
                prob,
                copy(x0),
                ConjugateGradient(update = update),
                OptimizationOptions(),
            )
            @test res.info.minimum <= fp.minimum + 1e-8
        end

        # HS and LS stall far from the minimum on this start
        @testset "conjugate gradient update $(typeof(update)) stalls" for update in
                                                                          (HS(), LS())
            res = solve(
                prob,
                copy(x0),
                ConjugateGradient(update = update),
                OptimizationOptions(),
            )
            @test_broken res.info.minimum <= fp.minimum + 1e-8
        end

        @testset "samplers" begin
            f_at_x0 = fp.inplace.f(x0)
            res = solve(prob, copy(x0), NelderMead(), OptimizationOptions())
            @test res.info.minimum < f_at_x0
            res = solve(prob, copy(x0), SimulatedAnnealing(), OptimizationOptions())
            @test res.info.minimum < f_at_x0
        end

        @testset "quasi-Newton methods" begin
            qn_method_sweep(fp, x0, x0s, fp.minimum, 1e-8)
        end

        @testset "backtracking" begin
            res = solve(
                prob,
                copy(x0),
                LineSearch(BFGS(Inverse()), Backtracking()),
                OptimizationOptions(),
            )
            @test res.info.minimum <= fp.minimum + 1e-8
            @test res.info.iter == 30
            res = solve(
                OptimizationProblem(fp.static; inplace = false),
                x0s,
                LineSearch(BFGS(Inverse()), Backtracking()),
                OptimizationOptions(),
            )
            @test res.info.minimum <= fp.minimum + 1e-8
            @test res.info.iter == 30
            # the quadratic interpolation variant stalls on this start
            for (p, x) in
                ((prob, x0), (OptimizationProblem(fp.static; inplace = false), x0s))
                res = solve(
                    p,
                    copy(x),
                    LineSearch(BFGS(Inverse()), Backtracking(interp = FFQuadInterp())),
                    OptimizationOptions(),
                )
                @test_broken res.info.minimum <= fp.minimum + 1e-8
            end
        end
    end

    @testset "Himmelblau" begin
        hb = TestProblems.himmelblau
        x0 = hb.x0()
        x0s = SVector{2}(hb.x0())

        @testset "quasi-Newton methods" begin
            qn_method_sweep(hb, x0, x0s, hb.minimum, 1e-8)
        end

        @testset "solutions are known minimizers" begin
            for m in (BFGS(Inverse()), BFGS(Direct()), DFP(Inverse()))
                res = solve(
                    OptimizationProblem(hb.inplace),
                    hb.x0(),
                    LineSearch(m),
                    OptimizationOptions(),
                )
                dist =
                    minimum(norm(res.info.solution - xopt, Inf) for xopt in hb.minimizers)
                @test dist < 1e-4
            end
        end
    end

    # SR1 has no curvature safeguard on its update, so several line-search
    # runs stall away from the minimum; the trust-region runs are fine. The
    # broken marks document today's behavior per problem and form.
    @testset "SR1" begin
        fp = TestProblems.fletcher_powell
        hb = TestProblems.himmelblau

        @testset "Fletcher-Powell line searches stall" begin
            prob = OptimizationProblem(fp.inplace)
            prob_oop = OptimizationProblem(fp.inplace; inplace = false)
            prob_static = OptimizationProblem(fp.static; inplace = false)
            x0s = @SVector [-1.0, 0.0, 0.0]
            for m in (Inverse(), Direct())
                res = solve(prob, fp.x0(), LineSearch(SR1(m)), OptimizationOptions())
                @test_broken res.info.minimum <= fp.minimum + 1e-8
                res = solve(prob_oop, fp.x0(), LineSearch(SR1(m)), OptimizationOptions())
                @test_broken res.info.minimum <= fp.minimum + 1e-8
                res = solve(prob_static, x0s, LineSearch(SR1(m)), OptimizationOptions())
                @test_broken res.info.minimum <= fp.minimum + 1e-8
            end
        end

        @testset "Himmelblau" begin
            prob = OptimizationProblem(hb.inplace)
            prob_oop = OptimizationProblem(hb.inplace; inplace = false)
            prob_static = OptimizationProblem(hb.static; inplace = false)
            x0s = SVector{2}(hb.x0())
            # Inverse: only the in-place form converges
            res = solve(prob, hb.x0(), LineSearch(SR1(Inverse())), OptimizationOptions())
            @test res.info.minimum <= hb.minimum + 1e-8
            res =
                solve(prob_oop, hb.x0(), LineSearch(SR1(Inverse())), OptimizationOptions())
            @test_broken res.info.minimum <= hb.minimum + 1e-8
            res = solve(prob_static, x0s, LineSearch(SR1(Inverse())), OptimizationOptions())
            @test_broken res.info.minimum <= hb.minimum + 1e-8
            # Direct converges in every form
            res = solve(prob, hb.x0(), LineSearch(SR1(Direct())), OptimizationOptions())
            @test res.info.minimum <= hb.minimum + 1e-8
            res = solve(prob_oop, hb.x0(), LineSearch(SR1(Direct())), OptimizationOptions())
            @test res.info.minimum <= hb.minimum + 1e-8
            res = solve(prob_static, x0s, LineSearch(SR1(Direct())), OptimizationOptions())
            @test res.info.minimum <= hb.minimum + 1e-8
        end

        @testset "trust regions converge" begin
            for problem in (fp, hb), sp in (NWI(), NTR())
                res = solve(
                    OptimizationProblem(problem.inplace),
                    problem.x0(),
                    TrustRegion(SR1(Direct()), sp),
                    OptimizationOptions(),
                )
                @test res.info.minimum <= problem.minimum + 1e-8
            end
        end
    end
end
