using NLSolvers
using LinearAlgebra
using Test

@testset "QN skip strategies" begin
    # Set up Rosenbrock — a standard test where BFGS/L-BFGS should converge
    function rosenbrock!(∇f, x)
        fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        if ∇f !== nothing
            ∇f[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
            ∇f[2] = 200.0 * (x[2] - x[1]^2)
        end
        fx
    end
    function rosenbrock_fg!(∇f, x)
        fx = rosenbrock!(∇f, x)
        fx, ∇f
    end
    obj = ScalarObjective(
        f=rosenbrock!,
        g=rosenbrock!,
        fg=rosenbrock_fg!,
    )
    prob = OptimizationProblem(obj)
    x0 = [-1.0, 1.0]
    opts = OptimizationOptions()

    @testset "should_skip unit tests" begin
        s = [1.0, 0.0]
        y = [1.0, 1.0]
        # NoSkip never skips
        @test NLSolvers.should_skip(NoSkip(), s, y, 1.0) == false
        @test NLSolvers.should_skip(NoSkip(), s, y, 0.0) == false

        # LBFGSBSkip: skip when s'y ≤ eps * |dφ0|
        # s'y = 1.0, dφ0 = 1.0 → 1.0 ≤ eps * 1.0 is false
        @test NLSolvers.should_skip(LBFGSBSkip(), s, y, 1.0) == false
        # s'y ≈ 0, dφ0 = 1.0 → should skip
        s_tiny = [eps(Float64), 0.0]
        @test NLSolvers.should_skip(LBFGSBSkip(), s_tiny, y, 1.0) == true
        # s'y = 1.0, dφ0 = 0.0 → 1.0 ≤ 0.0 is false (no skip)
        @test NLSolvers.should_skip(LBFGSBSkip(), s, y, 0.0) == false

        # LiFukushimaSkip: skip when s'y / ||s||² < ε * ||∇f||
        # s'y = 1.0, ||s||² = 1.0, ε = 1e-6, ||∇f|| = 1.0 → 1.0 < 1e-6 is false
        @test NLSolvers.should_skip(LiFukushimaSkip(), s, y, 1.0) == false
        # s'y small relative to ||s||² and gradient norm → should skip
        # s'y = 1e-12, ||s||² = 1.0, ε = 1e-6, ||∇f|| = 1.0 → 1e-12 < 1e-6
        s_weak = [1.0, 0.0]
        y_weak = [1e-12, 0.0]
        @test NLSolvers.should_skip(LiFukushimaSkip(), s_weak, y_weak, 1.0) == true
        # Large gradient norm makes condition easier to trigger
        @test NLSolvers.should_skip(LiFukushimaSkip(ε=1.0), s, y, 2.0) == true  # 1.0/1.0 < 1.0*2.0
    end

    @testset "skip_aux dispatch" begin
        ∇f = [3.0, 4.0]
        dφ0 = -5.0
        @test NLSolvers.skip_aux(NoSkip(), dφ0, ∇f) === nothing
        @test NLSolvers.skip_aux(LBFGSBSkip(), dφ0, ∇f) == 5.0
        @test NLSolvers.skip_aux(LiFukushimaSkip(), dφ0, ∇f) == 5.0
    end

    @testset "qn_skip dispatch" begin
        @test NLSolvers.qn_skip(BFGS()) isa NoSkip
        @test NLSolvers.qn_skip(BFGS(; skip=LBFGSBSkip())) isa LBFGSBSkip
        @test NLSolvers.qn_skip(BFGS(; skip=LiFukushimaSkip())) isa LiFukushimaSkip
        @test NLSolvers.qn_skip(LBFGS()) isa NoSkip
        @test NLSolvers.qn_skip(LBFGS(; skip=LBFGSBSkip())) isa LBFGSBSkip
        @test NLSolvers.qn_skip(LBFGS(; skip=LiFukushimaSkip())) isa LiFukushimaSkip
        @test NLSolvers.qn_skip(SR1()) isa NoSkip
        @test NLSolvers.qn_skip(DFP()) isa NoSkip
    end

    @testset "BFGS with skip strategies converges" begin
        for skip in (NoSkip(), LBFGSBSkip(), LiFukushimaSkip())
            res = solve(prob, copy(x0), LineSearch(BFGS(; skip)), opts)
            @test res.info.minimum < 1e-10
        end
    end

    @testset "L-BFGS with skip strategies converges" begin
        for skip in (NoSkip(), LBFGSBSkip(), LiFukushimaSkip())
            res = solve(prob, copy(x0), LineSearch(LBFGS(; skip)), opts)
            @test res.info.minimum < 1e-10
        end
    end

    @testset "BFGS constructor backwards compatibility" begin
        @test BFGS() isa BFGS
        @test BFGS(Inverse()) isa BFGS{Inverse}
        @test BFGS(Direct()) isa BFGS{Direct}
        @test BFGS(; inverse=false) isa BFGS{Direct}
        @test BFGS(; skip=LBFGSBSkip()).skip isa LBFGSBSkip
    end

    @testset "LBFGS constructor backwards compatibility" begin
        @test LBFGS() isa LBFGS
        @test LBFGS(5) isa LBFGS
        @test LBFGS(Inverse(), 10) isa LBFGS
        # 4-arg form used by preconditioning tests
        @test LBFGS(Inverse(), NLSolvers.TwoLoop(), 5, nothing) isa LBFGS
        @test LBFGS(; memory=10, skip=LiFukushimaSkip()).skip isa LiFukushimaSkip
        @test LBFGS(; memory=10).memory == 10
    end

    @testset "Shanno-Phua scaling" begin
        s = [1.0, 2.0]
        y = [3.0, 4.0]
        sp = NLSolvers.ShannoPhua()
        # γ = s'y / y'y = (3+8) / (9+16) = 11/25
        @test sp(s, y) ≈ 11.0 / 25.0
        # Verify it uses y, not just s (old bug returned 1.0 for real vectors)
        @test sp(s, y) != 1.0
    end

    @testset "max_restarts option" begin
        @test OptimizationOptions().max_restarts == 10
        @test OptimizationOptions(max_restarts=3).max_restarts == 3
    end

    @testset "QN restart on line search failure" begin
        # Use HZAW which returns NaN on failure, triggering the restart path
        # Use very tight maxiter on HZAW to force failures
        tight_hz = HZAW(maxiter=1)
        res = solve(prob, copy(x0), LineSearch(BFGS(), tight_hz), OptimizationOptions(maxiter=200, max_restarts=5))
        # Should still run without error (restarts handle the LS failures)
        @test isfinite(res.info.minimum)

        # Same for L-BFGS
        res_l = solve(prob, copy(x0), LineSearch(LBFGS(), tight_hz), OptimizationOptions(maxiter=200, max_restarts=5))
        @test isfinite(res_l.info.minimum)
    end

    @testset "L-BFGS memory not incremented on skip" begin
        # Use LiFukushimaSkip with a huge ε so it always skips
        always_skip = LiFukushimaSkip(ε=Inf)
        short_opts = OptimizationOptions(maxiter=100)
        res_skip = solve(prob, copy(x0), LineSearch(LBFGS(; skip=always_skip)), short_opts)
        res_normal = solve(prob, copy(x0), LineSearch(LBFGS()), short_opts)
        # With all updates skipped, L-BFGS degrades to steepest descent.
        # Normal L-BFGS should converge much better in the same iteration budget.
        @test res_normal.info.minimum < res_skip.info.minimum
    end
end
