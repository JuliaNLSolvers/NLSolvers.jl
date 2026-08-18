using Test, NLSolvers
isdefined(Main, :TestProblems) || include(joinpath(@__DIR__, "testproblems.jl"))

@testset "Dogleg Direct vs Inverse" begin
    # Test that Dogleg works correctly with both Direct and Inverse
    # Hessian approximations. Dogleg requires PD approximations, so
    # only BFGS, DFP, and DBFGS are appropriate (not SR1 or Newton
    # on non-convex problems).

    # --- Himmelblau ---
    f = TestProblems.himmelblau.inplace
    prob_h = OptimizationProblem(f)

    @testset "Himmelblau - $name" for (name, scheme) in [
        ("BFGS Inverse", BFGS(Inverse())),
        ("BFGS Direct", BFGS(Direct())),
        ("DFP Inverse", DFP(Inverse())),
        ("DFP Direct", DFP(Direct())),
        ("DBFGS Inverse", DBFGS(Inverse())),
        ("DBFGS Direct", DBFGS(Direct())),
    ]
        x0 = TestProblems.himmelblau.x0()
        res = solve(prob_h, x0, TrustRegion(scheme, Dogleg()), OptimizationOptions())
        @test res.info.minimum < 1e-12
    end

    # --- Exponential (minimum at 2.0) ---
    f = TestProblems.exponential.inplace
    prob_e = OptimizationProblem(f)

    @testset "Exponential - $name" for (name, scheme) in [
        ("BFGS Inverse", BFGS(Inverse())),
        ("DFP Inverse", DFP(Inverse())),
        ("DBFGS Inverse", DBFGS(Inverse())),
    ]
        x0 = TestProblems.exponential.x0()
        res = solve(prob_e, x0, TrustRegion(scheme, Dogleg()), OptimizationOptions())
        @test res.info.minimum ≈ 2.0 atol = 1e-8
    end

    # --- Rosenbrock ---
    prob_r = OptimizationProblem(TestProblems.rosenbrock.inplace)

    @testset "Rosenbrock - $name" for (name, scheme) in [
        ("BFGS Inverse", BFGS(Inverse())),
        ("BFGS Direct", BFGS(Direct())),
        ("DBFGS Inverse", DBFGS(Inverse())),
        ("DBFGS Direct", DBFGS(Direct())),
    ]
        x0 = [-1.0, 2.0]
        res = solve(prob_r, x0, TrustRegion(scheme, Dogleg()), OptimizationOptions())
        @test res.info.minimum < 1e-10
    end
end
