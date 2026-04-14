@testset "Dogleg Direct vs Inverse" begin
    # Test that Dogleg works correctly with both Direct and Inverse
    # Hessian approximations. Dogleg requires PD approximations, so
    # only BFGS, DFP, and DBFGS are appropriate (not SR1 or Newton
    # on non-convex problems).

    # --- Himmelblau ---
    f = OPT_PROBS["himmelblau"]["array"]["mutating"]
    prob_h = OptimizationProblem(f)

    @testset "Himmelblau - $name" for (name, scheme) in [
        ("BFGS Inverse", BFGS(Inverse())),
        ("BFGS Direct", BFGS(Direct())),
        ("DFP Inverse", DFP(Inverse())),
        ("DFP Direct", DFP(Direct())),
        ("DBFGS Inverse", DBFGS(Inverse())),
        ("DBFGS Direct", DBFGS(Direct())),
    ]
        x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
        res = solve(prob_h, x0, TrustRegion(scheme, Dogleg()), OptimizationOptions())
        @test res.info.minimum < 1e-12
    end

    # --- Exponential (minimum at 2.0) ---
    f = OPT_PROBS["exponential"]["array"]["mutating"]
    prob_e = OptimizationProblem(f)

    @testset "Exponential - $name" for (name, scheme) in [
        ("BFGS Inverse", BFGS(Inverse())),
        ("DFP Inverse", DFP(Inverse())),
        ("DBFGS Inverse", DBFGS(Inverse())),
    ]
        x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
        res = solve(prob_e, x0, TrustRegion(scheme, Dogleg()), OptimizationOptions())
        @test res.info.minimum ≈ 2.0 atol = 1e-8
    end

    # --- Rosenbrock ---
    function rosen!(∇f, x)
        fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        if ∇f !== nothing
            ∇f[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
            ∇f[2] = 200.0 * (x[2] - x[1]^2)
        end
        return fx
    end
    function rosen_fg!(∇f, x)
        fx = rosen!(∇f, x)
        return fx, ∇f
    end
    rosen_obj = ScalarObjective(
        x -> rosen!(nothing, x),
        (∇f, x) -> (rosen!(∇f, x); ∇f),
        rosen_fg!,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
    prob_r = OptimizationProblem(rosen_obj)

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

@testset "Dogleg OutOfPlace" begin
    # OutOfPlace Rosenbrock (returns new arrays, no mutation)
    function rosen_oop(∇f, x)
        fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        if ∇f !== nothing
            g1 = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
            g2 = 200.0 * (x[2] - x[1]^2)
            return fx, [g1, g2]
        end
        return fx
    end
    function rosen_oop_fg(∇f, x)
        fx, g = rosen_oop(true, x)
        return fx, g
    end
    rosen_oop_obj = ScalarObjective(
        x -> rosen_oop(nothing, x),
        nothing,
        rosen_oop_fg,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
    prob_oop = OptimizationProblem(rosen_oop_obj; inplace = false)

    @testset "OOP Array - $name" for (name, scheme) in [
        ("BFGS Inverse", BFGS(Inverse())),
        ("BFGS Direct", BFGS(Direct())),
    ]
        x0 = [-1.0, 2.0]
        res = solve(prob_oop, x0, TrustRegion(scheme, Dogleg()), OptimizationOptions())
        @test res.info.minimum < 1e-10
    end

    # StaticArrays
    function rosen_static(∇f, x)
        fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        if ∇f !== nothing
            g1 = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
            g2 = 200.0 * (x[2] - x[1]^2)
            return fx, @SVector [g1, g2]
        end
        return fx
    end
    function rosen_static_fg(∇f, x)
        fx, g = rosen_static(true, x)
        return fx, g
    end
    rosen_static_obj = ScalarObjective(
        x -> rosen_static(nothing, x),
        nothing,
        rosen_static_fg,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
    prob_static = OptimizationProblem(rosen_static_obj; inplace = false)

    @testset "StaticArrays - $name" for (name, scheme) in [
        ("BFGS Inverse", BFGS(Inverse())),
        ("BFGS Direct", BFGS(Direct())),
    ]
        x0 = @SVector [-1.0, 2.0]
        res = solve(prob_static, x0, TrustRegion(scheme, Dogleg()), OptimizationOptions())
        @test res.info.minimum < 1e-10
    end
end
