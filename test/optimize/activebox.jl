using Test, NLSolvers, LinearAlgebra

function _ab_steplength(f, x, p, g, lower, upper, activeset, φ0)
    prob = OptimizationProblem(ScalarObjective(f = f); inplace = false)
    φ = (; prob, ∇fz = g, z = copy(x), x, p, φ0 = φ0, dφ0 = dot(g, p))
    NLSolvers.find_steplength(
        NLSolvers.OutOfPlace(),
        NLSolvers.ArmijoBertsekas(),
        φ,
        1.0,
        g,
        activeset,
        lower,
        upper,
        x,
        p,
        g,
        activeset,
    )
end

@testset "ActiveBox" begin
    @testset "ArmijoBertsekas rejects uphill steps" begin
        # f rises along p while g'p = -2 says descent, so the Armijo
        # threshold at any α sits strictly below φ0 = 0.5 and no step
        # may be accepted with f above it
        f_up(x) = 0.5 + 0.0001 * x[1]
        α, f_α, ls_success, x⁺ =
            _ab_steplength(f_up, [0.0], [1.0], [-2.0], [-10.0], [10.0], [false], 0.5)
        @test f_α.ϕ <= 0.5
        @test α < 1.0
    end

    @testset "ArmijoBertsekas accepts sufficient decrease" begin
        # threshold at α = 1 is φ0 - decrease*|g'p| = 0.5 - 2e-4
        f_ok(x) = 0.5 - 0.001 * x[1]
        α, f_α, ls_success, x⁺ =
            _ab_steplength(f_ok, [0.0], [1.0], [-2.0], [-10.0], [10.0], [false], 0.5)
        @test α == 1.0
        @test f_α.ϕ == 0.499
        @test ls_success
        # insufficient decrease at α = 1 must backtrack
        f_short(x) = 0.5 - 0.0001 * x[1]
        α, f_α, ls_success, x⁺ =
            _ab_steplength(f_short, [0.0], [1.0], [-2.0], [-10.0], [10.0], [false], 0.5)
        @test α < 1.0
    end

    @testset "active coordinates use the projected decrease" begin
        # x sits on the lower bound with the gradient pointing out, so the
        # projected point does not move and no decrease is required
        f_flat(x) = 0.5
        α, f_α, ls_success, x⁺ =
            _ab_steplength(f_flat, [0.0], [-1.0], [2.0], [0.0], [10.0], [true], 0.5)
        @test α == 1.0
        @test x⁺ == [0.0]
    end

    @testset "bounded solves decrease monotonically" begin
        # the fixture is defined in problems.jl, included by runtests.jl
        prob = OptimizationProblem(
            obj = OPT_PROBS["himmelblau"]["array"]["mutating"],
            bounds = ([3.5, -9.0], [13.0, 4.0]),
        )
        fs = Float64[]
        cb = info -> (push!(fs, info.state.fz); false)
        res = solve(prob, [3.7, 2.0], ActiveBox(), OptimizationOptions(callback = cb))
        @test res.info.solution ≈ [3.5, 1.6165968467448326]
        @test all(diff(fs) .<= 0)
    end
end
