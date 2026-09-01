using NLSolvers
using StaticArrays
using Test
using LinearAlgebra: norm
isdefined(Main, :TestProblems) || include(joinpath(@__DIR__, "optimize/testproblems.jl"))

_normdiff_allocs(x, z) = @allocated NLSolvers.normdiff(x, z)
_normdiff_allocs(x, z, p) = @allocated NLSolvers.normdiff(x, z, p)
_bertsekas_rhs_allocs(x, x⁺, g, p, α, activeset) =
    @allocated NLSolvers.bertsekas_rhs(x, x⁺, g, p, α, activeset)

@testset "lazy differences" begin
    x, z = rand(100), rand(100)
    @test NLSolvers.normdiff(x, z) ≈ norm(x .- z)
    @test NLSolvers.normdiff(x, z, 1) ≈ norm(x .- z, 1)
    @test NLSolvers.normdiff(x, z, Inf) == norm(x .- z, Inf)
    # the rescaled norm does not overflow where sqrt(sum(abs2, ...)) would
    big = fill(1e300, 4)
    @test NLSolvers.normdiff(big, -big) == norm(big .- (-big)) == 4e300

    xs, zs = SVector(1.0, 2.0), SVector(3.0, -1.0)
    @test NLSolvers.normdiff(xs, zs) ≈ norm(xs .- zs)

    _normdiff_allocs(x, z)
    _normdiff_allocs(x, z, Inf)
    @test _normdiff_allocs(x, z) == 0
    @test _normdiff_allocs(x, z, Inf) == 0

    g, p = rand(100), rand(100)
    activeset = isodd.(1:100)
    α = 0.5
    x⁺ = clamp.(x .+ α .* p, 0.0, 1.0)
    @test NLSolvers.bertsekas_rhs(x, x⁺, g, p, α, activeset) ≈
          sum(NLSolvers.bertsekas_R.(x, x⁺, g, p, α, activeset))
    _bertsekas_rhs_allocs(x, x⁺, g, p, α, activeset)
    @test _bertsekas_rhs_allocs(x, x⁺, g, p, α, activeset) == 0

    lower, upper = fill(-0.5, 100), fill(0.5, 100)
    @test NLSolvers.projected_gradient_norm(x, g, lower, upper, 2) ≈
          norm(x .- clamp.(x .- g, lower, upper))
    @test NLSolvers.projected_gradient_norm(x, g, lower, upper, Inf) ==
          norm(x .- clamp.(x .- g, lower, upper), Inf)
    _pgnorm_allocs(x, g, lower, upper, p) =
        @allocated NLSolvers.projected_gradient_norm(x, g, lower, upper, p)
    _pgnorm_allocs(x, g, lower, upper, 2)
    @test _pgnorm_allocs(x, g, lower, upper, 2) == 0
end

@testset "box_retract!! keeps immutable arrays out of place" begin
    lower, upper, x, p, α = [-1.0, -1.0], [1.0, 1.0], [0.0, 0.9], [0.5, 0.5], 1.0
    expected = NLSolvers.box_retract.(lower, upper, x, p, α)

    buffer = zeros(2)
    out = NLSolvers.box_retract!!(buffer, lower, upper, x, p, α)
    @test out === buffer
    @test out == expected

    xs = SVector(0.0, 0.9)
    outs = NLSolvers.box_retract!!(
        SVector(0.0, 0.0),
        SVector(-1.0, -1.0),
        SVector(1.0, 1.0),
        xs,
        SVector(0.5, 0.5),
        α,
    )
    @test outs isa SVector
    @test outs == expected
end

@testset "x_norm is called with an iterator" begin
    f = TestProblems.himmelblau.inplace
    x0 = TestProblems.himmelblau.x0()
    prob = OptimizationProblem(f)
    calls = Ref(0)
    xnorm = y -> (calls[] += 1; maximum(abs, y))
    res = solve(prob, x0, LineSearch(BFGS()), OptimizationOptions(x_norm = xnorm))
    @test calls[] > 0
    @test res.info.minimum < 1e-8
end

@testset "infeasible initial guess" begin
    @test NLSolvers.check_feasible([1.5], [1.0], [2.0]) === nothing
    @test_throws ErrorException NLSolvers.check_feasible([0.0], [1.0], [2.0])
    @test_throws DimensionMismatch NLSolvers.check_feasible([1.5, 1.0], [1.0], [2.0])

    prob_bounds = OptimizationProblem(
        obj = TestProblems.himmelblau.inplace,
        bounds = ([3.5, -9.0], [13.0, 4.0]),
    )
    @test_throws ErrorException solve(
        prob_bounds,
        [0.0, 0.0],
        ActiveBox(),
        OptimizationOptions(),
    )
end
