using Test, NLSolvers
using LinearAlgebra: norm

@testset "DFSANE spectral coefficient" begin
    # The roots of this polynomial system sit at (1, 1); the start is far
    # enough away that a degenerate spectral coefficient (the y = 0 defect,
    # where every iteration fell back to the safeguard) does not converge
    # within the iteration budget.
    function F_rosen!(Fx, x)
        Fx[1] = 1 - x[1]
        Fx[2] = 10(x[2] - x[1]^2)
        return Fx
    end

    @testset "converges with a real spectral step" begin
        obj = NLSolvers.VectorObjective(F_rosen!, nothing, nothing, nothing)
        prob = NEqProblem(obj)
        res = solve(prob, [-1.3, 2.7], DFSANE(), NEqOptions())
        @test norm(res.info.best_residual, Inf) < 1e-4
        @test res.info.iter < 200
    end

    @testset "the accepted residual is not re-evaluated" begin
        calls = Vector{Vector{Float64}}()
        function F_count!(Fx, x)
            push!(calls, copy(x))
            return F_rosen!(Fx, x)
        end
        obj = NLSolvers.VectorObjective(F_count!, nothing, nothing, nothing)
        prob = NEqProblem(obj)
        res = solve(prob, [-1.3, 2.7], DFSANE(), NEqOptions())
        dups = count(i -> calls[i] == calls[i-1], 2:length(calls))
        @test dups == 0
        # one initial evaluation plus at least one line-search trial per
        # iteration, and no hidden extras beyond the trials
        @test length(calls) >= res.info.iter + 1
    end
end
