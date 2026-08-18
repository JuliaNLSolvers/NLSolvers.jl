using NLSolvers, Test
using LinearAlgebra: norm

# https://github.com/JuliaNLSolvers/NLSolvers.jl/issues/88
# LineSearch(Newton()) defaults to HZAW which evaluates the gradient of the
# line objective, so the merit objective must implement upto_gradient. The
# TrustRegion solve for NEqProblem must have default options.
@testset "Newton globalizations for NEqProblem" begin
    function f_diffmcp!(fvec, x)
        fvec[1] = (1 - x[1])^2 - 1.01
        return fvec
    end

    function df_diffmcp!(dfvec, x)
        dfvec[1] = -2 * (1 - x[1])
        return dfvec
    end

    function fdf_diffmcp!(Fx, Jx, x)
        f_diffmcp!(Fx, x)
        df_diffmcp!(Jx, x)
        Fx, Jx
    end

    vectorobj = NLSolvers.VectorObjective(f_diffmcp!, df_diffmcp!, fdf_diffmcp!, nothing)
    vectorprob = NEqProblem(vectorobj)

    root = 1 - sqrt(1.01)
    for method in (
        LineSearch(Newton()),
        LineSearch(Newton(), HZAW()),
        LineSearch(Newton(), Backtracking()),
        LineSearch(Newton(), Static(1)),
        TrustRegion(Newton()),
    )
        res = solve(vectorprob, [0.0], method)
        @test norm(res.info.best_residual, Inf) < 1e-8
        @test res.info.solution[1] ≈ root atol = 1e-6
    end
end
