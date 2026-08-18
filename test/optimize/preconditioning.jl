using NLSolvers, SparseArrays, LinearAlgebra
using Test
isdefined(Main, :TestProblems) || include(joinpath(@__DIR__, "testproblems.jl"))
@testset "optimization preconditioning" begin
    debug_print = true
    for N in (10, 50, 100, 150, 250)
        if debug_print
            println("N = ", N)
        end
        initial_x = TestProblems.laplacian.x0(N)
        Plap = TestProblems.laplacian.precond(initial_x)
        ID = nothing
        iter = []
        mino = []
        for optimizer in (
            p -> LineSearch(GradientDescent(Direct(), p)),
            p -> LineSearch(ConjugateGradient(HZ(), p), HZAW()),
            p -> LineSearch(ConjugateGradient(HZ(), p), Backtracking()),
            p -> LineSearch(LBFGS(Inverse(), NLSolvers.TwoLoop(), 5, p)),
            p -> LineSearch(LBFGS(Inverse(), NLSolvers.TwoLoop(), 5, p), HZAW()),
        )
            for (P, wwo) in zip(
                (
                    nothing,
                    (x, P = nothing) -> inv(Array(TestProblems.laplacian.precond(N))),
                ),
                (" WITHOUT", " WITH"),
            )
                if debug_print
                    println(summary(optimizer(P)))
                end
                results = solve(
                    TestProblems.laplacian.problem,
                    copy(initial_x),
                    optimizer(P),
                    OptimizationOptions(g_abstol = 1e-6, f_abstol = 0.0),
                )
                push!(mino, results.info.minimum)
                push!(iter, results.info.iter)
            end
            if debug_print
                println("Iterations without precon: $(iter[end-1])")
                println("Iterations with    precon: $(iter[end])")
                println("Minimum without precon:    $(mino[end-1])")
                println("Minimum with    precon:    $(mino[end])")
                println()
            end
            if !(
                N == 10 &&
                optimizer(nothing).linesearcher isa HZAW &&
                optimizer(nothing).scheme isa LBFGS
            )
                @test iter[end-1] >= iter[end]
            end
        end
    end
end
