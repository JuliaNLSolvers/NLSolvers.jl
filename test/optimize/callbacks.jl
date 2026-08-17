using NLSolvers
using LinearAlgebra
using Test
isdefined(Main, :TestProblems) || include(joinpath(@__DIR__, "testproblems.jl"))

@testset "Callbacks" begin
    prob = OptimizationProblem(TestProblems.rosenbrock.inplace)
    x0 = [-3.0, -4.0]

    @testset "Default callback is nothing" begin
        @test OptimizationOptions().callback === nothing
    end

    @testset "Callback receives (iter, time, state)" begin
        info_seen = Ref{NamedTuple}()
        spy = info -> begin
            info_seen[] = info
            info.iter >= 1
        end
        solve(prob, copy(x0), LineSearch(BFGS()), OptimizationOptions(callback = spy))
        info = info_seen[]
        @test haskey(info, :iter)
        @test haskey(info, :time)
        @test haskey(info, :state)
        @test info.iter == 1
        @test info.time >= 0.0
    end

    @testset "Early stop callback" begin
        stop_at_5 = info -> info.iter >= 5

        res = solve(
            prob,
            copy(x0),
            LineSearch(BFGS()),
            OptimizationOptions(callback = stop_at_5),
        )
        @test res.info.iter == 5

        res = solve(
            prob,
            copy(x0),
            LineSearch(LBFGS()),
            OptimizationOptions(callback = stop_at_5),
        )
        @test res.info.iter == 5

        res = solve(
            prob,
            copy(x0),
            ConjugateGradient(),
            OptimizationOptions(callback = stop_at_5),
        )
        @test res.info.iter <= 5

        res = solve(
            prob,
            copy(x0),
            TrustRegion(Newton(), NWI()),
            OptimizationOptions(callback = stop_at_5),
        )
        @test res.info.iter <= 5

        res = solve(prob, copy(x0), NelderMead(), OptimizationOptions(callback = stop_at_5))
        @test res.info.iter == 5
    end

    @testset "No-op callback doesn't change results" begin
        noop = info -> false
        opts_noop = OptimizationOptions(callback = noop, maxiter = 50)
        opts_none = OptimizationOptions(maxiter = 50)

        res_noop = solve(prob, copy(x0), LineSearch(BFGS()), opts_noop)
        res_none = solve(prob, copy(x0), LineSearch(BFGS()), opts_none)
        @test res_noop.info.iter == res_none.info.iter
        @test res_noop.info.minimum ≈ res_none.info.minimum
    end

    @testset "Line-search solver state has objvars fields" begin
        info_seen = Ref{NamedTuple}()
        spy = info -> begin
            info_seen[] = info
            info.iter >= 1
        end
        solve(prob, copy(x0), LineSearch(BFGS()), OptimizationOptions(callback = spy))
        state = info_seen[].state
        @test haskey(state, :x)
        @test haskey(state, :fx)
        @test haskey(state, :∇fx)
        @test haskey(state, :z)
        @test haskey(state, :fz)
        @test haskey(state, :∇fz)
        @test haskey(state, :B)
    end

    @testset "Trust region state has Δ and rejected" begin
        info_seen = Ref{NamedTuple}()
        spy = info -> begin
            info_seen[] = info
            info.iter >= 1
        end
        solve(
            prob,
            copy(x0),
            TrustRegion(Newton(), NWI()),
            OptimizationOptions(callback = spy),
        )
        state = info_seen[].state
        @test haskey(state, :Δ)
        @test haskey(state, :rejected)
        @test haskey(state, :z)
        @test haskey(state, :fz)
    end

    @testset "Simulated Annealing state has temperature" begin
        info_seen = Ref{NamedTuple}()
        spy = info -> begin
            info_seen[] = info
            info.iter >= 1
        end
        solve(prob, copy(x0), SimulatedAnnealing(), OptimizationOptions(callback = spy))
        state = info_seen[].state
        @test haskey(state, :temperature)
        @test haskey(state, :x_best)
        @test haskey(state, :f_best)
        @test haskey(state, :x_now)
        @test haskey(state, :f_now)
    end

    @testset "Nelder-Mead state has simplex fields" begin
        info_seen = Ref{NamedTuple}()
        spy = info -> begin
            info_seen[] = info
            info.iter >= 1
        end
        solve(prob, copy(x0), NelderMead(), OptimizationOptions(callback = spy))
        state = info_seen[].state
        @test haskey(state, :simplex_vector)
        @test haskey(state, :simplex_value)
        @test haskey(state, :nm_obj)
        @test haskey(state, :x_centroid)
    end

    @testset "Callback collecting trace" begin
        trace = Float64[]
        collector = info -> begin
            push!(trace, info.state.fz)
            false
        end
        solve(
            prob,
            copy(x0),
            LineSearch(BFGS()),
            OptimizationOptions(callback = collector, maxiter = 20, g_abstol = 0.0),
        )
        @test length(trace) == 20
        @test trace[end] < trace[1]
    end

    @testset "Callback storing solutions (with copy)" begin
        history_x = Vector{Vector{Float64}}()
        history_f = Float64[]
        collector = info -> begin
            push!(history_x, copy(info.state.z))
            push!(history_f, info.state.fz)
            false
        end
        solve(
            prob,
            copy(x0),
            LineSearch(BFGS()),
            OptimizationOptions(callback = collector, maxiter = 10, g_abstol = 0.0),
        )
        @test length(history_x) == 10
        @test length(history_f) == 10
        # Later iterates should be closer to minimum [1, 1]
        @test norm(history_x[end] .- [1.0, 1.0]) < norm(history_x[1] .- [1.0, 1.0])
    end
end
