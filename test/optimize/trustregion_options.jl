using Test, NLSolvers, LinearAlgebra
isdefined(Main, :TestProblems) || include(joinpath(@__DIR__, "testproblems.jl"))

_tro_prob() = OptimizationProblem(TestProblems.rosenbrock.inplace; inplace = true)

@testset "trust region options" begin
    @testset "BTR defaults match the historical constants" begin
        b = NLSolvers.BTR()
        @test b.Δmin === nothing
        @test b.η == 1e-4
        @test b.Δ0 == 20.0
        @test b.t2 == 1 / 4
        @test b.t3 == 1 / 4
        @test b.t4 == 1 / 2
        @test b.λ34 == 0.0
        @test b.γ == 2.5
        @test b.λγ == 1 / 2
        @test b.Δmax == 1e5
        @test b.σ == 1 / 4
    end

    @testset "update_trust_region reads BTR" begin
        p = [3.0, 4.0] # norm 5
        boundary = (Δ = 8.0, interior = false)
        interior = (Δ = 8.0, interior = true)
        b = NLSolvers.BTR()
        # rejected boundary step: t4 * Δ
        @test NLSolvers.update_trust_region(b, boundary, -1.0, false, p) == 4.0
        # rejected interior step: σ * ||p||
        @test NLSolvers.update_trust_region(b, interior, -1.0, false, p) == 1.25
        # distrusted acceptance (R < t2): t4 * Δ
        @test NLSolvers.update_trust_region(b, boundary, 0.1, true, p) == 4.0
        # trusted acceptance: growth λγ*Δ + (1 - λγ)*γ*Δ
        @test NLSolvers.update_trust_region(b, boundary, 0.9, true, p) == 14.0
        # the interval form: λ34 = 1 shrinks to t3 * ||p||
        b34 = NLSolvers.BTR(lambda34 = 1.0)
        @test NLSolvers.update_trust_region(b34, boundary, -1.0, false, p) == 1.25
        # growth cap
        bcap = NLSolvers.BTR(deltamax = 10.0)
        @test NLSolvers.update_trust_region(bcap, boundary, 0.9, true, p) == 10.0
    end

    @testset "delta0 and the initial_Δ override" begin
        approach = TrustRegion(BFGS(Direct()), Dogleg(), NLSolvers.BTR(delta0 = 1e6))
        res_field =
            solve(_tro_prob(), ([-1.2, 1.0], nothing), approach, OptimizationOptions())
        res_kw = solve(
            _tro_prob(),
            ([-1.2, 1.0], nothing),
            TrustRegion(BFGS(Direct()), Dogleg()),
            OptimizationOptions();
            initial_Δ = 1e6,
        )
        @test res_field.info.minimum == res_kw.info.minimum
        @test res_field.info.iter == res_kw.info.iter
        @test res_field.info.minimum < 1e-10
    end

    @testset "deltamax bounds the radius" begin
        seen = Float64[]
        cb = info -> (push!(seen, info.state.Δ); false)
        approach = TrustRegion(BFGS(Direct()), Dogleg(), NLSolvers.BTR(deltamax = 30.0))
        res = solve(
            _tro_prob(),
            ([-1.2, 1.0], nothing),
            approach,
            OptimizationOptions(callback = cb),
        )
        @test !isempty(seen)
        @test maximum(seen) <= 30.0
        @test res.info.minimum < 1e-10
    end

    @testset "deltamin stopping rule" begin
        # unit level: nothing resolves to a rounding-level floor, 0 disables
        objvars = (
            x = [0.0, 0.0],
            fx = 10.0,
            ∇fx = [1.0, 1.0],
            z = [1.0, 1.0],
            fz = 9.0,
            ∇fz = [1.0, 1.0],
            B = nothing,
            Pg = nothing,
        )
        opts = OptimizationOptions()
        auto = TrustRegion(NLSolvers.Newton(), NLSolvers.NTR())
        # the auto floor is the iterate's resolution: eps(T) * max(1, ||z||)
        tiny = eps(Float64) / 4
        @test any(NLSolvers.converged(auto, objvars, 1.0, opts, true, tiny))
        # a deep but representable radius must not stop the solve
        @test !any(NLSolvers.converged(auto, objvars, 1.0, opts, true, 1e-9))
        off = TrustRegion(NLSolvers.Newton(), NLSolvers.NTR(); deltamin = 0)
        @test !any(NLSolvers.converged(off, objvars, 1.0, opts, true, tiny))
        num = TrustRegion(NLSolvers.Newton(), NLSolvers.NTR(); deltamin = 1.0)
        @test any(NLSolvers.converged(num, objvars, 1.0, opts, true, 0.5))
    end

    @testset "eta is honored" begin
        # strict threshold still converges, just with more rejections
        approach = TrustRegion(BFGS(Direct()), Dogleg(); eta = 0.3)
        res = solve(_tro_prob(), ([-1.2, 1.0], nothing), approach, OptimizationOptions())
        @test res.info.minimum < 1e-10
        @test NLSolvers.tr_acceptance(0.2, 1.0, 0.3)[2] == false
        @test NLSolvers.tr_acceptance(0.2, 1.0, 1e-4)[2] == true
    end

    @testset "sub-problem solver fields" begin
        H = [2.0 0.3; 0.3 1.0]
        g = [1.0, 1.5]
        # a one-iteration budget cannot satisfy the boundary tolerance
        ntr_short = NLSolvers.NTR(maxiter = 1, κeasy = 1e-10, κhard = 1e-10)
        res = ntr_short(g, copy(H), 0.5, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
        @test !res.solved
        # per-call keyword still overrides the field
        res = ntr_short(
            g,
            copy(H),
            0.5,
            zeros(2),
            NLSolvers.Newton(),
            NLSolvers.InPlace();
            maxiter = 50,
            κeasy = 1e-8,
        )
        @test res.solved
        # driver-level: a solve with a configured subsolver still converges.
        # Dogleg requires a positive definite model, so it gets BFGS; the
        # nearly-exact solvers get Newton.
        for (scheme, sp) in (
            (NLSolvers.Newton(), NLSolvers.NTR(abstol = 1e-8)),
            (NLSolvers.Newton(), NLSolvers.NWI(abstol = 1e-8)),
            (NLSolvers.Newton(), NLSolvers.TCG(abstol = 1e-8)),
            (BFGS(Direct()), NLSolvers.Dogleg(abstol = 1e-8)),
        )
            res = solve(
                _tro_prob(),
                ([-1.2, 1.0], nothing),
                TrustRegion(scheme, sp),
                OptimizationOptions(),
            )
            @test res.info.minimum < 1e-10
        end
    end
end
