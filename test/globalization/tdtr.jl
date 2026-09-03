using Test, NLSolvers, LinearAlgebra, StaticArrays
import Random

@testset "TDTR" begin
    solvers = (NLSolvers.TDTR(boundary = :quartic), NLSolvers.TDTR(boundary = :newton))

    function tdtr_kkt_ok(g, H, Δ, res; stat_tol = 1e-8, bound_tol = 1e-6)
        p, σ = res.p, res.λ
        Hm = H isa UniformScaling ? Matrix(H, 2, 2) : Matrix(H)
        ok = norm(p) <= Δ * (1 + 1e-10)
        ok &= norm(Hm * p + σ * p + g) <= stat_tol * max(1, norm(g))
        ok &= σ >= -1e-12
        ok &= σ + eigmin(Symmetric(Hm)) >= -1e-8 * max(1, abs(eigmin(Symmetric(Hm))))
        if res.interior
            ok &= iszero(σ)
        else
            ok &= abs(norm(p) - Δ) <= bound_tol * Δ
        end
        ok
    end

    @testset "random problems: KKT, optimality, strategy agreement" begin
        rng = Random.MersenneTwister(20260902)
        θgrid = range(0, 2π; length = 513)[1:end-1]
        for trial = 1:400
            A = randn(rng, 2, 2)
            H = Matrix(Symmetric(A + A')) + rand(rng, (-3, -1, 0, 1, 3)) * I
            g = randn(rng, 2) * 10.0^rand(rng, -3:3)
            Δ = 10.0^rand(rng, -3:2)
            results = map(solvers) do sp
                res = sp(g, copy(H), Δ, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
                @test tdtr_kkt_ok(g, H, Δ, res)
                @test res.solved
                res
            end
            # the two boundary strategies find the same step
            @test norm(results[1].p - results[2].p) <= 1e-6 * Δ

            # not worse than the model value anywhere on the boundary or at 0
            m(p) = dot(g, p) + dot(p, H * p) / 2
            mgrid = minimum(m([Δ * cos(θ), Δ * sin(θ)]) for θ in θgrid)
            for res in results
                @test res.mz <= min(mgrid, 0.0) + 1e-8 * max(1, abs(mgrid))
                @test res.mz ≈ m(res.p) atol = 1e-8 * max(1, abs(res.mz))
            end

            # agreement with NWI whenever NWI's step is feasible
            rnwi = NLSolvers.NWI()(
                g,
                copy(H),
                Δ,
                zeros(2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            if norm(rnwi.p) <= Δ * (1 + 1e-10)
                for res in results
                    @test res.mz <= rnwi.mz + 1e-6 * max(1, abs(rnwi.mz))
                end
            end
        end
    end

    @testset "hard case" begin
        # diagonal H makes the eigenbasis exact, so g̃₁ = 0 exactly
        for sp in solvers, Δ in (0.5, 1.0, 10.0)
            H = Diagonal([-2.0, 3.0])
            g = [0.0, 0.4]
            res = sp(g, H, Δ, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
            @test res.hard_case
            @test res.λ ≈ 2.0
            @test norm(res.p) ≈ Δ
            @test tdtr_kkt_ok(g, H, Δ, res)
            @test res.p[2] ≈ -0.4 / 5
        end
        # small enough radius and the constraint binds without the hard case
        for sp in solvers
            H = Diagonal([-2.0, 3.0])
            g = [0.0, 0.4]
            res = sp(g, H, 0.05, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
            @test !res.hard_case
            @test res.λ > 2.0
            @test norm(res.p) ≈ 0.05
        end
        # singular Hessian: the hard case taxonomy includes λ₁ = 0, and tiny
        # |λ₁| of either sign must not degrade the boundary solve
        for sp in solvers, λ1 in (0.0, 1e-300, -1e-300)
            H = Diagonal([λ1, 3.0])
            g = [1e-16, 0.7]
            res = sp(g, H, 0.5, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
            @test res.solved
            @test norm(res.p) ≈ 0.5
            @test tdtr_kkt_ok(g, H, 0.5, res)
        end
        for sp in solvers
            H = Diagonal([0.0, 3.0])
            g = [0.0, 0.7]
            res = sp(g, H, 0.5, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
            @test res.hard_case
            @test res.λ == 0.0
            @test norm(res.p) ≈ 0.5
            @test tdtr_kkt_ok(g, H, 0.5, res)
        end
        # rotated (numerically) hard case and the near-hard sweep
        θ = 0.7
        Q = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        H = Q * Diagonal([-2.0, 3.0]) * Q'
        for sp in solvers, δ in (0.0, 1e-6, 1e-9, 1e-14)
            g = 0.4 * Q[:, 2] + δ * Q[:, 1]
            res = sp(g, copy(H), 10.0, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
            @test tdtr_kkt_ok(g, H, 10.0, res)
            @test norm(res.p) ≈ 10.0
            @test res.λ ≈ 2.0 rtol = 1e-6
        end
    end

    @testset "zero gradient" begin
        for sp in solvers
            res = sp(
                [0.0, 0.0],
                Matrix(1.0I, 2, 2),
                1.0,
                zeros(2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            @test res.interior
            @test res.p == [0.0, 0.0]
            @test res.mz == 0.0

            res = sp(
                [0.0, 0.0],
                Matrix(-1.0I, 2, 2),
                1.0,
                zeros(2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            @test !res.interior
            @test res.hard_case
            @test norm(res.p) ≈ 1.0
            @test res.mz ≈ -0.5
        end
    end

    @testset "UniformScaling and Diagonal Hessians" begin
        g = [1.0, 2.0]
        for sp in solvers,
            (Hu, Hm) in ((2.0 * I, [2.0 0.0; 0.0 2.0]), (I, [1.0 0.0; 0.0 1.0]))

            for Δ in (0.1, 10.0)
                ru = sp(g, Hu, Δ, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
                rm = sp(g, copy(Hm), Δ, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
                @test ru.p ≈ rm.p
                @test ru.mz ≈ rm.mz
            end
        end
        for sp in solvers
            rd = sp(
                g,
                Diagonal([1.0, 2.0]),
                0.5,
                zeros(2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            rm = sp(
                g,
                [1.0 0.0; 0.0 2.0],
                0.5,
                zeros(2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            @test rd.p ≈ rm.p
            @test tdtr_kkt_ok(g, Diagonal([1.0, 2.0]), 0.5, rd)
        end
    end

    @testset "interior/boundary flip" begin
        H = [2.0 0.3; 0.3 1.0]
        g = [1.0, 1.5]
        pN = -(H \ g)
        for sp in solvers
            rin = sp(
                g,
                copy(H),
                norm(pN) * 1.01,
                zeros(2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            @test rin.interior
            @test iszero(rin.λ)
            @test rin.p ≈ pN
            rout = sp(
                g,
                copy(H),
                norm(pN) * 0.99,
                zeros(2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            @test !rout.interior
            @test rout.λ > 0
            @test norm(rout.p) ≈ norm(pN) * 0.99
        end
    end

    @testset "scale invariance" begin
        H = [2.0 1.0; 1.0 -1.0]
        g = [1.0, 2.0]
        for sp in solvers, t in (1e-8, 1e8)
            r1 = sp(g, copy(H), 2.0, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
            rt = sp(t * g, t * H, 2.0, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
            @test r1.p ≈ rt.p rtol = 1e-8
            @test rt.λ ≈ t * r1.λ rtol = 1e-8
        end
    end

    @testset "generic number types" begin
        for sp in solvers
            H32 = Float32[2 1; 1 -1]
            g32 = Float32[1, 2]
            r32 = sp(
                g32,
                copy(H32),
                2.0f0,
                zeros(Float32, 2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            @test eltype(r32.p) == Float32
            @test norm(H32 * r32.p + r32.λ * r32.p + g32) <= 5e-6

            Hb = BigFloat[2 1; 1 -1]
            gb = BigFloat[1, 2]
            rb = sp(
                gb,
                copy(Hb),
                big"2.0",
                zeros(BigFloat, 2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            @test eltype(rb.p) == BigFloat
            @test norm(Hb * rb.p + rb.λ * rb.p + gb) <= big"1e-50"
        end
        # integer-typed inputs promote
        for sp in solvers
            ri = sp(
                [1.0, 2.0],
                [2 0; 0 3],
                1,
                zeros(2),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            @test tdtr_kkt_ok([1.0, 2.0], [2.0 0.0; 0.0 3.0], 1.0, ri)
        end
    end

    @testset "static arrays out of place" begin
        H = @SMatrix [2.0 1.0; 1.0 -1.0]
        g = @SVector [1.0, 2.0]
        p0 = @SVector [0.0, 0.0]
        for sp in solvers
            res = sp(g, H, 2.0, p0, NLSolvers.Newton(), NLSolvers.OutOfPlace())
            @test res.p isa SVector{2,Float64}
            @test tdtr_kkt_ok(g, Matrix(H), 2.0, res)
        end
    end

    @testset "Direct and Inverse model forms agree" begin
        H = [2.0 0.4; 0.4 1.0]
        g = [1.0, -2.0]
        for sp in solvers
            rd = sp(g, copy(H), 0.5, zeros(2), BFGS(Direct()), NLSolvers.InPlace())
            ri = sp(g, inv(H), 0.5, zeros(2), BFGS(Inverse()), NLSolvers.InPlace())
            @test rd.p ≈ ri.p
            @test rd.mz ≈ ri.mz
        end
    end

    @testset "solver fields and keyword overrides" begin
        H = [2.0 0.3; 0.3 1.0]
        g = [1.0, 1.5]
        short = NLSolvers.TDTR(boundary = :newton, maxiter = 1, abstol = 1e-14)
        res = short(g, copy(H), 0.5, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
        @test !res.solved
        res = short(
            g,
            copy(H),
            0.5,
            zeros(2),
            NLSolvers.Newton(),
            NLSolvers.InPlace();
            maxiter = 50,
        )
        @test res.solved
    end

    @testset "argument errors" begin
        @test_throws ArgumentError NLSolvers.TDTR(boundary = :cubic)
        for sp in solvers
            @test_throws ArgumentError sp(
                ones(3),
                Matrix(1.0I, 3, 3),
                1.0,
                zeros(3),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
        end
    end
end
