using Test, NLSolvers, LinearAlgebra

@testset "NWI" begin
    @testset "UniformScaling Hessian keeps its scale" begin
        g = [1.0, 2.0]
        for (Hu, Hm) in ((2.0 * I, [2.0 0.0; 0.0 2.0]), (I, [1.0 0.0; 0.0 1.0]))
            for Δ in (0.1, 10.0)
                ru = NLSolvers.NWI()(
                    g,
                    Hu,
                    Δ,
                    zeros(2),
                    NLSolvers.Newton(),
                    NLSolvers.InPlace(),
                )
                rm = NLSolvers.NWI()(
                    g,
                    copy(Hm),
                    Δ,
                    zeros(2),
                    NLSolvers.Newton(),
                    NLSolvers.InPlace(),
                )
                @test ru.p ≈ rm.p
                @test ru.mz ≈ rm.mz
            end
        end
    end

    @testset "boundary solutions near the hard case" begin
        # Orthogonal eigenbasis from a Householder reflector
        v = [1.0, 2.0, 3.0]
        Q = Matrix(1.0I, 3, 3) - 2 * (v * v') / dot(v, v)
        H = Q * Diagonal([-1.0, 1.0, 2.0]) * Q'
        # small but nonzero component along q₁ puts λ* just above -λ₁,
        # so the shifted matrix is barely positive definite at the root
        for δ in (1e-6, 1e-9), Δ in (0.5, 2.0, 10.0)
            g = δ * Q[:, 1] + 0.3 * Q[:, 2] + 0.1 * Q[:, 3]
            res = NLSolvers.NWI()(
                g,
                copy(H),
                Δ,
                zeros(3),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            # the λ-scale stopping rule only controls the boundary distance
            # loosely when the secular root is steep, hence the tolerance
            @test norm(res.p) ≈ Δ rtol = 1e-3
            # KKT stationarity on the boundary: (H + λI)p = -g with λ ≥ -λ₁
            @test res.λ ≥ 1 - 1e-6
            @test norm(H * res.p + res.λ * res.p + g) ≤ 1e-4 * norm(g)
        end
    end

    @testset "badly scaled Hessian recovers from failed factorizations" begin
        # eigen's λ₁ and Cholesky's positive-definiteness judgment disagree
        # by more than the sqrt(eps) shift here, so the Newton iteration
        # hits failed factorizations at λ ≥ λ_lb. Before the failure
        # branch stepped into the bracket, these returned p = 0 with
        # solved = false after spinning maxiter times on the same λ.
        v = [1.0, 2.0, 3.0]
        Q = Matrix(1.0I, 3, 3) - 2 * (v * v') / dot(v, v)
        for scale in (1e8, 1e10), Δ in (1e-4, 1.0)
            H = Q * Diagonal([-scale, 1.0, 2.0]) * Q'
            g = 0.3 * Q[:, 1] + 0.3 * Q[:, 2] + 0.1 * Q[:, 3]
            res = NLSolvers.NWI()(
                g,
                copy(H),
                Δ,
                zeros(3),
                NLSolvers.Newton(),
                NLSolvers.InPlace(),
            )
            @test norm(res.p) ≈ Δ rtol = 1e-5
            @test norm(H * res.p + res.λ * res.p + g) ≤ 1e-4 * norm(g)
        end
    end
end
