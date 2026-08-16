using Test, NLSolvers, LinearAlgebra
@testset "truncated SPR solver" begin
    m = NLSolvers.TCG()

    H = [0.3 0.0; 0.0 0.9]
    g = [0.2, 0.4]

    m(g, H, 0.7, rand(2), 1, NLSolvers.InPlace())

    # Since -H is negative definite the solution is guaranteed
    # to be at the boundary unless g = 0
    for Δ in range(0, 100; step = 0.1)
        @test norm(m(g, -H, Δ, rand(2), 1, NLSolvers.InPlace()).p, 2) ≈ Δ
    end

    # Small gradient entries
    g .= [1e-12, 1e-9]
    for Δ in range(0, 100; step = 0.1)
        @test norm(m(g, -H, Δ, rand(2), 1, NLSolvers.InPlace()).p, 2) ≈ Δ
    end

    # Mixed gradient entries
    g .= [1e12, 1e-9]
    for Δ in range(0, 100; step = 0.1)
        @test norm(m(g, -H, Δ, rand(2), 1, NLSolvers.InPlace()).p, 2) ≈ Δ
    end

    # Zero case
    g = [0.0, 0.0]
    for Δ in range(0, 100; step = 0.1)
        @test norm(m(g, -H, Δ, rand(2), 1, NLSolvers.InPlace()).p, 2) == 0
    end

    @testset "iteration cap scales with the dimension" begin
        # 50 distinct eigenvalues, so CG needs 50 iterations; a cap of 5 cannot
        # produce the interior solution
        n = 50
        Hn = Matrix(Diagonal(collect(1.0:n)))
        gn = ones(n)
        res = NLSolvers.TCG()(gn, Hn, 1e6, zeros(n), 1, NLSolvers.InPlace())
        @test res.solved
        @test res.interior
        @test norm(Hn * res.p + gn) <= 1e-8
        @test res.p ≈ -(Hn \ gn) rtol = 1e-8
    end

    @testset "truncation is reported" begin
        n = 50
        Hn = Matrix(Diagonal(collect(1.0:n)))
        gn = ones(n)
        short = NLSolvers.TCG(maxiter = 2)
        res = short(gn, Hn, 1e6, zeros(n), 1, NLSolvers.InPlace())
        @test !res.solved
        @test res.interior
        # the truncated step still decreases the model
        @test res.mz < 0
        # a per-call keyword overrides the field
        res = short(gn, Hn, 1e6, zeros(n), 1, NLSolvers.InPlace(); maxiter = n)
        @test res.solved
    end

    @testset "residual tolerance stops the interior iteration early" begin
        # one distinct eigenvalue: CG converges in a single iteration
        n = 50
        Hn = Matrix(Diagonal(fill(2.0, n)))
        gn = collect(range(0.1, 1.0; length = n))
        res = NLSolvers.TCG(maxiter = 1)(gn, Hn, 1e6, zeros(n), 1, NLSolvers.InPlace())
        @test res.solved
        @test res.interior
        @test res.p ≈ -(Hn \ gn)
    end
end
