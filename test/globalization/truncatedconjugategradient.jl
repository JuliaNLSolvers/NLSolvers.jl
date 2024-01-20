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
end
