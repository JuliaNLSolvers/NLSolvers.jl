using Test, NLSolvers, LinearAlgebra

@testset "Dogleg subproblem" begin
    H = [2.0 0.0; 0.0 10.0]
    g = [1.0, 1.0]
    d_c = -g * norm(g)^2 / dot(g, H * g)
    p_n = -(H \ g)
    m(p) = dot(g, p) + dot(p, H * p) / 2

    dl = NLSolvers.Dogleg()
    scheme = NLSolvers.Newton()

    @testset "boundary crossing" begin
        for Δ in range(norm(d_c) + 1e-3, norm(p_n) - 1e-3; length = 25)
            res = dl(g, copy(H), Δ, zeros(2), scheme, NLSolvers.InPlace())
            @test norm(res.p) ≈ Δ
            # reference: the positive root of ||d_c + t*(p_n - d_c)|| = Δ
            a = dot(p_n - d_c, p_n - d_c)
            b = 2 * dot(d_c, p_n - d_c)
            c = dot(d_c, d_c) - Δ^2
            tref = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
            @test 0 < tref < 1
            @test res.p ≈ d_c + tref * (p_n - d_c)
            @test res.mz ≈ m(res.p)
            @test m(res.p) < m(d_c)
        end
    end

    @testset "scaled Cauchy step" begin
        Δ = norm(d_c) / 2
        res = dl(g, copy(H), Δ, zeros(2), scheme, NLSolvers.InPlace())
        @test res.p ≈ (Δ / norm(d_c)) * d_c
        @test norm(res.p) ≈ Δ
        @test !res.interior
    end

    @testset "interior Newton step" begin
        Δ = 2 * norm(p_n)
        res = dl(g, copy(H), Δ, zeros(2), scheme, NLSolvers.InPlace())
        @test res.p ≈ p_n
        @test res.interior
    end
end
