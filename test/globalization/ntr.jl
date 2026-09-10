using Test, NLSolvers, LinearAlgebra

@testset "NTR" begin
    @testset "hard case" begin
        # Orthogonal eigenbasis from a Householder reflector
        v = [1.0, 2.0, 3.0]
        Q = Matrix(1.0I, 3, 3) - 2 * (v * v') / dot(v, v)
        H = Q * Diagonal([-1.0, 1.0, 2.0]) * Q'
        # g has no component along the eigenvector of the smallest
        # eigenvalue, so the solution is in the hard case
        g = 0.3 * Q[:, 2] + 0.1 * Q[:, 3]
        Δ = 10.0
        m(p) = dot(g, p) + dot(p, H * p) / 2

        # Exact solution: λ = -λ₁ and s = -(H - λ₁I)⁺g + τq₁ with ‖s‖ = Δ
        p_perp = -(0.3 / 2 * Q[:, 2] + 0.1 / 3 * Q[:, 3])
        τ = sqrt(Δ^2 - dot(p_perp, p_perp))
        m_exact = m(p_perp + τ * Q[:, 1])

        ntr = NLSolvers.NTR()
        res = ntr(
            g,
            copy(H),
            Δ,
            zeros(3),
            NLSolvers.Newton(),
            NLSolvers.InPlace();
            abstol = 1e-10,
        )
        @test res.solved
        @test res.hard_case
        @test norm(res.p) ≈ Δ
        # default κhard = 2/10 only guarantees an approximate solution
        @test m(res.p) <= 0.98 * m_exact
        # The reported model value has to be the model in the unshifted
        # Hessian: the caller divides the actual reduction by it to get the
        # acceptance ratio. 𝓖_root sees H(λ) = H + λI, so an uncorrected
        # value is off by λ‖s‖²/2, which is enough to flip the sign of the
        # predicted reduction and accept a step that must be rejected.
        @test res.mz ≈ m(res.p)

        res_tight = ntr(
            g,
            copy(H),
            Δ,
            zeros(3),
            NLSolvers.Newton(),
            NLSolvers.InPlace();
            abstol = 1e-10,
            κeasy = 1e-10,
            κhard = 1e-9,
        )
        @test res_tight.solved
        @test res_tight.hard_case
        @test norm(res_tight.p) ≈ Δ
        @test m(res_tight.p) ≈ m_exact rtol = 1e-8
        @test res_tight.mz ≈ m(res_tight.p)
        # and it must be a predicted decrease, which is what the acceptance
        # ratio's denominator relies on
        @test res_tight.mz < 0
    end

    @testset "easy boundary case" begin
        H = [2.0 0.3; 0.3 1.0]
        g = [1.0, 1.5]
        Δ = 0.5

        ntr = NLSolvers.NTR()
        res = ntr(
            g,
            copy(H),
            Δ,
            zeros(2),
            NLSolvers.Newton(),
            NLSolvers.InPlace();
            abstol = 1e-10,
            κeasy = 1e-8,
        )
        @test res.solved
        @test !res.hard_case
        @test norm(res.p) ≈ Δ rtol = 1e-6
        # KKT: (H + λI)p = -g with λ >= 0 on the boundary
        r = H * res.p + g
        λ̂ = -dot(r, res.p) / dot(res.p, res.p)
        @test λ̂ > 0
        @test norm(r + λ̂ * res.p) < 1e-6
    end

    @testset "UniformScaling Hessian keeps its scale" begin
        g = [1.0, 2.0]
        ntr = NLSolvers.NTR()
        for (Hu, Hm) in ((2.0 * I, [2.0 0.0; 0.0 2.0]), (I, [1.0 0.0; 0.0 1.0]))
            for Δ in (0.1, 10.0)
                ru = ntr(g, Hu, Δ, zeros(2), NLSolvers.Newton(), NLSolvers.InPlace())
                rm = ntr(
                    g,
                    copy(Hm),
                    Δ,
                    zeros(2),
                    NLSolvers.Newton(),
                    NLSolvers.InPlace(),
                )
                @test ru.p ≈ rm.p
            end
        end
    end

    @testset "interior case" begin
        H = [2.0 0.3; 0.3 1.0]
        g = [1.0, 1.5]
        Δ = 10.0

        ntr = NLSolvers.NTR()
        res = ntr(
            g,
            copy(H),
            Δ,
            zeros(2),
            NLSolvers.Newton(),
            NLSolvers.InPlace();
            abstol = 1e-10,
        )
        @test res.solved
        @test res.interior
        @test res.p ≈ -(H \ g)
    end
end
