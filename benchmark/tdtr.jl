# Benchmark of the 2x2 trust-region sub-problem solvers.
#
# Run in an environment with NLSolvers (this checkout), BenchmarkTools, and
# StaticArrays, e.g.
#
#   julia --project=<env> benchmark/tdtr.jl
#
# Solvers that shift the Hessian diagonal in place (NWI, NTR) get a fresh copy
# of H inside the timed region; TDTR never mutates its inputs, so it is timed
# on the shared matrices directly. Dogleg is inexact and only included as a
# reference point in the regimes where its positive definite requirement holds.

using NLSolvers, BenchmarkTools, StaticArrays, LinearAlgebra
import Random

const RNG = Random.MersenneTwister(20260902)
const NPROB = 100

rot(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

function problem_batch(regime)
    batch = Vector{Tuple{Matrix{Float64},Vector{Float64},Float64}}()
    for _ = 1:NPROB
        Q = rot(2π * rand(RNG))
        if regime === :pd_boundary
            H = Q * Diagonal([0.5 + 4.5 * rand(RNG), 0.5 + 4.5 * rand(RNG)]) * Q'
            g = randn(RNG, 2)
            Δ = 0.3 * norm(H \ g)
        elseif regime === :indefinite
            H = Q * Diagonal([-3 * rand(RNG) - 0.1, 2 * rand(RNG) + 0.1]) * Q'
            g = randn(RNG, 2)
            Δ = 10.0^(2 * rand(RNG) - 1)
        elseif regime === :near_hard
            D = Diagonal([-2 * rand(RNG) - 0.1, 3 * rand(RNG) + 0.1])
            H = Q * D * Q'
            g = 0.5 * Q[:, 2] + 1e-10 * Q[:, 1]
            Δ = 10.0
        elseif regime === :interior
            H = Q * Diagonal([0.5 + 4.5 * rand(RNG), 0.5 + 4.5 * rand(RNG)]) * Q'
            g = randn(RNG, 2)
            Δ = 2.0 * norm(H \ g)
        end
        push!(batch, (H, g, Δ))
    end
    batch
end

function run_batch(sp, batch, p, scheme, copyH::Bool)
    acc = 0.0
    @inbounds for (H, g, Δ) in batch
        Hs = copyH ? copy(H) : H
        acc += sp(g, Hs, Δ, p, scheme, NLSolvers.InPlace()).mz
    end
    acc
end

function main()
    scheme = NLSolvers.Newton()
    solvers = (
        ("TDTR(:quartic)", NLSolvers.TDTR(boundary = :quartic), false),
        ("TDTR(:newton)", NLSolvers.TDTR(boundary = :newton), false),
        ("NWI", NLSolvers.NWI(), true),
        ("NTR", NLSolvers.NTR(), true),
        ("Dogleg", NLSolvers.Dogleg(), true),
    )
    println("median ns/solve over $NPROB problems per regime (machine load matters):")
    for regime in (:pd_boundary, :indefinite, :near_hard, :interior)
        batch = problem_batch(regime)
        println("regime: $regime")
        for (name, sp, copyH) in solvers
            if sp isa Dogleg && regime in (:indefinite, :near_hard)
                continue
            end
            p = zeros(2)
            b = @benchmark run_batch($sp, $batch, $p, $scheme, $copyH) evals = 1
            run_batch(sp, batch, p, scheme, copyH)
            al = @allocated run_batch(sp, batch, p, scheme, copyH)
            println(
                "  ",
                rpad(name, 16),
                lpad(round(Int, median(b.times) / NPROB), 8),
                " ns/solve  ",
                lpad(round(Int, al / NPROB), 8),
                " bytes/solve (incl. any H copy)",
            )
        end
    end

    # static arrays, out of place: allocation-free exact solves
    H = @SMatrix [2.0 1.0; 1.0 -1.0]
    g = @SVector [1.0, 2.0]
    p0 = @SVector [0.0, 0.0]
    sp = NLSolvers.TDTR()
    b = @benchmark $sp($g, $H, 2.0, $p0, $scheme, NLSolvers.OutOfPlace())
    solve1(sp, g, H, p0, scheme) = sp(g, H, 2.0, p0, scheme, NLSolvers.OutOfPlace()).mz
    solve1(sp, g, H, p0, scheme)
    al = @allocated solve1(sp, g, H, p0, scheme)
    println("static TDTR single solve: $(round(Int, median(b.times))) ns, $al bytes")
end

main()
