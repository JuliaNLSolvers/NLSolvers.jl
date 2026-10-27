# A line searcher reports whether what the objective holds is the gradient at
# the step it returned. The drivers act on that without checking, so the report
# has to be true rather than usually true: a searcher that overstates it hands
# back a gradient belonging to some other point, and the run continues with a
# quasi-Newton pair built from two different places.
#
# HZAW claims it on every successful return. Most of them hand back the trial
# just evaluated; the one that hands back a bracket endpoint evaluates it again
# first. That branch is the one worth pinning, and it is not reachable on demand,
# so this checks every return by sweeping directions and step lengths.

function currency_fgcore!(G, x)
    G[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    G[2] = 200 * (x[2] - x[1]^2)
    (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
end
const currency_obj = ScalarObjective(
    f = x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
    g = (G, x) -> (currency_fgcore!(G, x); G),
    fg = (G, x) -> (currency_fgcore!(G, x), G),
)
const currency_prob = OptimizationProblem(currency_obj; inplace = true)

@testset "the gradient belongs to the step reported" begin
    checked = 0
    reported = 0
    for x in ([-1.2, 1.0], [0.3, 0.7], [2.0, -1.0], [1.0, 1.0], [-0.5, 2.5]),
        dir in ([1.0, 0.0], [0.0, 1.0], [-1.0, -1.0], [0.7, -0.3], [-0.2, 0.9]),
        λ in (1.0, 0.1, 10.0)

        ∇fx = similar(x)
        fx, _ = NLSolvers.upto_gradient(currency_prob, ∇fx, x)
        d = dir ./ norm(dir)
        dφ0 = dot(∇fx, d)
        dφ0 < 0 || continue                      # the line search wants a descent direction
        ∇fz, z = similar(x), similar(x)
        φ = NLSolvers._lineobjective(NLSolvers.InPlace(), currency_prob, ∇fz, z, x, d, fx, dφ0)
        α, φα, success, g_current = NLSolvers.find_steplength(NLSolvers.InPlace(), HZAW(), φ, λ)
        success || continue
        checked += 1
        g_current || continue
        reported += 1

        # what the step it returned actually implies
        zα = NLSolvers.retract(currency_prob, similar(x), x, d, α)
        expected = similar(x)
        fα_expected, _ = NLSolvers.upto_gradient(currency_prob, expected, zα)
        @test ∇fz == expected
        @test φα == fα_expected
    end
    @test checked >= 30                           # the sweep really did run
    @test reported == checked                     # HZAW claims it on every success
end
