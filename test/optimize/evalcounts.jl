# The line-search drivers must not evaluate the objective twice at the same
# point. Before the accepted-step reuse, every in-place driver re-evaluated the
# accepted point that the line search had just evaluated as its final trial, so
# these runs recorded one consecutive duplicate per iteration.
#
# A remaining duplicate can only come from the line search revisiting a trial
# point: the driver either takes the buffered gradient, evaluating nothing, or
# evaluates at a point the last trial did not use.

const evalcount_points = Vector{Vector{Float64}}()

function evalcount_fgcore!(G, x)
    G[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    G[2] = 200 * (x[2] - x[1]^2)
    (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
end
function evalcount_hess!(H, x)
    H[1, 1] = 2 - 400 * x[2] + 1200 * x[1]^2
    H[1, 2] = -400 * x[1]
    H[2, 1] = -400 * x[1]
    H[2, 2] = 200
    H
end

const evalcount_obj = ScalarObjective(
    f = x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
    g = (G, x) -> (push!(evalcount_points, copy(x)); evalcount_fgcore!(G, x); G),
    fg = (G, x) -> (push!(evalcount_points, copy(x)); (evalcount_fgcore!(G, x), G)),
    h = evalcount_hess!,
    fgh = (G, H, x) -> begin
        push!(evalcount_points, copy(x))
        (evalcount_fgcore!(G, x), G, evalcount_hess!(H, x))
    end,
)
const evalcount_prob = OptimizationProblem(evalcount_obj; inplace = true)
const evalcount_x0 = [-1.2, 1.0]

@testset "no repeated gradient evaluation" begin
    # SR1 is left out: its HZAW run revisits one trial point, which the driver
    # cannot cause and cannot remove.
    for (name, approach) in (
        ("GradientDescent/HZAW", LineSearch(GradientDescent(), HZAW())),
        ("ConjugateGradient/HZAW", LineSearch(ConjugateGradient(), HZAW())),
        ("BFGS/HZAW", LineSearch(BFGS(Inverse()), HZAW())),
        ("DFP/HZAW", LineSearch(DFP(Inverse()), HZAW())),
        ("LBFGS/HZAW", LineSearch(LBFGS(5), HZAW())),
        ("Newton/HZAW", LineSearch(Newton(), HZAW())),
        ("BFGS/Backtracking", LineSearch(BFGS(Inverse()), Backtracking())),
        ("Newton/Backtracking", LineSearch(Newton(), Backtracking())),
    )
        empty!(evalcount_points)
        res = solve(
            evalcount_prob,
            copy(evalcount_x0),
            approach,
            OptimizationOptions(maxiter = 200),
        )
        repeats = count(
            i -> evalcount_points[i] == evalcount_points[i-1],
            2:length(evalcount_points),
        )
        @test repeats == 0
        # Guard against the objective never being called at all.
        @test length(evalcount_points) >= res.info.iter
    end
end
