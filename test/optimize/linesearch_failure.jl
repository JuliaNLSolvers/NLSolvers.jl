# A failed line search must leave the solver at its last accepted point, must not
# be reported as convergence, and must not be retried once the retry from the
# reset approximation has failed as well.
#
# In place, the line objective writes each trial point into z and its gradient
# into ∇fz. The drivers used to keep that trial state after a failed search while
# fz still held f(x), so the returned minimum was not f(solution). Out of place, z
# stayed at x, and the zero step passed the x test under the default x_abstol = 0.

lsfail_f(x) = sum(abs2, abs2.(x))
lsfail_g!(G, x) = (G .= 4 .* x .^ 3; G)
# From here every trial step overflows, so every line search fails.
const lsfail_x0 = [1e30, 1e30]

@testset "failed line search" begin
    # out-of-place L-BFGS does not run at all, so it is left out
    for (inplace, approach) in (
        (true, LineSearch(BFGS(), HZAW())),
        (true, LineSearch(BFGS(), Backtracking())),
        (true, LineSearch(LBFGS(5), HZAW())),
        (false, LineSearch(BFGS(), HZAW())),
        (false, LineSearch(BFGS(), Backtracking())),
    )
        obj =
            inplace ?
            ScalarObjective(
                f = lsfail_f,
                g = lsfail_g!,
                fg = (G, x) -> (lsfail_f(x), lsfail_g!(G, x)),
            ) :
            ScalarObjective(
                f = lsfail_f,
                g = (G, x) -> lsfail_g!(similar(x), x),
                fg = (G, x) -> (lsfail_f(x), lsfail_g!(similar(x), x)),
            )
        res = solve(
            OptimizationProblem(obj; inplace),
            copy(lsfail_x0),
            approach,
            OptimizationOptions(),
        )
        info = res.info
        @test info.solution == lsfail_x0
        @test info.minimum == lsfail_f(info.solution)
        @test !info.ls_success
        @test info.iter == 2
        @test occursin("line search failed", sprint(show, res))
    end
end
