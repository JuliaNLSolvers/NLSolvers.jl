# solve must be type stable. The drivers branch per iteration (on line search
# success, and on whether the accepted step already has a gradient), and a
# branch whose arms return different types turns every loop variable into a
# union: correct, but it allocates per iteration and is invisible to tests that
# only check the minimum. One such branch cost 8 kB per solve on Newton with
# Backtracking, a combination that reaches neither arm of interest.

function inference_fgcore!(G, x)
    G[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    G[2] = 200 * (x[2] - x[1]^2)
    (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
end
function inference_hess!(H, x)
    H[1, 1] = 2 - 400 * x[2] + 1200 * x[1]^2
    H[1, 2] = -400 * x[1]
    H[2, 1] = -400 * x[1]
    H[2, 2] = 200
    H
end
const inference_obj = ScalarObjective(
    f = x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
    g = (G, x) -> (inference_fgcore!(G, x); G),
    fg = (G, x) -> (inference_fgcore!(G, x), G),
    h = inference_hess!,
    fgh = (G, H, x) -> (inference_fgcore!(G, x), G, inference_hess!(H, x)),
)
const inference_prob = OptimizationProblem(inference_obj; inplace = true)
const inference_prob_oop = OptimizationProblem(inference_obj; inplace = false)
const inference_x0 = [-1.2, 1.0]

@testset "inference" begin
    @testset "line search, in place" begin
        for scheme in (
            GradientDescent(),
            ConjugateGradient(),
            BFGS(Inverse()),
            BFGS(Direct()),
            DFP(Inverse()),
            SR1(Inverse()),
            DBFGS(Inverse()),
            LBFGS(5),
            Newton(),
        )
            for linesearcher in (HZAW(), Backtracking(), Static(0.01))
                @test (@inferred solve(
                    inference_prob,
                    copy(inference_x0),
                    LineSearch(scheme, linesearcher),
                    OptimizationOptions(maxiter = 20),
                )) isa NLSolvers.ConvergenceInfo
            end
        end
    end

    @testset "line search, out of place" begin
        for scheme in (BFGS(Inverse()), Newton())
            for linesearcher in (HZAW(), Backtracking())
                @test (@inferred solve(
                    inference_prob_oop,
                    copy(inference_x0),
                    LineSearch(scheme, linesearcher),
                    OptimizationOptions(maxiter = 20),
                )) isa NLSolvers.ConvergenceInfo
            end
        end
        # The out-of-place ConjugateGradient driver infers Any for the solution
        # and the minimum, on 1.10 and on 1.12.
        @test_broken (@inferred solve(
            inference_prob_oop,
            copy(inference_x0),
            LineSearch(ConjugateGradient(), HZAW()),
            OptimizationOptions(maxiter = 20),
        )) isa NLSolvers.ConvergenceInfo
    end

    @testset "trust region" begin
        for scheme in (Newton(), BFGS(Direct()), SR1(), DBFGS())
            for subsolver in (NWI(), NTR())
                @test (@inferred solve(
                    inference_prob,
                    copy(inference_x0),
                    TrustRegion(scheme, subsolver),
                    OptimizationOptions(maxiter = 20),
                )) isa NLSolvers.ConvergenceInfo
            end
        end
    end
end
