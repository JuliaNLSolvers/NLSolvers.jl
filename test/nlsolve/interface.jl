using NLSolvers
using LinearAlgebra
using SparseDiffTools
using SparseArrays
using IterativeSolvers
using DoubleFloats
using ForwardDiff
using Test
using IterativeSolvers
function inexact_linsolve(x0, JvOp, Fx, ηₖ)
    krylov_iter = IterativeSolvers.gmres_iterable!(x0, JvOp, Fx; maxiter = 50)
    res = copy(Fx)
    rhs = ηₖ * norm(Fx, 2)
    for item in krylov_iter
        res = krylov_iter.residual.current
        if res <= rhs
            break
        end
    end
    return x0, res
end
@testset "nlequations interface" begin
    #
    # Have Anderosn use FixedPointProblem
    #
    #



    function theta(x)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    function F_fletcher_powell!(Fx, x)
        if (x[1]^2 + x[2]^2 == 0)
            dtdx1 = 0
            dtdx2 = 0
        else
            dtdx1 = -x[2] / (2 * pi * (x[1]^2 + x[2]^2))
            dtdx2 = x[1] / (2 * pi * (x[1]^2 + x[2]^2))
        end
        Fx[1] =
            -2000.0 * (x[3] - 10.0 * theta(x)) * dtdx1 +
            200.0 * (sqrt(x[1]^2 + x[2]^2) - 1) * x[1] / sqrt(x[1]^2 + x[2]^2)
        Fx[2] =
            -2000.0 * (x[3] - 10.0 * theta(x)) * dtdx2 +
            200.0 * (sqrt(x[1]^2 + x[2]^2) - 1) * x[2] / sqrt(x[1]^2 + x[2]^2)
        Fx[3] = 200.0 * (x[3] - 10.0 * theta(x)) + 2.0 * x[3]
        Fx
    end

    function F_jacobian_fletcher_powell!(Fx, Jx, x)
        ForwardDiff.jacobian!(Jx, F_fletcher_powell!, Fx, x)
        Fx, Jx
    end

    jv = JacVec(F_fletcher_powell!, rand(3); autodiff = false)
    function jvop(x)
        jv.x .= x
        jv
    end
    prob_obj = NLSolvers.VectorObjective(
        F_fletcher_powell!,
        nothing,
        F_jacobian_fletcher_powell!,
        jvop,
    )

    prob = NEqProblem(prob_obj)

    x0 = [-1.0, 0.0, 0.0]
    res = solve(prob, x0, LineSearch(Newton(), Backtracking()))
    @test norm(res.info.best_residual, Inf) < 1e-12
    @test norm(solution(res) .- [1.0, 0.0, 0.0]) < 1e-8

    #x0 = [-1.0, 0.0, 0.0]
    #res = solve(prob, x0, Anderson(), NEqOptions())
    #@test norm(res.info.best_residual, Inf) < 1e-12

    x0 = [-1.0, 0.0, 0.0]
    res = solve(prob, x0, TrustRegion(Newton()), NEqOptions())
    @test norm(res.info.best_residual, Inf) < 1e-12

    #x0 = [-1.0, 0.0, 0.0]
    #res = solve(prob, x0, TrustRegion(DBFGS()), NEqOptions())
    #@test norm(res.info.best_residual, Inf) < 1e-12

    #x0 = [-1.0, 0.0, 0.0]
    #res = solve(prob, x0, TrustRegion(BFGS()), NEqOptions())
    #@test norm(res.info.best_residual, Inf) < 1e-12

    x0 = [-1.0, 0.0, 0.0]
    res = solve(prob, x0, TrustRegion(SR1()), NEqOptions())
    @test norm(res.info.best_residual, Inf) < 1e-8
    @test norm(solution(res) .- [1.0, 0.0, 0.0]) < 1e-8

    x0 = [-1.0, 0.0, 0.0]
    state = (z = copy(x0), d = copy(x0), Fx = copy(x0), Jx = zeros(3, 3))
    res = solve(prob, x0, LineSearch(Newton(), Backtracking()), NEqOptions(), state)
    @test norm(res.info.best_residual, Inf) < 1e-12

    #x0 = [-1.0, 0.0, 0.0]
    #res = solve(prob, x0, InexactNewton(FixedForceTerm(1e-3), 1e-3, 300), NEqOptions())
    #@test norm(res.info.best_residual, Inf) < 1e-12

    #x0 = [-1.0, 0.0, 0.0]
    #@show solve(prob, x0, InexactNewton(EisenstatWalkerA(), 1e-8, 300), NEqOptions())
    #@test norm(res.info.best_residual, Inf) < 1e-12

    #x0 = [-1.0, 0.0, 0.0]
    #@show solve(prob, x0, InexactNewton(EisenstatWalkerB(), 1e-8, 300), NEqOptions())
    #@test norm(res.info.best_residual, Inf) < 1e-12

    x0 = [-1.0, 0.0, 0.0]
    res = solve(prob, x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions(maxiter = 1000))
    @test norm(res.info.best_residual, Inf) < 1e-0

    function dfsane_exponential(Fx, x)
        Fx[1] = exp(x[1] - 1) - 1
        for i = 2:length(Fx)
            Fx[i] = i * (exp(x[i] - 1) - x[i])
        end
        Fx
    end

    function FJ_dfsane_exponential!(Fx, Jx, x)
        ForwardDiff.jacobian!(Jx, dfsane_exponential, Fx, x)
        Fx, Jx
    end
    prob_obj = NLSolvers.VectorObjective(
        dfsane_exponential,
        nothing,
        FJ_dfsane_exponential!,
        nothing,
    )

    n = 5000
    x0 = fill(n / (n - 1), n)
    res = solve(
        NEqProblem(prob_obj),
        x0,
        NLSolvers.DFSANE(),
        NLSolvers.NEqOptions(maxiter = 1000),
    )
    @test norm(res.info.best_residual, Inf) < 1e-5

    n = 1000
    x0 = fill(n / (n - 1), n)
    res = solve(
        NEqProblem(prob_obj),
        x0,
        LineSearch(NLSolvers.Newton()),
        NLSolvers.NEqOptions(f_abstol = 1e-6),
    ) #converges well but doing this for time
    @test norm(res.info.best_residual, Inf) < 1e-6

    n = 5000
    x0 = fill(n / (n - 1), n)
    res = solve(
        NEqProblem(prob_obj),
        x0,
        TrustRegion(NLSolvers.Newton()),
        NLSolvers.NEqOptions(f_abstol = 1e-6),
    ) #converges well but doing this for time
    @test norm(res.info.best_residual, Inf) < 1e-6

    #n = 1000
    #x0 = fill(n/(n-1), n)
    #@show solve(NEqProblem(prob_obj), x0, TrustRegion(NLSolvers.BFGS()), NLSolvers.NEqOptions())

    function dfsane_exponential2(Fx, x)
        Fx[1] = exp(x[1]) - 1
        for i = 2:length(Fx)
            Fx[i] = i / 10 * (exp(x[i]) + x[i-1] - 1)
        end
        Fx
    end
    dfsane_prob2 = NLSolvers.VectorObjective(dfsane_exponential2, nothing, nothing, nothing)
    n = 500
    x0 = fill(1 / (n^2), n)
    res = solve(NEqProblem(dfsane_prob2), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions())
    @test norm(res.info.best_residual, Inf) < 1e-5
    res = solve(
        NEqProblem(dfsane_prob2),
        Double64.(x0),
        NLSolvers.DFSANE(),
        NLSolvers.NEqOptions(),
    )
    @test norm(res.info.best_residual, Inf) < 1e-5
    n = 2000
    x0 = fill(1 / (n^2), n)
    res = solve(NEqProblem(dfsane_prob2), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions())
    @test norm(res.info.best_residual, Inf) < 1e-5




    # Take from problems
    prob = NEqProblem(NLE_PROBS["rosenbrock"]["array"]["mutating"])
    x0 = copy(NLE_PROBS["rosenbrock"]["array"]["x0"])
    X0 = [[-0.8, 1.0], [-0.7, 1.0], [-0.7, 1.2], [-0.5, 0.7], [0.5, 0.4], [-1.2, 1.0]]
    for x0 in X0
        res = solve(prob, x0, TrustRegion(Newton(), Dogleg()), NEqOptions())
        @test norm(res.info.best_residual, Inf) < 1e-12
        res = solve(prob, x0, TrustRegion(Newton(), NWI()), NEqOptions())
        @test norm(res.info.best_residual, Inf) < 1e-12
        res = solve(prob, x0, TrustRegion(Newton(), NTR()), NEqOptions())
        @test norm(res.info.best_residual, Inf) < 1e-12
        res = solve(prob, x0, LineSearch(Newton(), Static(0.6)), NEqOptions())
        @test norm(res.info.best_residual, Inf) < 1e-8
        res = solve(prob, copy(x0), DFSANE(), NEqOptions())
        @test norm(res.info.best_residual, Inf) < 1e-6
        res = solve(
            prob,
            copy(x0),
            InexactNewton(inexact_linsolve, FixedForceTerm(0.001), 1e-4, 300),
            NEqOptions(),
        )
        @test norm(res.info.best_residual, Inf) < 1e-8
    end

    @testset "fixedpoints" begin
        function G(Gx, x)
            V1 = [1.0, 0.0] .+ 0.99 * [0.1 0.9; 0.5 0.5] * x
            V2 = [0.0, 2.0] .+ 0.99 * [0.5 0.5; 1.0 0.0] * x
            K = max(maximum(V1), maximum(V2))
            Gx .= K .+ log.(exp.(V1 .- K) .+ exp.(V2 .- K))
        end
        fp1 = NLSolvers.fixedpoint!(G, zeros(2), Anderson(10000000, 1, nothing, nothing))
        fp2 = NLSolvers.fixedpoint!(G, zeros(2), Anderson(2, 2, 0.3, 1e2))
        fp3 = NLSolvers.fixedpoint!(G, zeros(2), Anderson(2, 2, 0.01, 1e2))
        @test norm(fp1.info.best_residual .- fp2.info.best_residual) < 1e-7
        @test norm(fp1.info.best_residual .- fp3.info.best_residual) < 1e-7

        fp1 = NLSolvers.solve(
            NEqProblem(
                NLSolvers.VectorObjective(
                    (F, x) -> G(F, x) .- x,
                    nothing,
                    nothing,
                    nothing,
                ),
            ),
            [0.0, 0.0],
            Anderson(10000000, 1, nothing, nothing),
            NEqOptions(),
        )
        fp2 = NLSolvers.solve(
            NEqProblem(
                NLSolvers.VectorObjective(
                    (F, x) -> G(F, x) .- x,
                    nothing,
                    nothing,
                    nothing,
                ),
            ),
            [0.0, 0.0],
            Anderson(2, 2, 0.3, 1e2),
            NEqOptions(),
        )
        fp2 = NLSolvers.solve(
            NEqProblem(
                NLSolvers.VectorObjective(
                    (F, x) -> G(F, x) .- x,
                    nothing,
                    nothing,
                    nothing,
                ),
            ),
            [0.0, 0.0],
            Anderson(2, 2, 0.01, 1e2),
            NEqOptions(),
        )
        @test norm(fp1.info.best_residual .- fp2.info.best_residual) < 1e-7
        @test norm(fp1.info.best_residual .- fp3.info.best_residual) < 1e-7
    end


    @testset "complementarity" begin
        # using NLsolve

        # M = [0  0 -1 -1 ;
        #      0  0  1 -2 ;
        #      1 -1  2 -2 ;
        #      1  2 -2  4 ]

        # q = [2; 2; -2; -6]

        # function f!(x, fvec)
        #     fvec = M * x + q
        # end


        # r = mcpsolve(f!, [0., 0., 0., 0.], [Inf, Inf, Inf, Inf],
        #              [1.25, 0., 0., 0.5], reformulation = :smooth, autodiff = true)

        # x = r.zero  # [1.25, 0.0, 0.0, 0.5]
        # @show dot( M*x + q, x )  # 0.5

        # sol = [2.8, 0.0, 0.8, 1.2]
        # @show dot( M*sol + q, sol )  # 0.0

    end
end

@testset "lsqfit" begin
    # boxbod_f(x, b) = b[1]*(1-exp(-b[2]*x))

    # ydata = [109, 149, 149, 191, 213, 224]
    # xdata = [1, 2, 3, 5, 7, 10]

    # start1 = [1.0, 1.0]
    # start2 = [100.0, 0.75]

    # using Plots
    # using NLSolvers

    # nd = NonDiffed(t->sum(abs2, boxbod_f.(xdata, Ref(t)).-ydata))
    # bounds = (fill(0.0, 2), fill(500.0, 2))
    # problem = MinProblem(; obj=nd, bounds=bounds)
    # @show minimize!(problem, zeros(2), ParticleSwarm(), OptimizationOptions())
    # @show minimize!(nd, [200.0, 1], NelderMead(), OptimizationOptions())

    # @show minimize!(nd, minimize!(nd, [200.0, 10.0], NelderMead(), OptimizationOptions()).info.solution, NelderMead(), OptimizationOptions())
    # @show minimize!(nd, minimize!(nd, [200.0, 5.0], NelderMead(), OptimizationOptions()).info.solution, NelderMead(), OptimizationOptions())

    # function F(b, F, J=nothing)
    #   @. F = b[1]*(1 - exp(-b[2]*xdata)) - ydata
    #   if !isa(J, Nothing)
    #   	@. J[:, 1] = 1 - exp(-b[2]*xdata)
    #   	@. @views J[:, 2] = xdata*b[1]*(J[:, 1]-1)
    #     return F, J
    #   end
    #   F
    # end

    # Fc = zeros(6)
    # minimize!(lsqwrap, [100.0, 1.0], NelderMead(), OptimizationOptions())

    # function F(F, b)
    #   @. F = b[1]*(1 - exp(-b[2]*xdata)) - ydata

    #   F
    # end
    # function F(J, F, b)
    #   @. F = b[1]*(1 - exp(-b[2]*xdata)) - ydata
    #   if !isa(J, Nothing)
    #   	@. J[:, 1] = 1 - exp(-b[2]*xdata)
    #   	@. @views J[:, 2] = xdata*b[1]*(J[:, 1]-1)
    #     return F, J
    #   end
    #   F
    # end
    # OnceDiffed(F)(rand(2), rand(6), rand(6,2)

    # Fc = zeros(6)




    # #using Plots
    # #theme(:ggplot2)
    # #gr(size=(500,500))
    # #X = range(000.0, 350.0; length=420)
    # #Y = range(0.0, 3.00; length=420)
    # #contour(X, Y, (x, y)->nd([x, y]);
    # #       fill=true,
    # #       c=:turbid,levels=200, ls=:dash,
    # #       xlims=(minimum(X), maximum(X)),
    # #       ylims=(minimum(Y), maximum(Y)),
    # #       colorbar=true)


end

@testset "Krylov" begin
    # Start with a stupid example
    n = 10
    A = sprand(10, 10, 0.1)
    A = A * A' + I
    F(Fx, x) = mul!(Fx, A, x)

    x0, = rand(10)
    xp = copy(x0)
    Fx = copy(xp)

    function theta(x)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    function F_powell!(Fx, x)
        if (x[1]^2 + x[2]^2 == 0)
            dtdx1 = 0
            dtdx2 = 0
        else
            dtdx1 = -x[2] / (2 * pi * (x[1]^2 + x[2]^2))
            dtdx2 = x[1] / (2 * pi * (x[1]^2 + x[2]^2))
        end
        Fx[1] =
            -2000.0 * (x[3] - 10.0 * theta(x)) * dtdx1 +
            200.0 * (sqrt(x[1]^2 + x[2]^2) - 1) * x[1] / sqrt(x[1]^2 + x[2]^2)
        Fx[2] =
            -2000.0 * (x[3] - 10.0 * theta(x)) * dtdx2 +
            200.0 * (sqrt(x[1]^2 + x[2]^2) - 1) * x[2] / sqrt(x[1]^2 + x[2]^2)
        Fx[3] = 200.0 * (x[3] - 10.0 * theta(x)) + 2.0 * x[3]
        Fx
    end

    function F_jacobian_powell!(Fx, Jx, x)
        ForwardDiff.jacobian!(Jx, F_powell!, Fx, x)
        Fx, Jx
    end
    x0 = [-1.0, 0.0, 0.0]

    Fc, Jc = zeros(3), zeros(3, 3)
    F_jacobian_powell!(Fc, Jc, x0)

    jv = JacVec(F_powell!, rand(3); autodiff = false)
    function jvop(x)
        jv.x .= x
        jv
    end
    prob_obj = NLSolvers.VectorObjective(F_powell!, nothing, F_jacobian_powell!, jvop)

    prob = NEqProblem(prob_obj)
    res = solve(prob, copy(x0), LineSearch(Newton(), Backtracking()))
    @test norm(res.info.best_residual) < 1e-15

    res = solve(
        prob,
        copy(x0),
        InexactNewton(inexact_linsolve, FixedForceTerm(0.001), 1e-4, 300),
        NEqOptions(maxiter = 1000),
    )
    @test norm(res.info.best_residual) < 1e-10

end

@testset "scalar nlsolves" begin
    function ff(x)
        x^2
    end

    function fgg(Jx, x)
        x^2, 2x
    end

    prob_obj = NLSolvers.ScalarObjective(f = ff, fg = fgg)

    prob = NEqProblem(prob_obj; inplace = false)

    x0 = 0.3
    res = solve(prob, x0, LineSearch(Newton(), Backtracking()))
end

function solve_static()
    function F_rosenbrock_static(Fx, x)
        Fx1 = 1 - x[1]
        Fx2 = 10(x[2] - x[1]^2)
        return @SVector([Fx1,Fx2])
    end
    function J_rosenbrock_static(Jx, x)
        Jx11 = -1
        Jx12 = 0
        Jx21 = -20 * x[1]
        Jx22 = 10
        return @SArray([Jx11 Jx12; Jx21 Jx22])
    end
    function FJ_rosenbrock_static(Fx, Jx, x)
        Fx = F_rosenbrock_static(Fx, x)
        Jx = J_rosenbrock_static(Jx, x)
        Fx, Jx
    end
    obj = NLSolvers.VectorObjective(
        F_rosenbrock_static,
        J_rosenbrock_static,
        FJ_rosenbrock_static,
        nothing,
    )

    prob_static =  NEqProblem(obj; inplace=false)
    x0_static = @SVector([-1.2, 1.0])
    res = solve(prob_static, x0_static, TrustRegion(Newton(), Dogleg()), NEqOptions())
end

solve_static()
alloced = @allocated solve_static()
@test alloced == 0
