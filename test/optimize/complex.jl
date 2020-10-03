using NLSolvers, LinearAlgebra
n = 4
A = randn(n,n) + im*randn(n,n)
A = A'A + I
b = randn(n) + im*randn(n)
μ = 1.0

fcomplex(x) = real(dot(x,A*x)/2 - dot(b,x)) + μ*sum(abs.(x).^4)
gcomplex(x) = A*x-b + 4μ*(abs.(x).^2).*x
gcomplex!(stor,x) = copyto!(stor,gcomplex(x))
function fgcomplex!(stor,x)
copyto!(stor,gcomplex(x))
fcomplex(x), stor
end
obj = ScalarObjective(fcomplex, gcomplex!, fgcomplex!, nothing, nothing, nothing,nothing, nothing)
prob = OptimizationProblem(obj)
prob_oop = OptimizationProblem(obj; inplace=false)
x0 = randn(n)+im*randn(n)



####### WTF ! Inplace takes 1000 iters and oop doesn't
# solve(prob, x0, APSO(), OptimizationOptions()) # fails
solve(prob, x0, SimulatedAnnealing(), OptimizationOptions())
solve(prob, copy(x0), LineSearch(LBFGS(), Backtracking()), OptimizationOptions()) # fails due to bounds
solve(prob, copy(x0), LineSearch(LBFGS(), Backtracking()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
solve(prob, copy(x0), LineSearch(LBFGS()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
solve(prob, copy(x0), LineSearch(LBFGS(), HZAW()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
solve(prob, copy(x0), LineSearch(BFGS(), Backtracking()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
solve(prob, copy(x0), LineSearch(BFGS()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
solve(prob, copy(x0), LineSearch(BFGS(), HZAW()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
solve(prob_oop, x0, LineSearch(BFGS(), Backtracking()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
solve(prob_oop, x0, LineSearch(BFGS()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
solve(prob_oop, x0, LineSearch(BFGS(), HZAW()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds

solve(prob, copy(x0), LineSearch(BFGS(), Backtracking()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
solve(prob_oop, x0, LineSearch(BFGS(), Backtracking()), OptimizationOptions(g_abstol=1e-6)) # fails due to bounds
#=

@testset "Finite difference setup" begin
    oda1 = OnceDifferentiable(fcomplex, x0)
    fx, gx = NLSolversBase.value_gradient!(oda1, x0)
    @test fx == fcomplex(x0)
    @test gcomplex(x0) ≈ NLSolversBase.gradient(oda1)
end

@testset "Zeroth and second order methods" begin
    for method in (Optim.NelderMead,Optim.ParticleSwarm,Optim.Newton)
        @test_throws Any Optim.optimize(fcomplex, gcomplex!, x0, method())
    end
    #not supposed to converge, but it should at least go through without errors
    res = Optim.optimize(fcomplex, gcomplex!, x0, Optim.SimulatedAnnealing())
end


@testset "First order methods" begin
    options = Optim.Options(allow_f_increases=true)
    # TODO: AcceleratedGradientDescent fail to converge?
    for method in (Optim.GradientDescent(), Optim.ConjugateGradient(), Optim.LBFGS(), Optim.BFGS(),
                    Optim.NGMRES(), Optim.OACCEL(),Optim.MomentumGradientDescent(mu=0.1))

        debug_printing && printstyled("Solver: $(summary(method))\n", color=:green)
        res = Optim.optimize(fcomplex, gcomplex!, x0, method, options)
        debug_printing && printstyled("Iter\tf-calls\tg-calls\n", color=:green)
        debug_printing && printstyled("$(Optim.iterations(res))\t$(Optim.f_calls(res))\t$(Optim.g_calls(res))\n", color=:red)
        if !Optim.converged(res)
            @warn("$(summary(method)) failed.")
            display(res)
            println("########################")
        end
        ressum = summary(res) # Just check that no errors arise when doing display(res)
        @test typeof(fcomplex(x0)) == typeof(Optim.minimum(res))
        @test eltype(x0) == eltype(Optim.minimizer(res))
        @test Optim.converged(res)
        @test Optim.minimizer(res) ≈ xref rtol=1e-4

        res = Optim.optimize(fcomplex, x0, method, options)
        @test Optim.converged(res)
        @test Optim.minimizer(res) ≈ xref rtol=1e-4

        # # To compare with the equivalent real solvers
        # to_cplx(x) = x[1:n] + im*x[n+1:2n]
        # from_cplx(x) = [real(x);imag(x)]
        # freal(x) = fcomplex(to_cplx(x))
        # greal!(stor,x) = copyto!(stor, from_cplx(gcomplex(to_cplx(x))))
        # opt = Optim.Options(allow_f_increases=true,show_trace=true)
        # println("$(summary(method)) cplx")
        # res_cplx = Optim.optimize(fcomplex,gcomplex!,x0,method,opt)
        # println("$(summary(method)) real")
        # res_real = Optim.optimize(freal,greal!,from_cplx(x0),method,opt)
    end
end
=#