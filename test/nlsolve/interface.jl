using Revise
using NLSolvers
using LinearAlgebra
using SparseDiffTools
using SparseArrays
using IterativeSolvers
using DoubleFloats
using ForwardDiff
using Test

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
    if ( x[1]^2 + x[2]^2 == 0 )
        dtdx1 = 0;
        dtdx2 = 0;
    else
        dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
        dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
    end
    Fx[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
    200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
    Fx[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
    200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
    Fx[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
    Fx
end

function F_jacobian_fletcher_powell!(Fx, Jx, x)
    ForwardDiff.jacobian!(Jx, F_fletcher_powell!, Fx, x)
    Fx, Jx
end

jv = JacVec(F_fletcher_powell!, rand(3); autodiff=false)
function jvop(x)
    jv.u .= x
    jv
end
prob_obj = NLSolvers.NEqObjective(F_fletcher_powell!, nothing, F_jacobian_fletcher_powell!, jvop)

prob = NEqProblem(prob_obj)

x0 = [-1.0, 0.0, 0.0]
res = solve(prob, x0, LineSearch(Newton(), Backtracking()))
@test norm(res.info.best_residual, Inf) < 1e-12

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
@test norm(res.info.best_residual, Inf) < 1e-9

x0 = [-1.0, 0.0, 0.0]
state = (z=copy(x0), d=copy(x0), Fx=copy(x0), Jx=zeros(3,3))
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
res = solve(prob, x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions(maxiter=1000))
@test norm(res.info.best_residual, Inf) < 1e-0

function dfsane_exponential(Fx, x)
    Fx[1] = exp(x[1]-1)-1
    for i = 2:length(Fx)
  	    Fx[i] = i*(exp(x[i]-1)-x[i])
    end
    Fx
end

function FJ_dfsane_exponential!(Fx, Jx, x)
  ForwardDiff.jacobian!(Jx, dfsane_exponential, Fx, x)
  Fx, Jx
end
prob_obj = NLSolvers.NEqObjective(dfsane_exponential, nothing, FJ_dfsane_exponential!, nothing)

n = 5000
x0 = fill(n/(n-1), n)
res = solve(NEqProblem(prob_obj), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions(maxiter=1000))
@test norm(res.info.best_residual, Inf)<1e-5

n = 1000
x0 = fill(n/(n-1), n)
res = solve(NEqProblem(prob_obj), x0, LineSearch(NLSolvers.Newton()), NLSolvers.NEqOptions(f_abstol=1e-6)) #converges well but doing this for time
@test norm(res.info.best_residual, Inf)<1e-6

n = 5000
x0 = fill(n/(n-1), n)
res = solve(NEqProblem(prob_obj), x0, TrustRegion(NLSolvers.Newton()), NLSolvers.NEqOptions(f_abstol=1e-6)) #converges well but doing this for time
@test norm(res.info.best_residual, Inf)<1e-6

#n = 1000
#x0 = fill(n/(n-1), n)
#@show solve(NEqProblem(prob_obj), x0, TrustRegion(NLSolvers.BFGS()), NLSolvers.NEqOptions())

function dfsane_exponential2(Fx, x)
    Fx[1] = exp(x[1])-1
    for i = 2:length(Fx)
        Fx[i] = i/10*(exp(x[i])+x[i-1]-1)
    end
    Fx
end
dfsane_prob2 = NLSolvers.NEqObjective(dfsane_exponential2, nothing, nothing,nothing)
n = 500
x0 = fill(1/(n^2), n)
res = solve(NEqProblem(dfsane_prob2), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions())
@test norm(res.info.best_residual, Inf) < 1e-5 
res = solve(NEqProblem(dfsane_prob2), Double64.(x0), NLSolvers.DFSANE(), NLSolvers.NEqOptions())
@test norm(res.info.best_residual, Inf) < 1e-5 
n = 2000
x0 = fill(1/(n^2), n)
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
    #res = solve(prob, copy(initial), DFSANE(), NEqOptions())
    #res = solve(prob, copy(initial), InexactNewton(), NEqOptions())    
end

@testset "fixedpoints" begin
# function G(x, Gx)
#     V1 = [1.0, 0.0] .+ 0.99*[0.1 0.9; 0.5 0.5]*x
#     V2 = [0.0, 2.0] .+ 0.99*[0.5 0.5; 1.0 0.0]*x
#     K = max(maximum(V1), maximum(V2))
#     Gx .= K .+ log.(exp.(V1 .- K) .+ exp.(V2 .- K))
#   end
#   fp1 = NLSolvers.fixedpoint!(G, zeros(2), Anderson(10000000,1, nothing,nothing))
#   fp2 = NLSolvers.fixedpoint!(G, zeros(2), Anderson(2, 2, 0.3, 1e2))
#   fp3 = NLSolvers.fixedpoint!(G, zeros(2), Anderson(2, 2, 0.01, 1e2))
#   @test all(fp1.x .≈ fp2.x)
#   @test all(fp1.x .≈ fp3.x)
  
#   fp1 = NLSolvers.solve(NEqProblem((x, F)->G(x, F).-x), zeros(2), Anderson(10000000,1, nothing,nothing))
#   fp2 = NLSolvers.solve(NEqProblem((x, F)->G(x, F).-x), zeros(2), Anderson(2, 2, 0.3, 1e2))
#   fp3 = NLSolvers.solve(NEqProblem((x, F)->G(x, F).-x), zeros(2), Anderson(2, 2, 0.01, 1e2))
#   @test all(fp1.x .≈ fp2.x)
#   @test all(fp1.x .≈ fp3.x)
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
# @show minimize!(problem, zeros(2), APSO(), OptimizationOptions())
# @show minimize!(nd, [200.0, 1], NelderMead(), OptimizationOptions())

# @show minimize!(nd, minimize!(nd, [200.0, 10.0], NelderMead(), OptimizationOptions()).info.minimizer, NelderMead(), OptimizationOptions())
# @show minimize!(nd, minimize!(nd, [200.0, 5.0], NelderMead(), OptimizationOptions()).info.minimizer, NelderMead(), OptimizationOptions())

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

# lsqwrap = NLSolvers.LsqWrapper(F, zeros(6), zeros(6,2))
# minimize!(MinProblem(;obj=lsqwrap,bounds=([0.0,0.0],[250.0,2.0])), [100.0, 1.0], APSO(), OptimizationOptions())

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


# lsqwrap = NLSolvers.LsqWrapper(OnceDiffed(F), zeros(6), zeros(6,2))
# minimize!(lsqwrap, [100.0, 1.0], LineSearch(LBFGS()), OptimizationOptions())


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