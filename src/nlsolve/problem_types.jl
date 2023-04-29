"""
  NEqProblem(residuals)
  NEqProblem(residuals, options)

mathematical problem of finding zeros in the residual function of square systems
of equations. The problem is defined by `residuals` which is an appropriate objective
type (for example `NonDiffed`, `OnceDiffed`, ...) for the types of algorithm to be used.

Options are stored in `options` and are of the `NEqOptions` type. See more information
about options using `?NEqOptions`.

The package NLSolversAD.jl adds automatic conversion of problems to match algorithms
that require higher order derivates than provided by the user. It also adds AD
constructors for a target number of derivatives.
"""
struct NEqProblem{TR,Tb,Tm<:Manifold,I}
    R::TR
    bounds::Tb
    manifold::Tm
    mstyle::I
end
function NEqProblem(residuals; inplace = true)
    mstyle = inplace === true ? InPlace() : OutOfPlace()
    NEqProblem(residuals, nothing, Euclidean(0), mstyle)
end
mstyle(problem::NEqProblem) = problem.mstyle
_manifold(prob::NEqProblem) = prob.manifold
is_inplace(problem::NEqProblem) = mstyle(problem) === InPlace()

#function value(nleq::NEqProblem, x)
#    nleq.R(x)
#end
function value(nleq::NEqProblem, x, F)
    value(nleq.R, x, F)
end
struct VectorObjective{TF,TJ,TFJ,TJv}
    F::TF
    J::TJ
    FJ::TFJ
    Jv::TJv
end
function value(nleq::VectorObjective, F, x)
    nleq.F(F, x)
end
function value_jacobian!(nleq::NEqProblem{<:VectorObjective,<:Any,<:Any}, F, J, x)
    nleq.FJ(F, J, x)
end

"""
  NEqOptions(; ...)

NEqOptions are used to control the behavior of solvers for non-linear systems of
equations. Current options are:

  - `maxiter` [= 10000]: number of major iterations where appropriate
"""
struct NEqOptions{T,Tmi,Call}
    f_limit::T
    f_abstol::T
    f_reltol::T
    maxiter::Tmi
    callback::Call
    show_trace::Bool
end
NEqOptions(;
    f_limit = 0.0,
    f_abstol = 1e-8,
    f_reltol = 1e-12,
    maxiter = 10^4,
    callback = nothing,
    show_trace = false,
) = NEqOptions(f_limit, f_abstol, f_reltol, maxiter, callback, show_trace)

function OptimizationOptions(neq::NEqOptions)
    return OptimizationOptions(;
    x_abstol = 0.0,
    x_reltol = 0.0,
    x_norm = x -> norm(x, Inf),
    g_abstol = 2*neq.f_abstol,
    g_reltol = 2*neq.f_reltol,
    g_norm = x -> norm(x, Inf),
    f_limit = neq.f_limit,
    f_abstol = 0.0,
    f_reltol = 0.0,
    nm_tol = 1e-8,
    maxiter = neq.maxiter,
    show_trace = neq.show_trace,
)
end

function Base.show(io::IO, ci::ConvergenceInfo{<:Any,<:Any,<:NEqOptions})
    opt = ci.options
    info = ci.info

    println(io, "Results of solving non-linear equations\n")
    println(io, "* Algorithm:")
    println(io, "  $(summary(ci.solver))")
    println(io)
    println(io, "* Candidate solution:")
    println(
        io,
        "  Final residual 2-norm:      $(@sprintf("%.2e", norm(ci.info.best_residual, 2)))",
    )
    println(
        io,
        "  Final residual Inf-norm:    $(@sprintf("%.2e", norm(ci.info.best_residual, Inf)))",
    )
    if haskey(info, :temperature)
        println(io, "  Final temperature:        $(@sprintf("%.2e", ci.info.temperature))")
    end
    println(io)
    println(io, "  Initial residual 2-norm:    $(@sprintf("%.2e", info.ρ2F0))")
    println(io, "  Initial residual Inf-norm:  $(@sprintf("%.2e", info.ρF0))")
    println(io)
    println(io, "* Stopping criteria")
    if true
        #    println(io, "  |x - x'|              = $(@sprintf("%.2e", info.ρs)) <= $(@sprintf("%.2e", opt.x_abstol)) ($(info.ρs<=opt.x_abstol))")
        #    println(io, "  |x - x'|/|x|          = $(@sprintf("%.2e", info.ρs/info.ρx)) <= $(@sprintf("%.2e", opt.x_reltol)) ($(info.ρs/info.ρx <= opt.x_reltol))")
        if isfinite(opt.f_limit)
            ρF = norm(info.best_residual, Inf)
            println(
                io,
                "  |F(x')|               = $(@sprintf("%.2e", ρF)) <= $(@sprintf("%.2e", opt.f_limit)) ($(ρF<=opt.f_limit))",
            )
        end
        if haskey(info, :fx)
            Δf = abs(info.fx - info.minimum)
            println(
                io,
                "  |f(x) - f(x')|        = $(@sprintf("%.2e", Δf)) <= $(@sprintf("%.2e", opt.f_abstol)) ($(Δf<=opt.f_abstol))",
            )
            println(
                io,
                "  |f(x) - f(x')|/|f(x)| = $(@sprintf("%.2e", Δf/abs(info.fx))) <= $(@sprintf("%.2e", opt.f_reltol)) ($(Δf/abs(info.fx)<=opt.f_reltol))",
            )
        end
        if haskey(info, :∇fz)
            ρ∇f = opt.g_norm(info.∇fz)
            println(
                io,
                "  |g(x)|                = $(@sprintf("%.2e", ρ∇f)) <= $(@sprintf("%.2e", opt.g_abstol)) ($(ρ∇f<=opt.g_abstol))",
            )
            println(
                io,
                "  |g(x)|/|g(x₀)|        = $(@sprintf("%.2e", ρ∇f/info.∇f0)) <= $(@sprintf("%.2e", opt.g_reltol)) ($(ρ∇f/info.∇f0<=opt.g_reltol))",
            )
        end
    end
    println(io)
    println(io, "* Work counters")
    println(io, "  Seconds run:   $(@sprintf("%.2e", info.time))")
    println(io, "  Iterations:    $(info.iter)")
end

function upto_gradient(merit::MeritObjective, Fx, x)
    upto_gradient(merit.prob, Fx, x)
    #merit.F .= Fx
    #return res
end
upto_gradient(neq::NEqProblem, Fx, x) =
    upto_gradient(neq.R, Fx, x)

function upto_gradient(vo::VectorObjective, Fx, x)
    value(vo,Fx,x)
    obj = norm(Fx)^2/2
    return obj, Fx
end

function upto_hessian(merit::MeritObjective, ∇f, ∇²f, x)
    upto_hessian(merit.prob, ∇f, ∇²f, x)
end

upto_hessian(neq::NEqProblem, ∇f, ∇²f, x) =
    upto_hessian(neq.R, ∇f, ∇²f, x)

function upto_hessian(vo::VectorObjective, ∇f, ∇²f, x)
    Fx, Jx = vo.FJ(∇f, ∇²f, x)
    obj = norm(Fx)^2/2
    return obj, Fx, Jx
end

function LineObjective(obj::TP,∇fz::T1,z::T2,x::T2,d::T2,φ0::T3,dφ0::T3) where {TP<:(NLSolvers.OptimizationProblem{<:NLSolvers.MeritObjective}),T1,T2,T3}
    return LineObjective(obj.objective,∇fz,z,x,d,φ0,dφ0)
end

function LineObjective!(obj::TP,∇fz::T1,z::T2,x::T2,d::T2,φ0::T3,dφ0::T3) where {TP<:(NLSolvers.OptimizationProblem{<:NLSolvers.MeritObjective}),T1,T2,T3}
    return LineObjective!(obj.objective,∇fz,z,x,d,φ0,dφ0)
end

function ls_has_gradient(obj::MeritObjective)
    if obj.prob.R.Jv === nothing
        throw(error("You need to provide Jv for using LineSearch with gradient information."))
    end
    return nothing
end

function ls_upto_gradient(merit::MeritObjective, Fx, x)
    upto_gradient(merit.prob, Fx, x)
end
upto_gradient(neq::NEqProblem, Fx, x) =
    upto_gradient(neq.R, Fx, x)

function upto_gradient(vo::VectorObjective, Fx, x)
    vo.Jv(x)
    value(vo,Fx,x)
    obj = norm(Fx)^2/2
    return obj, Fx
end


_manifold(merit::NLSolvers.MeritObjective) = _manifold(merit.prob)

bounds(neq::NEqProblem) = neq.bounds



