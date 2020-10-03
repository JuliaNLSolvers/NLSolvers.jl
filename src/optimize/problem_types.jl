"""
  OptimizationProblem(...)
A OptimizationProblem (Minimization Problem), is used to represent the mathematical problem
of finding local minima of the given objective function. The problem is defined by `objective`
which is an appropriate objective type (for example `NonDiffed`, `OnceDiffed`, ...)
for the types of algorithm to be used. The constraints of the problem are encoded
in `constraints`. See the documentation for supported types of constraints
including convex sets, and more. It is possible to explicitly state that there
are bounds constraints and manifold constraints on the inputs.

Options are stored in `options` and must be an appropriate options type. See more information
about options using `?OptimizationOptions`.
"""
struct OptimizationProblem{O, B, M<:Manifold, C, I, X} <: AbstractProblem
    objective::O
    bounds::B
    manifold::M
    constraints::C
    mstyle::I
    initial::X
end
value(prob::OptimizationProblem, x) = value(prob.objective, x)
batched_value(prob::OptimizationProblem, X) = batched_value(prob.objective, X)
batched_value(prob::OptimizationProblem, F, X) = batched_value(prob.objective, F, X)
upto_gradient(prob::OptimizationProblem, ∇f, x) = upto_gradient(prob.objective, ∇f, x)
upto_hessian(prob::OptimizationProblem, ∇f, ∇²f, x) = upto_hessian(prob.objective, ∇f, ∇²f, x)
_manifold(prob::OptimizationProblem) = prob.manifold
lowerbounds(mp::OptimizationProblem) = mp.bounds[1]
upperbounds(mp::OptimizationProblem) = mp.bounds[2]
hasbounds(mp::OptimizationProblem) = mp.bounds isa Tuple
bounds(mp::OptimizationProblem) = (lower=lowerbounds(mp), upper=upperbounds(mp))
isboundedonly(::OptimizationProblem{<:Any, <:Nothing, <:Any, <:Nothing}) = false
isboundedonly(::OptimizationProblem{<:Any, <:Nothing, <:Any, <:Any}) = false
isboundedonly(::OptimizationProblem{<:Any, <:Any, <:Any, <:Nothing}) = true
#value(p::OptimizationProblem, args...) = p.objective(args...)
constraints(p::OptimizationProblem, args...) = p.constraints(args...)
OptimizationProblem(; obj=nothing, bounds=nothing, manifold=Euclidean(0), constraints=nothing, inplace=true, initial_x=nothing) =
  OptimizationProblem(obj, bounds, manifold, constraints, inplace===true ? InPlace() : OutOfPlace(), initial_x)
OptimizationProblem(obj; inplace=true) = OptimizationProblem(obj, nothing, Euclidean(0), nothing, inplace===true ? InPlace() : OutOfPlace(), nothing)

struct ConvergenceInfo{Ts, T, O}
  solver::Ts
  info::T
  options::O
end
function Base.show(io::IO, ci::ConvergenceInfo)
  opt = ci.options
  info = ci.info

  println(io, "Results of minimization\n")
  println(io, "* Algorithm:")
  println(io, "  $(summary(ci.solver))")
  println(io)
  println(io, "* Candidate solution:")
  println(io, "  Final objective value:    $(@sprintf("%.2e", ci.info.minimum))")
  if haskey(info, :∇fz)
    println(io, "  Final gradient norm:      $(@sprintf("%.2e", opt.g_norm(info.∇fz)))")
    if haskey(info, :prob) && hasbounds(info.prob)
      ρP = opt.g_norm(info.minimizer.-clamp.(info.minimizer.-info.∇fz, info.prob.bounds...))
      println(io, "  Final projected gradient norm:  $(@sprintf("%.2e", ρP))")
    end
  end
  if haskey(info, :temperature)
    println(io, "  Final temperature:        $(@sprintf("%.2e", ci.info.temperature))")
  end
  println(io)
  println(io, "  Initial objective value:  $(@sprintf("%.2e", ci.info.f0))")
  if haskey(info, :∇f0)
    println(io, "  Initial gradient norm:    $(@sprintf("%.2e", info.∇f0))")
  end
  println(io)
  println(io, "* Convergence measures")
  if isa(ci.solver, NelderMead)
    nm_converged(r) = 0.0
    println(io, "  √(Σ(yᵢ-ȳ)²)/n         = $(@sprintf("%.2e", info.nm_obj)) <= $(@sprintf("%.2e", opt.nm_tol)) ($(info.nm_obj<=opt.nm_tol))")
  elseif isa(ci.solver, SimulatedAnnealing)
  else
    if haskey(info, :ρs)
      println(io, "  |x - x'|              = $(@sprintf("%.2e", info.ρs)) <= $(@sprintf("%.2e", opt.x_abstol)) ($(info.ρs<=opt.x_abstol))")
      println(io, "  |x - x'|/|x|          = $(@sprintf("%.2e", info.ρs/info.ρx)) <= $(@sprintf("%.2e", opt.x_reltol)) ($(info.ρs/info.ρx <= opt.x_reltol))")
    end
    if isfinite(opt.f_limit)
      println(io, "  |f(x')|               = $(@sprintf("%.2e", info.minimum)) <= $(@sprintf("%.2e", opt.f_limit)) ($(info.minimum<=opt.f_limit))")
    end
    if haskey(info, :fx)
      Δf = abs(info.fx-info.minimum)
      println(io, "  |f(x) - f(x')|        = $(@sprintf("%.2e", Δf)) <= $(@sprintf("%.2e", opt.f_abstol)) ($(Δf<=opt.f_abstol))")
      println(io, "  |f(x) - f(x')|/|f(x)| = $(@sprintf("%.2e", Δf/abs(info.fx))) <= $(@sprintf("%.2e", opt.f_reltol)) ($(Δf/abs(info.fx)<=opt.f_reltol))")
    end
    if haskey(info, :∇fz)
      ρ∇f = opt.g_norm(info.∇fz)
      if haskey(info, :prob) && hasbounds(info.prob)
        println(io, "  |x - P(x - g(x))|     = $(@sprintf("%.2e", ρP)) <= $(@sprintf("%.2e", opt.g_abstol)) ($(ρP<=opt.g_abstol))")
      end
      println(io, "  |g(x)|                = $(@sprintf("%.2e", ρ∇f)) <= $(@sprintf("%.2e", opt.g_abstol)) ($(ρ∇f<=opt.g_abstol))")
      println(io, "  |g(x)|/|g(x₀)|        = $(@sprintf("%.2e", ρ∇f/info.∇f0)) <= $(@sprintf("%.2e", opt.g_reltol)) ($(ρ∇f/info.∇f0<=opt.g_reltol))")
    end
    if haskey(info, :Δ)
      Δmin = ci.solver.Δupdate.Δmin isa Nothing ? 0 : ci.solver.Δupdate.Δmin
      if isnan(info.Δ)
        println(io, "  Δ                     = $(@sprintf("%.2e", info.Δ)) (updated radius was not finite)")
      else
        Δtest = info.Δ<=Δmin
        println(io, "  Δ                     = $(@sprintf("%.2e", info.Δ)) <= $(@sprintf("%.2e", Δmin)) ($Δtest)")
      end
    end
    if haskey(info, :prob) && hasbounds(info.prob)
      if any(iszero, info.minimizer.-info.prob.bounds[1]) || any(iszero, info.prob.bounds[2].-info.minimizer)
        println(io, "\n  !!! Solution is at the boundary !!!")
      end
    end
  end
  println(io)
  println(io, "* Work counters")
  println(io, "  Seconds run:   $(@sprintf("%.2e", info.time))")
  println(io, "  Iterations:    $(info.iter)")
end

struct OptimizationOptions{T1, T2, T3, T4, Txn, Tgn, Tlog}
  x_abstol::T1
  x_reltol::T1
  x_norm::Txn
  g_abstol::T2
  g_reltol::T2
  g_norm::Tgn
  f_limit::T3
  f_abstol::T3
  f_reltol::T3
  nm_tol::T3
  maxiter::T4
  show_trace::Bool
  logger::Tlog
end

OptimizationOptions(; x_abstol=0.0, x_reltol=0.0, x_norm=x->norm(x, Inf),
             g_abstol=1e-8, g_reltol=0.0, g_norm=x->norm(x, Inf),
             f_limit=-Inf, f_abstol=0.0, f_reltol=0.0,
             nm_tol=1e-8, maxiter=10000, show_trace=false, logger=show_trace ? ConsoleLogger() : NullLogger()) =
  OptimizationOptions(x_abstol, x_reltol, x_norm,
             g_abstol, g_reltol, g_norm,
             f_limit, f_abstol, f_reltol,
             nm_tol,
             maxiter, show_trace, logger)

struct MinResults{Tr, Tc<:ConvergenceInfo, Th, Ts, To}
  res::Tr
  conv::Tc
  history::Th
  solver::Ts
  options::To
end
convinfo(mr::MinResults) = mr.conv
function converged(MinResults)
end 
function Base.show(io::IO, mr::MinResults)
  println(io, "* Status: $(any(converged(mr)))")
  println(io)
  println(io, "* Candidate solution")
  println(io, "  MinimizerMinOpt: ", minimizer(mr))
  println(io, "  Minimum:   ", minimum(mr))
  println(io)
  println(io, "* Found with")
  println(io, "  Algorithm: ", summary(mr.solver))
  println(io, "  Initial point: ", initial_point(mr))
  println(io, "  Initial value: ", initial_value(mr))

  println(" * Trace stored: ", has_history(mr))


  println(io)
  println(io, " * Convergence measures\n")
  show(io, convinfo(mr))
end
#=
* Work counters
  Seconds run:   0  (vs limit Inf)
  Iterations:    16
  f(x) calls:    53
  ∇f(x) calls:   53
=#

function prepare_variables(prob, approach, x0, ∇fz, B)
    objective = prob.objective
    z = x0
    x = copy(z)
    if isboundedonly(prob)
        !any(clamp.(x0, lowerbounds(prob), upperbounds(prob)) .!= x0) || error("Initial guess not in the feasible region")
    end

    if isa(B, Nothing)  # didn't provide a B
        if modelscheme(approach) isa GradientDescent
            # We don't need to maintain a dense matrix for Gradient Descent
            B = I
        elseif modelscheme(approach) isa LBFGS
            B = nothing
        else
            # Construct a matrix on the correct form and of the correct type
            # with the content of I_{n,n}
            B = I + abs.(0*x*x')
        end
    end
    # first evaluation
    if isa(modelscheme(approach), Newton)
        fz, ∇fz, B = upto_hessian(prob, ∇fz, B, x)
    else
        fz, ∇fz = upto_gradient(prob, ∇fz, x)
    end
    fx = copy(fz)
    ∇fx = copy(∇fz)
    Pg = ∇fz
    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=B, Pg=Pg)
end

function g_converged(∇fz, ∇f0, options)
  g_converged = options.g_norm(∇fz) ≤ options.g_abstol
  g_converged = g_converged || options.g_norm(∇fz) ≤ ∇f0*options.g_reltol
  g_converged = g_converged || any(isnan, ∇fz)
  return g_converged
end

function x_converged(x, z, options)
  x_converged = false
  if x !== nothing # if not calling from initial_converged
    y = x .- z
    ynorm = options.x_norm(y)
    x_converged = x_converged || ynorm ≤ options.x_abstol
    x_converged = x_converged || ynorm ≤ options.x_norm(x)*options.x_reltol
  end
  x_converged = x_converged || any(isnan, z)
  return x_converged
end
function f_converged(fx, fz, options)
  f_converged = false
  if fx !== nothing # if not calling from initial_converged
    y = fx - fz
    ynorm = abs(y)
    f_converged = f_converged || ynorm ≤ options.f_abstol
    f_converged = f_converged || ynorm ≤ abs(fx)*options.f_reltol
  end
  f_converged = f_converged || fz ≤ options.f_limit
  f_converged = f_converged || isnan(fz)
  return f_converged
end
function converged(approach, objvars, ∇f0, options, skip=nothing, Δkp1=nothing)
  x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
  xcon, gcon, fcon = x_converged(x, z, options), g_converged(∇fz, ∇f0, options), f_converged(fx, fz, options)
  if approach isa TrustRegion
    Δcon = isnan(Δkp1)
    Δcon = Δcon || (!(approach.Δupdate.Δmin isa Nothing) && Δkp1 < approach.Δupdate.Δmin)
    if skip == true
      # special logic for region reduction here
      return false, false, false, Δcon
    end
    return xcon, gcon, fcon, Δcon 
  end
  return xcon, gcon, fcon
end


function initial_converged(approach, objvars, ∇f0, options, skip=nothing, Δkp1=nothing)
  x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
  objvars = (x=nothing, fx=nothing, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=B, Pg=Pg)
  converged(approach, objvars, ∇f0, options, skip, Δkp1)
end