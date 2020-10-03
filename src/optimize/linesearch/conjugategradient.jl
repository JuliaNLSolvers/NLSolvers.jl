#===============================================================================
  Conjugate Gradient Descent

  We implement a generic conjugate gradient descent method that includes many
  different β-choices, preconditioning, precise line searches, and more.

  A conjugate gradient method here will have to implement a CGUpdate type (CD,
  HZ, HS, ...) that controls the β update via the update_parameter function.

  The update formulae are mostly taken from HZ2006

  Let N be a neighborhood of {x ∈ R^n : f(x) ⩽ f(x₀)}

  [LA; Lipschitz Assumption]: ∃ L < ∞ : ||∇f(x)-∇f(y)|| ⩽ L||x-y|| for x,y ∈ N 

  Todos:
  Might consider Daniel 1967 that uses second order
===============================================================================#

abstract type CGUpdate end
struct ConjugateGradient{Tu, TP}
  update::Tu
  P::TP
end
ConjugateGradient(;update=HZ()) = ConjugateGradient(update, nothing)
hasprecon(::ConjugateGradient{<:Any, <:Nothing}) = NoPrecon()
hasprecon(::ConjugateGradient{<:Any, <:Any}) = HasPrecon()

struct CGVars{T1, T2, T3}
    y::T1 # change in successive gradients
    d::T2 # search direction
    α::T3
    β::T3
    ls_success::Bool
end

function prepare_variables(problem, approach::LineSearch{<:ConjugateGradient, <:Any, <:Any}, x0, ∇fz)
    z = x0
    x = copy(z)
    fz, ∇fz = upto_gradient(problem, ∇fz, x)

    fx = copy(fz)
    ∇fx = copy(∇fz)

    Pg = approach.scheme.P isa Nothing ? ∇fz : copy(∇fz)
    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=nothing, Pg=Pg)
end

summary(cg::ConjugateGradient) = "Conjugate Gradient Descent ($(summary(cg.update)))"
#===============================================================================
  Conjugate Descent [Fletcher] (CD)


  R. Fletcher. "Practical Methods of Optimization vol. 1: Unconstrained 
  Optimization." John Wiley & Sons, New York (1987).
===============================================================================#
struct CD <: CGUpdate end
function update_parameter(mstyle, cg::CD, d, ∇fz, ∇fx, y, P, P∇fz)
  -dot(∇fz, P∇fz)/dot(d, ∇fx)
end

#===============================================================================
  Hager-Zhang (HZ)
===============================================================================#
struct HZ{Tη} <: CGUpdate
    η::Tη # a "forcing term"
end
HZ() = HZ(0.4)
function update_parameter(mstyle, cg::HZ, d, ∇fz, ∇fx, y, P, P∇fz)
    T = eltype(∇fz)
    θ = T(2)
    η = T(cg.η)
    # βHS = update_parameter(mstyle, HS(), d,  ∇fz, ∇fx, y, P, P∇fz)
    # but let's save the dy calculation from begin repeated for
    # efficiency's sake        
    dy = dot(d, y)
    βHS = dot(y, P∇fz)/dy

    # Apply preconditioner to y
    Py = apply_preconditioner(mstyle, P, copy(P∇fz), y)
    βN = βHS - θ*dot(y, Py)/dy*dot(d, ∇fz)/dy
    
    # 2013 version is scale invariant
    Pinv_y = apply_inverse_preconditioner(mstyle, P, copy(P∇fz), y)
    ηk = η*dot(d, ∇fx)/dot(d, Pinv_y)
    # 2006 version
    # ηk = -inv(norm(d)*min(T(cg.η), norm(∇fx)))

    βkp1 = max(βN, ηk)
end

#===============================================================================
  Hestenes-Stiefel (HS)

  Jamming protection from y in the numerator. x-x' small means ∇f(x)-∇f(x')
  should also be small.

  Hestenes, Magnus R., and Eduard Stiefel. "Methods of conjugate gradients for
  solving linear systems." Journal of research of the National Bureau of 
  Standards 49, no. 6 (1952): 409-436.
===============================================================================#
struct HS <: CGUpdate end
function update_parameter(mstyle, cg::HS, d, ∇fz, ∇fx, y, P, P∇fz)
  dot(y, P∇fz)/dot(d, y)
end
#===============================================================================
  Fletcher-Reeves (FR)
  
  Fletcher, Reeves, and Colin M. Reeves. "Function minimization by conjugate
  gradients." The computer journal 7, no. 2 (1964): 149-154.
===============================================================================#
struct FR <: CGUpdate end
function update_parameter(mstyle, cg::FR, d, ∇fz, ∇fx, y, P, P∇fz)
    dot(∇fz, P∇fz)/dot(∇fx, ∇fx)
end
#===============================================================================
  Polak-Ribiére-Polyak (PRP)

  Jamming protection from y in the numerator. x-x' small means ∇f(x)-∇f(x')
  should also be small.

  Polak, Elijah, and Gerard Ribiere. "Note sur la convergence de méthodes de
  directions conjuguées." ESAIM: Mathematical Modelling and Numerical Analysis-
  Modélisation Mathématique et Analyse Numérique 3, no. R1 (1969): 35-43.
  
  Polyak, Boris T. "The conjugate gradient method in extremal problems." USSR
  Computational Mathematics and Mathematical Physics 9, no. 4 (1969): 94-112.
===============================================================================#
struct PRP{Plus} end
PRP(;plus=true) = PRP{plus}()
function update_parameter(mstyle, ::PRP, d, ∇fz, ∇fx, y, P, P∇fz)
    dot(y, P∇fz)/dot(∇fx, ∇fx)
end
function update_parameter(::PRP{true}, d, ∇fz, ∇fx, y, P, P∇fz)
  βPR = update_parameter(RP{false}(), d, ∇fz, ∇fx, y, P, P∇fz)
  max(0, β)
end

#===============================================================================
  Liu-Storey (LS)

  Liu-Storey is identical to PRP for exact line search.

  Jamming protection from y in the numerator. x-x' small means ∇f(x)-∇f(x')
  should also be small.

  Liu, Y., and C. Storey. "Efficient generalized conjugate gradient algorithms,
  part 1: theory." Journal of optimization theory and applications 69, no. 1
  (1991): 129-137.
===============================================================================#
struct LS <: CGUpdate end
function update_parameter(mstyle, ::LS, d, ∇fz, ∇fx, y, P, P∇fz)
  -dot(y, P∇fz)/dot(d, ∇fx)
end
#===============================================================================
  Dai-Yuan (DY)

  Different from CD and FR. With Wolfe line search it always generates descent
  directions. Global convergence with Lipchitz Assumption [LA].

  Dai, Yu-Hong, and Yaxiang Yuan. "A nonlinear conjugate gradient method with a
  strong global convergence property." SIAM Journal on optimization 10, no. 1
  (1999): 177-182.
===============================================================================#
struct DY <: CGUpdate end
function update_parameter(mstyle, cg::DY, d, ∇fz, ∇fx, y, P, P∇fz)
  dot(∇fz, P∇fz)/dot(d, y)
end

#===============================================================================
  Wei-Yao-Liu (VPRP)

  Yu, Gaohang, Yanlin Zhao, and Zengxin Wei. "A descent nonlinear conjugate
  gradient method for large-scale unconstrained optimization." Applied
  mathematics and computation 187, no. 2 (2007): 636-643.
===============================================================================#
struct VPRP <: CGUpdate end
function update_parameter(mstyle, cg::VPRP, d, ∇fz, ∇fx, y, P, P∇fz)
  a = dot(∇fz, P∇fz)
  b = dot(∇fx, P∇fz)
  c = norm(∇fx, 2)
  num = a - sqrt(a)/c*b
  num/dot(d, y)
end



# using an "initial vectors" function we can initialize s if necessary or nothing if not to save on vectors
function solve(problem::OptimizationProblem, x0, cg::ConjugateGradient, options::OptimizationOptions)
  _solve(problem, x0, LineSearch(cg, HZAW()), options, mstyle(problem))
end
function solve(problem::OptimizationProblem, x0, approach::LineSearch{<:ConjugateGradient, <:LineSearcher}, options::OptimizationOptions)
  _solve(problem, x0, approach, options, mstyle(problem))
end
function _solve(problem::OptimizationProblem, x0, approach::LineSearch{<:ConjugateGradient, <:LineSearcher}, options::OptimizationOptions, mstyle::MutateStyle)
    t0 = time()
    #==============
         Setup
    ==============#
    Tx = eltype(x0)

    objvars = prepare_variables(problem, approach, x0, copy(x0))
    f0, ∇f0 = objvars.fx, norm(objvars.∇fx, Inf) # use user norm

    y, d, α, β = copy(objvars.∇fz), -copy(objvars.∇fx), Tx(0), Tx(0)
    cgvars = CGVars(y, d, α, β, true)

    k = 1
    objvars, P, cgvars = iterate(mstyle, cgvars, objvars, approach, problem, options)
    is_converged = converged(approach, objvars, ∇f0, options)
    while k < options.maxiter && !any(is_converged)
        k += 1
        objvars, P, cgvars = iterate(mstyle, cgvars, objvars, approach, problem, options, P, false)
        is_converged = converged(approach, objvars, ∇f0, options) 
    end
    x, fx, ∇fx, z, fz, ∇fz, B = objvars
    return ConvergenceInfo(approach, (beta=β, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fx=∇fx, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=k, time=time()-t0), options)
end
function iterate(mstyle::InPlace, cgvars::CGVars, objvars, approach::LineSearch{<:ConjugateGradient, <:Any, <:Any}, problem::OptimizationProblem, options::OptimizationOptions, P=nothing, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    Tx = eltype(x)
  
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    y, d, α, β = cgvars.y, cgvars.d, cgvars.α, cgvars.β

    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Precondition current gradient and calculate the search direction
    P = update_preconditioner(scheme, x, P)
    P∇fz = apply_preconditioner(mstyle, P, Pg, ∇fz)
    @. d = -P∇fz + β*d

    α_0 = initial(scheme.update, a->value(problem, x.+a.*d), α, x, fx, dot(d, ∇fx), ∇fx, is_first)
    φ = _lineobjective(mstyle, problem, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tx(1))

    # Calculate final step vector and update the state
    if ls_success
        z = retract(problem, z, x, d, α)
        fz, ∇fz = upto_gradient(problem, ∇fz, z)
        @. y = ∇fz - ∇fx
    else
        # if no succesful search direction is found, reset to gradient
        y .= .-∇fz
    end
    β = update_parameter(mstyle, scheme.update, d, ∇fz, ∇fx, y, P, P∇fz)

    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=nothing, Pg=Pg), P, CGVars(y, d, α, β, ls_success)
end

function iterate(mstyle::OutOfPlace, cgvars::CGVars, objvars, approach::LineSearch{<:ConjugateGradient, <:Any, <:Any}, problem::OptimizationProblem, options::OptimizationOptions, P=nothing, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    Tx = eltype(x)
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    d, α, β = cgvars.d, cgvars.α, cgvars.β

    # Move nexts into currs
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)

    # Update preconditioner
    P = update_preconditioner(scheme, x, P)
    P∇fz = apply_preconditioner(mstyle, P, Pg, ∇fz)
    # Update current gradient and calculate the search direction
    d = @. -P∇fz + β*d

    α_0 = initial(scheme.update, a-> value(problem, x .+ a.*d), α, x, fx, dot(d, ∇fx), ∇fx, is_first)
    φ = _lineobjective(mstyle, problem, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tx(1))

    z = retract(problem, z, x, d, α)
    fz, ∇fz = upto_gradient(problem, ∇fz, z)
    y = @. ∇fz - ∇fx
    β = update_parameter(mstyle, scheme.update, d, ∇fz, ∇fx, y, P, P∇fz)
    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=nothing, Pg=Pg), P, CGVars(y, d, α, β, ls_success)
end



function initial(cg, φ, α, x, φ₀, dφ₀, ∇fx, is_first)
    T = eltype(x)
    ψ₀ = T(0.01)
    ψ₁ = T(0.1)
    ψ₂ = T(2.0)
    quadstep = true
    if is_first isa Nothing
        if !all(x .≈ T(0)) # should we define "how approx we want?"
            return ψ₀ * norm(x, Inf)/norm(∇fx, Inf)
        elseif !(φ₀ ≈ T(0))
            return ψ₀ * abs(φ₀)/norm(∇fx, 2)^2
        else
            return T(1)
        end
    elseif quadstep
        R = ψ₁*α
        φR = φ(R)
        if φR ≤ φ₀
            c = (φR - φ₀ - dφ₀*R)/R^2
            if c > 0
               return -dφ₀/(T(2)*c) # > 0 by df0 < 0 and c > 0
            end
        end
    end
    return ψ₂*α
end
