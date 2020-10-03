module NLSolvers

import Base: show, summary

using Logging

#============================ LinearAlgebra ===========================
  We use often use the LinearAlgebra functions dot and norm for opera-
  tions related to assessing angles between vectors, size of vectors
  and so on.

  mul!, rmul!, ldiv!, ect can to make Array{T, N} operations as fast
  a possible.

  cholesky is impossible to live without to replace inverses

  UniformScaling (I), Symmetric, Diagonal are all usefull to handle
  shifted systems, hessians, etc

  eigen and diag are useful in trust region subprblems

  opnorm is used in the trust region subproblem safe guards to bound
  the estimate on lambda
 ============================ LinearAlgebra ===========================#

using LinearAlgebra:  dot, I, norm, # used everywhere in updates, convergence, etc
                      mul!, rmul!, ldiv!, # quasi-newton updates, apply factorizations, etc
                      cholesky, factorize, issuccess, # very useful in trust region solvers
                      UniformScaling, Diagonal, # simple matrices
                      Symmetric, Hermitian, # wrap before factorizations or eigensystems to avoid checks
                      diag, # mostly for trust region diagonal manipulation
                      eigen, # for the direct subproblem solver
                      opnorm, # for NWI safe guards
                      checksquare, UpperTriangular, givens, lmul!, cond, # For QR update
                      axpy! # for Anderson
import LinearAlgebra: mul!, dot # need to extend for preconditioners

# For better random number generators and rand!
using RandomNumbers

using Printf

function solve end
export solve
"""

"""
function objective_return(f, g, H=nothing)
  if g isa Nothing && H isa Nothing
    return f
  elseif !(g isa Nothing) && H isa Nothing
    return f, g
  elseif !(g isa Nothing) && !(H isa Nothing)
    return f, g, H
  end
end
export objective_return
isallfinite(x) = mapreduce(isfinite, *, x)

using StaticArrays
abstract type MutateStyle end

abstract type AbstractProblem end
abstract type AbstractOptions end

struct InPlace <: MutateStyle end
struct OutOfPlace <:MutateStyle end
include("precondition.jl")
include("Manifolds.jl")
include("objectives.jl")
include("linearalgebra.jl")
export NonDiffed, OnceDiffed, TwiceDiffed, ScalarObjective

# make this struct that has scheme and approx
abstract type QuasiNewton{T1} end


abstract type HessianApproximation end
struct Inverse <: HessianApproximation end
struct Direct <: HessianApproximation end
export Inverse, Direct
# problem and options types
include("optimize/problem_types.jl")
export OptimizationProblem, OptimizationOptions
# Globalization strategies
# TODO:

# Initial step
abstract type LineSearcher end
include("globalization/linesearches/root.jl")
export Backtracking, Static, HZAW
# step interpolations
export FFQuadInterp

include("globalization/trs_solvers/root.jl")
export NWI, Dogleg, NTR

# Quasi-Newton (including Newton and gradient descent) functionality
include("quasinewton/quasinewton.jl")
export DBFGS, BFGS, SR1, DFP, GradientDescent, Newton, BB, LBFGS, ActiveBox

# Include the actual functions that expose the functionality in this package.
include("optimize/linesearch/linesearch.jl")
export HS, CD, HZ, FR, PRP, VPRP, LS, DY
include("optimize/randomsearch/randomsearch.jl")
export SimulatedAnnealing, PureRandomSearch, APSO
include("optimize/directsearch/directsearch.jl")
export NelderMead
include("optimize/acceleration/root.jl")
export Adam, AdaMax

include("optimize/trustregions/trustregions.jl")
export minimize, minimize!, OptProblem, LineSearch, TrustRegion

include("optimize/projected/root.jl")

include("fixedpoints/root.jl")
export Anderson

include("nlsolve/root.jl")
export NEqProblem, NEqOptions
export InexactNewton, KrylovNEqProblem
export DFSANE
include("nlsolve/acceleration/anderson.jl")

# Forcing Terms
export FixedForceTerm, DemboSteihaug, EisenstatWalkerA, EisenstatWalkerB

function negate(problem::AbstractProblem, A)

end
function negate(::InPlace, A::AbstractArray)
  A .= .-A
  return A
end
function negate(::OutOfPlace, A::AbstractArray)
  -A
end

mstyle(problem::AbstractProblem) = problem.mstyle

"""
    retract(problem, z, x, p [, α])

Move from `x` along the direction `p` (or `α*p` if α is supplied) to a new point on the manifold in `problem` and store it in z if the problem is specified as inplace.  If the problem is inplace and updated `z` is returned, else a new vector is returned.
"""
retract(problem, z, x, p) = _retract(mstyle(problem), _manifold(problem), z, x, p)
retract(problem, z, x, p, α) = _retract(mstyle(problem), _manifold(problem), z, x, p, α)
function _retract(::InPlace, manifold::Manifold, z, x, p)
    retract!(manifold, z, x, p)
    return z
end
function _retract(::OutOfPlace, manifold::Manifold, z, x, p)
    z = retract(manifold, x, p)
    return z
end
# shouldn't have those here
function _retract(::InPlace, manifold::Euclidean, z, x, p, α)
    @. z = x + α*p
    return z
  end
function _retract(::OutOfPlace, manifold::Euclidean, z, x, p, α)
    z = @. x + α*p
    return z
end
function _retract(::InPlace, manifold::Euclidean, z, x, p)
    @. z = x + p
    return z
  end
function _retract(::OutOfPlace, manifold::Euclidean, z, x, p)
    z = @. x + p
    return z
end

"""
    project_tanget(problem, w, x, y)

Project g on the tangent space to the manifold (stored in `problem`) at x and store the result in `w` if the problem is specified as inplace. If the problem is inplace and updated `w` is returned, else a new vector is returned.
"""
function project_tangent(problem::OptimizationProblem, w, x, v)
    project_tanget(mstyle(problem), problem, w, x, v)    
end
project_tanget(::InPlace, problem, w, x, v) = copyto!(w, v)
project_tanget(::OutOfPlace, problem, w, x, v) = v

end # module