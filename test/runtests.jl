using NLSolvers
using Test
using StaticArrays
#using Optim
#using LineSearches
using Printf
using LinearAlgebra: norm, I
import Random
Random.seed!(41234)
include("qa.jl")
include("nlsolve/runtests.jl")
include("optimize/problems.jl")
include("optimize/interface.jl")
include("optimize/preconditioning.jl")
include("optimize/complex.jl")
include("optimize/param.jl")
include("lsqfit/interface.jl")
include("globalization/runtests.jl")
