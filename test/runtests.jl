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
include("optimize/geometry_types.jl")
include("optimize/testproblems.jl")
include("optimize/interface.jl")
include("optimize/inference.jl")
include("optimize/evalcounts.jl")
include("optimize/preconditioning.jl")
include("optimize/skip_strategies.jl")
include("optimize/complex.jl")
include("optimize/param.jl")
include("lsqfit/interface.jl")
include("globalization/runtests.jl")
include("optimize/callbacks.jl")
include("optimize/dogleg.jl")
include("optimize/trustregion_acceptance.jl")
include("optimize/trustregion_options.jl")
include("optimize/activebox.jl")
include("lazydiffs.jl")
include("nlsolve/runtests.jl")
# last: reseeds the global RNG, so nothing may consume the stream after it
include("optimize/mixed_tests.jl")
