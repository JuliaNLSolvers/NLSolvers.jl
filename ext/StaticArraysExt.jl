module StaticArraysExt

using NLSolvers, StaticArrays

import NLSolvers: mutation_style
NLSolvers.mutation_style(x::MArray, ::Nothing) = InPlace()
NLSolvers.mutation_style(x::SArray, ::Nothing) = OutOfPlace()
end