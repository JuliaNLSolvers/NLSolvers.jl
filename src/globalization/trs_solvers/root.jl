# [SOREN] TRUST REGION MODIFICATION OF NEWTON'S METHOD
# [N&W] Numerical optimization
# [Yuan] A review of trust region algorithms for optimization
abstract type TRSPSolver end
abstract type NearlyExactTRSP <: TRSPSolver end

include("solvers/NWI.jl")
include("solvers/Dogleg.jl")
include("solvers/NTR.jl")
#include("subproblemsolvers/TRS.jl") just make an example instead of relying onTRS.jl

function tr_return(;λ, ∇f, H, s, interior, solved, hard_case, Δ, m=nothing)
	m = m isa Nothing ? dot(∇f, s) + dot(s, H * s)/2 : m

	(p=s, mz=m, interior=interior, λ=λ, hard_case=hard_case, solved=solved, Δ=Δ)
end

function update_H!(H, h, λ=nothing)
  T = eltype(h)
  n = length(h)
  if !(λ ==T(0))
    for i = 1:n
      @inbounds H[i, i] = λ isa Nothing ? h[i] : h[i] + λ
    end
  end
  H
end
