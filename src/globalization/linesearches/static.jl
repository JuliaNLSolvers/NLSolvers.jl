"""
    Static

An object that specifies a constant step length.
"""
struct Static{T} <: LineSearcher
    α::T
end

find_steplength(mstyle, ls::Static, φ, λ::T) where {T} = T(ls.α), φ(T(ls.α)), true
