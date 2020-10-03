
"""
    Manifold
A manifold type. The `Manifold` is used to dispatch to different exponential
and logarithmic maps as well as other function on manifold.
"""
abstract type Manifold end

abstract type AbstractRetractionMethod end

"""
    ExponentialRetraction
Retraction using the exponential map.
"""
struct ExponentialRetraction <: AbstractRetractionMethod end

"""
    retract!(M::Manifold, y, x, v, [t=1], [method::AbstractRetractionMethod=ExponentialRetraction()])
Retraction (cheaper, approximate version of exponential map) of tangent
vector `t*v` at point `x` from manifold `M`.
Result is saved to `y`.
Retraction method can be specified by the last argument. Please look at the
documentation of respective manifolds for available methods.
"""
retract!(M::Manifold, y, x, v, method::ExponentialRetraction) = exp!(M, y, x, v)

retract!(M::Manifold, y, x, v) = retract!(M, y, x, v, ExponentialRetraction())

retract!(M::Manifold, y, x, v, t::Real) = retract!(M, y, x, t*v)

retract!(M::Manifold, y, x, v, t::Real, method::AbstractRetractionMethod) = retract!(M, y, x, t*v, method)
struct Euclidean{T<:Tuple} <: Manifold where {T} end

Euclidean(n::Int) = Euclidean{Tuple{n}}()
Euclidean(m::Int, n::Int) = Euclidean{Tuple{m,n}}()

function representation_size(::Euclidean{Tuple{n}}) where {n}
    return (n,)
end

function representation_size(::Euclidean{Tuple{m,n}}) where {m,n}
    return (m,n)
end

@generated manifold_dimension(::Euclidean{T}) where {T} = *(T.parameters...)

exp!(M::Euclidean, y, x, v) = (y .= x .+ v)

log!(M::Euclidean, v, x, y) = (v .= y .- x)

function zero_tangent_vector!(M::Euclidean, v, x)
    fill!(v, 0)
    return v
end

project_point!(M::Euclidean, x) = x

function project_tangent!(M::Euclidean, w, x, v)
    w .= v
    return w
end
retract(M::Euclidean, x, v, t::Real) = x+t*v

function retract(M::Manifold, x, v, method::AbstractRetractionMethod)
    xr = copy(x)
    retract!(M, xr, x, v, method)
    return xr
end

function retract(M::Manifold, x, v)
    xr = copy(x)
    retract!(M, xr, x, v)
    return xr
end

retract(M::Manifold, x, v, t::Real) = retract(M, x, t*v)

retract(M::Manifold, x, v, t::Real, method::AbstractRetractionMethod) = retract(M, x, t*v, method)
