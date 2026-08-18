"""
    qrdelete!(Q, R, k)
Delete the left-most column of F = Q[:, 1:k] * R[1:k, 1:k] by updating Q and R.
Only Q[:, 1:(k-1)] and R[1:(k-1), 1:(k-1)] are valid on exit.
"""
function qrdelete!(Q::AbstractMatrix, R::AbstractMatrix, k::Int)
    n, m = size(Q)
    m == checksquare(R) || throw(DimensionMismatch())
    1 ≤ k ≤ m || throw(ArgumentError(LazyString("The chosen k must be between 1 and m=",m," where m is the dimension of the Q matrix.")))
    # apply Givens rotations
    for i = 2:k
        g = first(givens(R, i - 1, i, i))
        lmul!(g, R)
        rmul!(Q, g')
    end

    # move columns of R
    @inbounds for j = 1:(k-1)
        for i = 1:(k-1)
            R[i, j] = R[i, j+1]
        end
    end

    Q, R
end

"""
    qradd!(Q, R, v, k)
Replace the right-most column of F = Q[:, 1:k] * R[1:k, 1:k] with v by updating Q and R.
This implementation modifies vector v as well. Only Q[:, 1:k] and R[1:k, 1:k] are valid on
exit.
"""
function qradd!(Q::AbstractMatrix, R::AbstractMatrix, v::AbstractVector, k::Int)
    n, m = size(Q)
    n == length(v) || throw(DimensionMismatch())
    m == checksquare(R) || throw(DimensionMismatch())
    1 ≤ k ≤ m || throw(ArgumentError(LazyString("The chosen k must be between 1 and m=",m,", where m is the dimension of the Q matrix.")))

    @inbounds for i = 1:(k-1)
        q = view(Q, :, i)
        r = dot(q, v)

        R[i, k] = r
        axpy!(-r, q, v)
    end

    @inbounds begin
        d = norm(v)
        R[k, k] = d
        @. Q[:, k] = v / d
    end

    Q, R
end

struct ΔVector{T,V1,V2} <: AbstractVector{T}
    v1::V1
    v2::V2
end

"""
    Δvec(x,y)

Returns a lazy vector, equal to x - y, without allocating.
"""
function Δvec(v1::V1, v2::V2) where {V1<:AbstractVector, V2<:AbstractVector}
    @boundscheck begin
        length(v1) == length(v2) ||
            throw(DimensionMismatch(lazy"vectors must have the same length"))
    end
    T = Base.promote_eltype(v1, v2)
    return ΔVector{T,V1,V2}(v1, v2)
end

Base.size(A::ΔVector) = (length(A.v1),)
Base.length(A::ΔVector) = length(A.v1)
Base.IndexStyle(::Type{<:ΔVector}) = IndexLinear()

@inline function Base.getindex(A::ΔVector, i::Int)
    @boundscheck checkbounds(A.v1, i)
    @inbounds return A.v1[i] - A.v2[i]
end
