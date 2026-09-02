struct DFP{T1,Tskip,Tscaling} <: QuasiNewton{T1}
    approx::T1
    skip::Tskip
    scaling::Tscaling
end
DFP(approx::HessianApproximation) = DFP(approx, NoPDSkip(), ShannoPhua())
DFP(; inverse = true, skip = NoPDSkip(), scaling = ShannoPhua()) =
    DFP(inverse ? Inverse() : Direct(), skip, scaling)
hasprecon(::DFP) = NoPrecon()
qn_scaling(scheme::DFP) = scheme.scaling

summary(dfp::DFP{Inverse}) = "Inverse DFP"
summary(dfp::DFP{Direct}) = "Direct DFP"

# function update!(scheme::DFP, B::Inverse, s, y)
#    B.A = B.A + s*s'/dot(s, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::DFP, B::Direct, s, y)
#    B.A = (I - y*s'/dot(y, s))*B.A*(I - s*y'/dot(y, s)) + y*y'/dot(y, s)
# end

function update(scheme::DFP{<:Inverse}, H, s, y)
    σ = dot(s, y)
    ρ = inv(σ)
    H = H + ρ * s * s' - H * (y * y') * H / (y' * H * y)
    H
end
function update(scheme::DFP{<:Direct}, B, s, y)
    σ = dot(s, y)
    ρ = inv(σ)

    C = (I - ρ * y * s')
    B = C * B * C' + ρ * y * y'
    B
end
function update!(scheme::DFP{<:Inverse}, H, s, y)
    σ = dot(s, y)
    ρ = inv(σ)

    # H .+= ρ*s*s' .- (H*y)*(H*y)'/(y'*H*y), as two rank-1 updates. Written out,
    # H*(y*y')*H is two matrix products for a rank-1 term: O(n^3) work and four
    # n by n temporaries. H is Hermitian, so y'*H is (H*y)'.
    Hy = H * y # vector temporary
    mul!(H, s, s', ρ, true)
    mul!(H, Hy, Hy', -inv(dot(y, Hy)), true)
    return H
end
function update!(scheme::DFP{<:Direct}, B, s, y)
    σ = dot(s, y)
    ρ = inv(σ)

    # The congruence transform C*B*C' + ρ*y*y' with C = I - ρ*y*s' expands, for
    # Hermitian B and b = B*s, into the rank-2 update
    #     B - ρ*y*b' - conj(ρ)*b*y' + (ρ*conj(ρ)*s'Bs + ρ)*y*y'
    # which avoids the two matrix products and the n by n C.
    b = B * s # vector temporary
    sBs = dot(s, b)
    mul!(B, y, b', -ρ, true)
    mul!(B, b, y', -conj(ρ), true)
    mul!(B, y, y', ρ * conj(ρ) * sBs + ρ, true)

    return B
end
update!(scheme::DFP{<:Inverse}, A::UniformScaling, s, y) = update(scheme, A, s, y)
update!(scheme::DFP{<:Direct}, A::UniformScaling, s, y) = update(scheme, A, s, y)
