# Symmetric Rank-1 (SR1) quasi-Newton update.
#
# The update is skipped when the denominator is too small relative to
# the numerator vectors. The safeguard condition [Nocedal & Wright, eq. 6.26]:
#
#   |s'(y - Bs)| ≥ r · ||s|| · ||y - Bs||      (Direct form)
#   |w'y|        ≥ r · ||w|| · ||y||             (Inverse form, w = s - Hy)
#
# with r ≈ 1e-8. When violated, the update is skipped.
struct SR1{T1,Tr} <: QuasiNewton{T1}
    approx::T1
    r::Tr
end
SR1(approx::HessianApproximation) = SR1(approx, 1e-8)
SR1(; inverse = false, r = 1e-8) = SR1(inverse ? Inverse() : Direct(), r)
hasprecon(::SR1) = NoPrecon()

summary(::SR1) = "SR1"

function update(scheme::SR1{<:Inverse}, H, s, y)
    T = real(eltype(s))
    w = s - H * y
    θ = real(dot(w, y))
    if abs(θ) ≥ T(scheme.r) * norm(w) * norm(y)
        H = H + (w * w') / θ
    end
    H
end
function update(scheme::SR1{<:Direct}, B, s, y)
    T = real(eltype(s))
    res = y - B * s
    θ = real(dot(res, s))
    if abs(θ) ≥ T(scheme.r) * norm(res) * norm(s)
        B = B + (res * res') / θ
    end
    B
end
function update!(scheme::SR1{<:Inverse}, H, s, y)
    T = real(eltype(s))
    w = s - H * y
    θ = real(dot(w, y))
    if abs(θ) ≥ T(scheme.r) * norm(w) * norm(y)
        H .= H .+ (w * w') / θ
    end
    H
end
function update!(scheme::SR1{<:Direct}, B, s, y)
    T = real(eltype(s))
    res = y - B * s
    θ = real(dot(res, s))
    if abs(θ) ≥ T(scheme.r) * norm(res) * norm(s)
        B .= B .+ (res * res') / θ
    end
    B
end
function update!(scheme::SR1{<:Inverse}, A::UniformScaling, s, y)
    update(scheme, A, s, y)
end
function update!(scheme::SR1{<:Direct}, A::UniformScaling, s, y)
    update(scheme, A, s, y)
end
