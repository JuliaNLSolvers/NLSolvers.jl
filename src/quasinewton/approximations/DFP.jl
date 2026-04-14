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

    H .+= ρ * s * s' - H * (y * y') * H / (y' * H * y)
    H
end
function update!(scheme::DFP{<:Direct}, B, s, y)
    σ = dot(s, y)
    ρ = inv(σ)

    C = (I - ρ * y * s')
    B .= C * B * C' + ρ * y * y'

    B
end
update!(scheme::DFP{<:Inverse}, A::UniformScaling, s, y) = update(scheme, A, s, y)
update!(scheme::DFP{<:Direct}, A::UniformScaling, s, y) = update(scheme, A, s, y)
