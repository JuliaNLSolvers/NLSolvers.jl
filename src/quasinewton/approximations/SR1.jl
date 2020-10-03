# make QuasiNewton{SR1, approx} callable and then pass that as the
# "update" parameter
# the minimize can take a single SR1 input as -> QuasiNewton{SR1, Approx}
struct SR1{T1} <: QuasiNewton{T1}
   approx::T1
end
SR1() = SR1(Direct())
hasprecon(::SR1) = NoPrecon()

summary(::SR1) = "SR1"
function update(scheme::SR1{<:Inverse}, H, s, y)
   T = real(eltype(s))
   w = s - H*y
   θ = dot(w, y) # angle between residual and change in gradient
   ρy = norm(y)
   if abs(θ) ≥ T(1e-12)*norm(w)*ρy && !iszero(ρy)
      H = H + (w*w')/θ
   end
   H
end
function update(scheme::SR1{<:Direct}, B, s, y)
   T = real(eltype(s))
   res = y - B*s # resesidual in secant equation
   θ = dot(res, s) # angle between residual and change in state
   if abs(inv(θ)) ≥ T(1e-12)*norm(res)*norm(s)
      if true #abs(θ) ≥ 1e-12
         B = B + (res*res')/θ
      end
   end
   B
end
function update!(scheme::SR1{<:Inverse}, H, s, y)
   T = real(eltype(s))
   w = s - H*y
   θ = dot(w, y) # angle between residual and change in gradient
   ρy = norm(y)
   if abs(θ) ≥ T(1e-12)*norm(w)*ρy && !iszero(ρy)
      H .= H .+ (w*w')/θ
   end
   H
end
function update!(scheme::SR1{<:Direct}, B, s, y)
   T = real(eltype(s))
   res = y - B*s
   θ = dot(res, s) # angle between residual and change in state
   if abs(inv(θ)) ≥ max(T(1e-12)*norm(res, 2)*norm(s, 2), sqrt(eps(T)))
      if true #abs(θ) ≥ 1e-12
         B .= B .+ (res*res')/θ
      end
   end
   B
end
function update!(scheme::SR1{<:Inverse}, A::UniformScaling, s, y)
   update(scheme, A, s, y)
end
function update!(scheme::SR1{<:Direct}, A::UniformScaling, s, y)
   update(scheme, A, s, y)
end
