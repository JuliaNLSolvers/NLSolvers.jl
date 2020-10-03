struct DBFGS{T1, T2, T3} <: QuasiNewton{T1}
   approx::T1
   theta::T2
   P::T3
end
DBFGS(approx) = DBFGS(approx, 0.2, nothing)
DBFGS(;inverse=true, theta=0.2) = DBFGS(inverse ? Inverse() : Direct(), theta, nothing)
hasprecon(::DBFGS{<:Any, <:Any, <:Nothing}) = NoPrecon()
hasprecon(::DBFGS{<:Any, <:Any, <:Any}) = HasPrecon()
summary(dbfgs::DBFGS{Inverse}) = "Inverse Damped BFGS"
summary(dbfgs::DBFGS{Direct}) = "Direct Damped BFGS"

function update!(scheme::DBFGS{<:Direct, <:Any, <:Any}, B, s, y)
   # We could write this as
   #     B .+= (y*y')/dot(s, y) - (B*s)*(s'*B)/(s'*B*s)
   #     B .+= (y*y')/dot(s, y) - b*b'/dot(s, b)
   # where b = B*s
   # But instead, we split up the calculations. First calculate the denominator
   # in the first term
   σ = dot(s, y)
   ρ = inv(σ) # scalar
   # Then calculate the vector b
   b = B*s # vector temporary
   sb = dot(s, b)
   if σ ≥ scheme.theta*sb
      θ = 1.0
      # Calculate one vector divided by dot(s, b)
      ρbb = inv(sb)*b
      # And calculate
      B .+= (inv(σ)*y)*y' .- ρbb*b'
   else
      θ = 0.8*sb/(sb-σ)
      r = y*θ + (1-θ)*b
      # Calculate one vector divided by dot(s, b)
      ρbb = inv(dot(s, b))*b
      # And calculate
      B .+= (inv(dot(s, r))*r)*r' .- ρbb*b'
   end
end
function update(scheme::DBFGS{<:Direct, <:Any}, B, s, y)
   # As above, but out of place

   σ = dot(s, y)
   b = B*s
   sb = dot(s, b)
   if σ ≥ scheme.theta*sb
      θ = 1.0
      # Calculate one vector divided by dot(s, b)
      ρbb = inv(sb)*b
      # And calculate
      return B = B .+ (inv(σ)*y)*y' .- ρbb*b'
   else
      θ = 0.8*sb/(sb-σ)
      r = y*θ + (1-θ)*b
      # Calculate one vector divided by dot(s, b)
      ρbb = inv(dot(s, b))*b
      # And calculate
      return B = B .+ (inv(dot(s, r))*r)*r' .- ρbb*b'
   end
end
function update(scheme::DBFGS{<:Inverse, <:Any}, H, s, y)
   σ = dot(s, y)
   ρ = inv(σ)
   #   if isfinite(ρ)
   C = (I - ρ*s*y')
   H = C*H*C' + ρ*s*s'
   #   end
   H
end
function update!(scheme::DBFGS{<:Inverse, <:Any}, H, s, y)
   σ = dot(s, y)
   ρ = inv(σ)

   if isfinite(ρ)
      Hy = H*y
      H .= H .+ ((σ+y'*Hy).*ρ^2)*(s*s')
      Hys = Hy*s'
      Hys .= Hys .+ Hys'
      H .= H .- Hys.*ρ
   end
   H
end

function update!(scheme::DBFGS{<:Inverse, <:Any}, A::UniformScaling, s, y)
   update(scheme, A, s, y)
end
function update!(scheme::DBFGS{<:Direct, <:Any}, A::UniformScaling, s, y)
   update(scheme, A, s, y)
end
