function move(::InPlace, z, x, s, d, α)
  @. s = α * d
  @. z = x + s
  z, s
end
function move(::OutOfPlace, z, x, s, d, α)
  s = @. α * d
  z = @. x + s
  z, s
end
include("static.jl")
include("backtracking.jl")
include("hagerzhangline.jl")
