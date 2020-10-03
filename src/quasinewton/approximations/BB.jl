struct BB{T1} <: QuasiNewton{T1}
 approx::T1
end
BB(;inverse=false) = BB(inverse ? Inverse() : Direct())
hasprecon(::BB) = NoPrecon()
update!(scheme::BB{<:Direct}, B, s, y) = _bb(scheme, B, s, y)
update(scheme::BB{<:Direct}, B, s, y) = _bb(scheme, B, s, y)
_bb(scheme::BB{<:Direct}, M, s, y) = dot(s, y)/dot(s, s)
