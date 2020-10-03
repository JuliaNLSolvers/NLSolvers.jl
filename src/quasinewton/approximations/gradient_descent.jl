struct GradientDescent{T1, TP} <: QuasiNewton{T1}
    approx::T1
    P::TP
end
GradientDescent() = GradientDescent(Direct(), nothing)
GradientDescent(m) = GradientDescent(m, nothing)
hasprecon(::GradientDescent{<:Any, <:Nothing}) = NoPrecon()
hasprecon(::GradientDescent{<:Any, <:Any}) = HasPrecon()

summary(gr::GradientDescent) = "Gradient Descent"
update!(scheme::GradientDescent, A, s, y) = A
update(scheme::GradientDescent, A, s, y) = A
