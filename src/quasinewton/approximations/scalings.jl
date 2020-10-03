abstract type QNScaling end
struct ShannoPhua <: QNScaling end # Matrix Conditionig and Nonlinear Optimization, 1978 Math. Prog
function (::ShannoPhua)(s, y)
    real(dot(s, s)) / sum(abs2, s)
end
struct InitialScaling{S} <: QNScaling
	scaling::S
end
(is::InitialScaling)(s, y) = is.scaling(s, y)
next(qns::QNScaling) = qns
next(is::InitialScaling) = is.scaling