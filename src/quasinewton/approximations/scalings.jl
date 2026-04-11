abstract type QNScaling end
struct ShannoPhua <: QNScaling end # Nocedal & Wright eq. 6.20; Shanno & Phua, Math. Prog. 1978
function (::ShannoPhua)(s, y)
    real(dot(s, y)) / real(dot(y, y))
end
struct InitialScaling{S} <: QNScaling
    scaling::S
end
(is::InitialScaling)(s, y) = is.scaling(s, y)
next(qns::QNScaling) = qns
next(is::InitialScaling) = is.scaling
