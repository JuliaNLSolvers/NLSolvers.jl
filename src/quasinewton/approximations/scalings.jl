abstract type QNScaling end

# Skip strategies for positive-definite quasi-Newton updates (BFGS, DFP, L-BFGS).
# These determine when to discard an (s, y) pair to maintain positive definiteness
# of the Hessian approximation. Not applicable to SR1, which has its own safeguard.
abstract type PDSkip end

"""
    NoPDSkip

Never skip the quasi-Newton update (default behavior).
"""
struct NoPDSkip <: PDSkip end

"""
    LBFGSBSkip

Skip the update when `s'y / (-∇f_old' · d) ≤ ε_mach`.

This is the condition used in L-BFGS-B (Zhu, Byrd, Lu, Nocedal 1997).
It guards against numerical breakdown when the curvature `s'y` is at
the level of floating-point noise relative to the directional derivative.
"""
struct LBFGSBSkip <: PDSkip end

function should_skip(::LBFGSBSkip, s, y, dφ0)
    ys = real(dot(y, s))
    ys ≤ eps(real(eltype(s))) * abs(dφ0)
end

"""
    LiFukushimaSkip(; ε = 1e-6)

Skip the update when `s'y / ||s||² < ε · ||∇f_k||`.

Li & Fukushima (SIAM J. Optim. 2001, eq. 2.10). This ensures
global convergence for nonconvex problems, even when the line search
only enforces the Armijo condition.
"""
struct LiFukushimaSkip{T} <: PDSkip
    ε::T
end
LiFukushimaSkip(; ε = 1e-6) = LiFukushimaSkip(ε)

function should_skip(lf::LiFukushimaSkip, s, y, ∇f_norm)
    ys = real(dot(y, s))
    ss = real(dot(s, s))
    ys / ss < lf.ε * ∇f_norm
end

should_skip(::NoPDSkip, args...) = false

# Compute the auxiliary value needed by each skip strategy from the iterate state.
# dφ0 is the directional derivative (∇f'·d), ∇fx is the gradient.
skip_aux(::NoPDSkip, dφ0, ∇fx) = nothing
skip_aux(::LBFGSBSkip, dφ0, ∇fx) = abs(dφ0)
skip_aux(::LiFukushimaSkip, dφ0, ∇fx) = norm(∇fx)

# Extract skip strategy from a scheme; default to NoPDSkip for schemes without one.
qn_skip(scheme) = NoPDSkip()
qn_skip(scheme::BFGS) = scheme.skip
qn_skip(scheme::DFP) = scheme.skip
qn_skip(scheme::LBFGS) = scheme.skip
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
