#===============================================================================
This file contains an independent implementation of the excellent  line search
algorithm described in:

	"Line Search Algorithms with Guaranteed Sufficient Decrease"
	Jorge J. Moré and David J. Thuente
    ACM Transactions on Mathematical Software (TOMS) (1994)
	doi: 10.1145/192115.192132

Features:
  - Finite termination
  - Wolfe conditions
  - α stays in minstep/maxstep
Rwquires:
  - c1 ≤ c2
  - ϕ is C1
===============================================================================#

struct MoreThuente{T1, T2} <: LineSearcher
    ratio::T1
	c::T1
	maxiter::T2
	verbose::Bool
end


#===============================================================================
                            Trial value selection
	Notation:
		αc is the minimizer of the cubic that interpolates fl, ft, gl, gt
		αq is the minimizer of the quadratic that interpolates fl, ft, gl
		αs is the minimizer of the quadratic that interpolates fl, gl, gt
===============================================================================#
#===============================================================================
	CASE 1: a) ft > fl

	Compute αc and αq and set αt⁺ = ifelse(|αc-αl|<|αq-αl, αc, (αq+αc)/2).
	This choice places us closer to αl - the step length that gives us the
	lowest function value so far.
===============================================================================#
αt⁺ = abs(αc - αl) < abs(αq - αl) ? αc : (αq + αc)/2
#===============================================================================
	CASE 2: a) ft ≤ fl
	        b) gt×gl < 0
===============================================================================#
αt⁺ = abs(αc - αt) < abs(αs - αt) ? αc : αs
#===============================================================================
	CASE 3: a) ft ≤ fl
            b) gt×gl ≥ 0
            c) |gt| ≤ |gl|

	αt⁺ :  cubic that interpolates fl, ft, gl, gt may not have a minimizer
===============================================================================#
if cubic tend to inf for alpha inc and is αc > αt?
	αt⁺ = ifelse(abs(αc - αt) < abs(αs - αt)), αc, αs
else
	αt⁺ = αs
end
#===============================================================================
	CASE 4: a) ft ≤ fl
	        b) gt×gl ≥ 0
	        c) |gt| < |gl|

	αt⁺ : minimizer of cubic that interpolates fu, ft, gu, gt
===============================================================================#

function find_steplength(mstyle, ls::Union{Backtracking, TwoPointQuadratic}, φ::T, d, x,
	                     λ, #
						 ϕ_0::Tf,
						 ∇f_x,
						 dϕ_0=dot(d, ∇f_x);
						 minlength,
						 maxlength) where {T, Tf}
	ratio, c, maxiter, verbose = Tf(ls.ratio), Tf(ls.c), ls.maxiter, ls.verbose

    if verbose
        println("Entering line search with step size: ", λ)
        println("Initial value: ", ϕ_0)
        println("Value at first step: ", φ(λ))
    end

    t = -c*dϕ_0

    iter, α, β = 0, λ, λ # iteration variables
	f_α = φ(α)   # initial function value

	is_solved = isfinite(f_α) && f_α <= ϕ_0 + c*α*t

    while !is_solved && iter <= maxiter
        iter += 1
        β, α, f_α = _solve_pol(ls, φ, ϕ_0, dϕ_0, α, f_α, ratio)
		is_solved = isfinite(f_α) && f_α <= ϕ_0 + c*α*t
    end

	ls_success = iter >= maxiter ? false : true

    if verbose
		!ls_success && println("maxiter exceeded in backtracking")
        println("Exiting line search with step size: ", α)
        println("Exiting line search with value: ", f_α)
    end
    return α, f_α, ls_success
end
