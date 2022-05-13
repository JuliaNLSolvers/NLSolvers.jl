# Notation:
# λ is the initial step length
# α current trial step length
# β next trial step length
# d is the search direction
# x is the current iterate
# f is the objective
# φ is the line search objective and is a function of the step length only

# This file contains several implementation of what we might call "Backtracking".
# The AbstractBacktracking line searches try to satisfy the Amijo(-Goldstein)
# condition:
#     |f(x + α*d)| < (1-c_1*α)*|f(x)|
# That is: the function should

# As per [Nocedal & Wright, pp. 37] we don't have to think about the curvature
# condition as long as we use backtracking.

abstract type AbstractBacktracking end
abstract type BacktrackingInterp end

"""
  _safe_α(α_candidate, α_curr, c, ratio)

Returns the safeguarded value of α in a Amijo
backtracking line search.

σ restriction 0 < c < ratio < 1
"""
function _safe_α(α_candidate, α_current, decrease=0.1, ratio=0.5)
  α_candidate < decrease*α_current && return decrease*α_current
  α_candidate > ratio*α_current    && return ratio*α_current

  α_candidate # if the candidate is in the interval, just return it
end

"""
  Backtracking(;)

Constructs an object to control the Backtracking line search algorithm. The
algorithm tries out an initial point and then it iteratively tries to find
a good enough step length as measured by the improvement compared to a first
order Taylor approximation around a step length of zero.

The Backtracking constructor takes the following keyword arguments
 - ratio: the ratio between the current trial step length and the next
 - decrease: a parameter that controls when descent is sufficient
 - maxiter: maximum number of times to search for a better step length
 - interp: which type of interpolation to use, if any
 - steprange: the allowed range for steps lengths to be in given as a tuple
 - verbose: the verbosity level

`ratio` controls how much time to spend looking for a large step size. It
is very common to chose 1/2, but others can be chosen. It's 

When chosing parameters, it's important to note that it must be true that
  0 < decrease < ratio

`maxiter` defaults to 26. With no interpolation and with a ratio of 1/2, this
means that we quit when the step size reaches sqrt(eps(Float64)). Provide another
value as appropriate.



"""
struct Backtracking{T1, T2, T3, TR} <: LineSearcher
  ratio::T1
	decrease::T1
	maxiter::T2
	interp::T3
	steprange::TR
	verbose::Bool
end
summary(bt::Backtracking) = "backtracking ("*summary(bt.interp)*")"
Backtracking(; ratio=0.5, decrease=1e-4, maxiter=26,
	           steprange=(lower=0, upper=Inf), interp=FixedInterp(),
	           verbose=false) =
 Backtracking(ratio, decrease, maxiter, interp, steprange, verbose)

struct FixedInterp <: BacktrackingInterp end
summary(fi::FixedInterp) = ("no interp")
struct FFQuadInterp <: BacktrackingInterp end
summary(ffq::FFQuadInterp) = ("quadratic interp")
struct FFFQuadInterp <: BacktrackingInterp end


"""
  interpolate(itp::BacktrackingInterp, ...)

Calculates the minimizer of a polynomial approximation to the line restricted
objective function.
"""
function interpolate(itp::FixedInterp, φ, φ0, dφ0, α, f_α, ratio)
	β = α
	α = ratio*α
	φ_α = φ(α)
	β, α, φ_α
end

# There are many ways to come up with the next step length trial value.
# If we want to interpolate, then we generally need to have some combination
# of values and derivatives of the function we're building an approximation
# for. In the non-linear equations case, a line search algorithm implies that
# some sort of mert function is used. When the squared two norm is used we
# can calculate a cubic interpolation using the "two point" method, and we
# have the two values and the derivative that we need from 
#
# f(α) = ||F(xₙ+αdₙ)||²₂
# f(0) = ||F(xₙ)||²₂
# f'(0) = 2*(F'(xₙ)'dₙ)'F(xₙ) = 2*F(xₙ)'*(F'(xₙ)*dₙ) < 0
#
# if f'(0) >= 0, then dₙ  is not a descent direction for the merit function.
# This can happen for broyden. Notice, that when calculating dₙ using an
# inexact Newton's method, F'(xₙ)*dₙ can be obtained from the final residual.
# In either case, this is stored in φ.dφ0 as should be handled using the LineObjective.

# two-point parabolic
# at α = 0 we know f and f' from F and J as written above
# at αc define
function twopoint(f, f_0, df_0, α, f_α, ratio)
	ρ_lo, ρ_hi = 0.1, ratio
    # get the minimum (requires df0 < 0)
	c = (f_α - f_0 - df_0*α)/α^2

	# p(α) = f0 + df_0*α + c*α^2 is the function
	# we have df_0 < 0. Then if  f_α > f(0) then c > 0
	# by the expression above, and p is convex. Then,
	# we have a minimum between 0 and α at

	γ = -df_0/(2*c) # > 0 by df0 < 0 and c > 0
    # safeguard α
    return max(min(γ, α*ρ_hi), α*ρ_lo) # σs
end

function interpolate(itp::FFQuadInterp, φ, φ0, dφ0, α, f_α, ratio)
	β = α
	α = twopoint(φ, φ0, dφ0, α, f_α, ratio)
	φ_α = φ(α)
	β, α, φ_α
end


"""
    find_steplength(---)

Returns a step length, (merit) function value at step length and success flag.
"""
function find_steplength(mstyle, ls::Backtracking, φ::T, λ) where T
 	#== unpack ==#
	φ0, dφ0 = φ.φ0, φ.dφ0
	Tf = typeof(φ0)
	ratio, decrease, maxiter, verbose = Tf(ls.ratio), Tf(ls.decrease), ls.maxiter, ls.verbose

	#== factor in Armijo condition ==#
    t = -decrease*dφ0

    iter, α, β = 0, λ, λ # iteration variables
	f_α = φ(α) # initial function value

	if verbose
		println("Entering line search with step size: ", λ)
		println("Initial value: ", φ0)
		println("Value at first step: ", f_α)
	end
	is_solved = isfinite(f_α.ϕ) && f_α.ϕ <= φ0 + α*t
    while !is_solved && iter <= maxiter
        iter += 1
        β, α, f_α = interpolate(ls.interp, φ, φ0, dφ0, α, f_α.ϕ, ratio)
		is_solved = isfinite(f_α.ϕ) && f_α.ϕ <= φ0 + α*t
    end

	ls_success = iter >= maxiter ? false : true

    if verbose
		!ls_success && println("maxiter exceeded in backtracking")
        println("Exiting line search with step size: ", α)
        println("Exiting line search with value: ", f_α.ϕ)
    end
    return α, f_α, ls_success
end
