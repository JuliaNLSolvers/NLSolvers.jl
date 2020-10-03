# fix eta k cchoice
# fix quadstep?
"""
  HZAW(;)

Implements the approximate Wolfe line search developed in [HZ2005].

theta controls the search in U3 of the paper. It defaults to 0.5, and as
a result uses bisection when (what is U3?)

We tweak the original algorithm slightly, by backtracking into a feasible
region if the original step length results in function values that are not
finite.
"""
struct HZAW{T} <: LineSearcher
  decrease::T
  curvature::T
  θ::T
  γ::T
end
Base.summary(::HZAW) = "Approximate Wolfe Line Search (Hager & Zhang)"
function HZAW(; decrease=0.1, curvature=0.9, theta=0.5, gamma=2/3)
  if !(0 < decrease ≤ curvature)
    println("Decrease constant must be positive and smaller than the curvature condition.")
  end
  if !(curvature < 1)
    println("Curvature constant must be smaller than one.")
  end
  HZAW(decrease, curvature, theta, gamma)
end

function find_steplength(mstyle, hzl::HZAW, φ, c, ϵk=1e-6; maxiter = 100)
  # c = initial(k) but this is done outisde
  T = typeof(φ.φ0)
  ϵk = T(ϵk)
  δ = T(hzl.decrease)
  σ = T(hzl.curvature)
  ρ = T(5)
  φ0, dφ0 = φ.φ0, φ.dφ0
  φc, dφc = φ(c, true)

  # Backtrack into feasible region; not part of original algorithm
  ctmp, c = c, c
  iter = 0
  while !isfinite(φc) && iter <= maxiter
    iter += 1
    # don't use interpolation, this is vanilla backtracking
    ctmp, c, φc = interpolate(FixedInterp(), φ, φ0, dφ0, c, φc, T(1)/10)
  end

  # initial convergence
  # Wolfe conditions
  if δ*dφ0 ≥ (φc-φ0)/c && dφc ≥ σ*dφ0
    return c, φc, true
  end
  # Approximate Wolfe conditions
  if (2*δ-1)*dφ0 ≥ dφc ≥ σ*dφ0 && φc ≤ φ0 + ϵk
    return c, φc, true
  end
  # Set up interval
  a0, b0 = bracket(hzl, c, φ, ϵk, ρ)
  j = 0
  aj, bj  = a0, b0
  # Main loop
  while j < 50
    a, b = secant²(hzl, φ, aj, bj, ϵk)
    if b - a > hzl.γ*(bj - aj)
      c = (a + b)/2
      φc, dφc = φ(c, true)
      a, b = update(hzl, a, b, c, φ, φc, dφc, ϵk)
    end

    aj, bj = a, b
    j += 1
    if _wolfe(φ0, dφ0, c, φc, dφc, δ, σ, ϵk) || _approx_wolfe(φ0, dφ0, c, φc, dφc, δ, σ, ϵk)
      return c, φc, true
    end
  end
  return T(NaN), T(NaN), false
end
_wolfe(φ0, dφ0, c, φc, dφc, δ, σ, ϵk) =  δ*dφ0 ≥ (φc-φ0)/c && dφc ≥ σ*dφ0
_approx_wolfe(φ0, dφ0, c, φc, dφc, δ, σ, ϵk) = (2*δ-1)*dφ0 ≥ dφc ≥ σ*dφ0 && φc ≤ φ0 + ϵk
"""
   _U3

Used to take step U3 of the updating procedure [HZ, p.123]. The other steps
are in update, but this step is separated out to be able to use it in
step B2 of bracket.
"""
function _U3(hzl::HZAW, φ, a::T, b::T, c::T, ϵk) where T
  # verified against paper description [p. 123, CG_DESCENT_851]
  φ0 = φ.φ0
  _a, _b = a, c
  # a)
  searching = true
  j = 1
  while searching && j < 50
    # convex combination of _a and _b; 0.5 implies bisection
    d = (1 - hzl.θ)*_a + hzl.θ*_b
    φd, dφd = φ(d, true)
    if dφd ≥ T(0) # found point of increasing objective; return with upper bound d
      _b = d
      return _a, _b
    else # now dφd < T(0)
      if φd ≤ φ0 + ϵk
        _a = d
      else # φ(d) ≥ φ0 + ϵk
        _b = d
      end
    end
    j += 1
  end
  _a, _b # throw error?
end

function update(hzl::HZ,
                 a::T, b::T, c::T,
                 φ, φc, dφc,
                 ϵk) where {HZ<:HZAW, T}

  # verified against paper description [p. 123, CG_DESCENT_851]
  φ0 = φ.φ0
  #== U0 ==#
  if c ≤ a || c ≥ b # c ∉ (a, b)
    return a, b, (a=false, b=false)
  end
  #== U1 ==#
  if dφc ≥ T(0)
    return a, c, (a=false, b=true)
  else # dφc < T(0)
    #== U2 ==#
    if φc ≤ φ0 + ϵk
      return c, b, (a=true, b=false)
    end
    #== U3 ==#
    a, b = _U3(hzl, φ, a, b, c, ϵk)
    return a, b, (a=a==c, b=b==c)
  end
end
"""
  bracket

Find an interval satisfying the opposite slope condition [OSC] starting from
[0, c] [pp. 123-124, CG_DESCENT_851].
"""
function bracket(hzl::HZAW, c::T, φ, ϵk, ρ) where T
  # verified against paper description [pp. 123-124, CG_DESCENT_851]
  # Note, we know that dφ(0) < 0 since we're accepted that the current step is in a
  # direction of descent.
  φ0 = φ.φ0

  #== B0 ==#
  cj = c
  φcj, dφcj = φ(cj, true)
  # we only want to store a number, so we don't store all iterates
  ci, φi = T(0), φ0

  maxj = 100
  for j = 1 :maxj
    #==================================================
      B1: φ is increasing at c, set b to cj as this is
          an upper bound, since φ is initially decrea-
          sing.
    ==================================================#
    if dφcj ≥ T(0)
      a, b = ci, cj
      return a, b
    else # dφcj < T(0)
      #== B2 : φ is decreasing at cj but function value is sufficiently larger than
      # φ0, use U3 to update. ==#
      if φcj > φ0 + ϵk
        a, b = _U3(hzl, φ, T(0), cj, c, ϵk)
        return a, b
      end
      #== B3 ==#
      # update ci instead of keeping all c's
      if φcj ≤ φ0 + ϵk
        ci = cj
        φci = φcj
      end
      # expand by factor ρ > 0 (shouldn't this be > 1?)
      cj = ρ*cj
      φcj, dφcj = φ(cj, true)
    end
  end
end

function secant(hzl::HZAW, a, dφa, b, dφb)
  # verified against paper description [p. 123, CG_DESCENT_851]
  #(a*dφb - b*dφa)/(dφb - dφa)
  # It has been observed that dφa can be very close to dφb,
  # so we avoid taking the difference
  a/(1 - dφa/dφb) + b/(1 - dφb/dφa)
end
function secant²(hzl::HZAW, φ, a, b, ϵk)
  # verified against paper description [p. 123, CG_DESCENT_851]
  #== S1 ==#
  φa, dφa = φ(a, true)
  φb, dφb = φ(b, true)
  c = secant(hzl, a, dφa, b, dφb)

  φc, dφc = φ(c, true)
  A, B, updates = update(hzl, a, b, c, φ, φc, dφc, ϵk)
  if updates.b # B == c
    #== S2: c is the upper bound ==#
    φB, dφB = φc, dφc
    _c = secant(hzl, b, dφb, B, dφB)
  elseif updates.a # A == c
    #== S3: c is the lower bound ==#
    φA, dφA = φc, dφc
    _c = secant(hzl, a, dφa, A, dφA)
  end
  updates
  if any(updates)
    #== S4.if: c was upper or lower bound ==#
    φ_c, dφ_c = φ(_c, true)
    _a, _b = update(hzl, A, B, _c, φ, φ_c, dφ_c, ϵk)
    return _a, _b
  else
    #== S4.otherwise: c was neither ==#
    return A, B
  end
end
