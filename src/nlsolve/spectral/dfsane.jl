#===============================================================================
 Robust NonMonotone Line search
 From DF-Sane paper https://www.ams.org/journals/mcom/2006-75-255/S0025-5718-06-01840-0/S0025-5718-06-01840-0.pdf
 This line search doesn't fit the rest of the API so I need to come up with an
 interface for nonmonotone line searches (it has to accept an fbar (or Q in Hager
 Zhang's notion) and a forcing term)
===============================================================================#
struct RNMS end

function find_steplength(rnms::RNMS, φ, φ0, fbar, ηk::T, τmin, τmax) where T
  α₊ = T(1)
  α₋ = α₊
  γ = T(1)/10000
  for k = 1:100
    φα₊ = φ(α₊)
    if φα₊ ≤ fbar + ηk - γ*α₊^2*φ0
      return α₊, φα₊
    end
    φα₋ = φ(-α₋)
    if φα₋ ≤ fbar + ηk - γ*α₋^2*φ0
      return -α₋, φα₋
    end

    # update alpha+ and alpha-
    αt = α₊^2*φ0/(φα₊ + (2*α₊-1)*φ0)
    α₊ = clamp(αt, τmin*α₊, τmax*α₊)

    αt = α₋^2*φ0/(φα₋ + (2*α₋-1)*φ0)
    α₋ = clamp(αt, τmin*α₋, τmax*α₋)
  end
  throw(ErrorException("Line search failed."))
end

function safeguard_σ(σ::T, σmin, σmax, F) where T
  if abs(σ) < σmin || abs(σ) > σmax
    normF = norm(F)
    if normF > T(1)
      σ = T(1)
    elseif T(1)/10^5 ≤ normF ≤ T(1)
      σ = inv(normF)
    elseif normF < T(1)/10^5
      σ = T(10)^5
    end
  end
  return σ
end

struct DFSANE end
function solve(prob::NEqProblem, x0, method::DFSANE, options::NEqOptions)
  if !(mstyle(prob) === InPlace())
    throw(ErrorException("solve() not defined for OutOfPlace() with DFSANE"))
  end

  t0 = time()
  F = prob.R.F
  T = eltype(x0)

  σmin, σmax = 1e-5, 1e5
  τmin, τmax = T(1)/10, T(1)/2
  M = 4
  nexp = 2


  x = copy(x0)
  ρs = norm(x)
  Fx = copy(x0)
  Fx = F(Fx, x)
  y = copy(Fx)
  ρ2F0 = norm(Fx, 2)
  ρF0 = norm(Fx, Inf)
  fx = ρ2F0^nexp
  fvals = [fx]

  abstol = 1e-5
  reltol = 1e-8

  σ₀ = T(1)
  σ = safeguard_σ(σ₀, σmin, σmax, Fx)
  iter = 0
  while iter < options.maxiter
    iter += 1
    z = copy(x) # FIXME
    push!(fvals, fx)
    if length(fvals) > M
      popfirst!(fvals)
    end
    fbar = maximum(fvals)
    d = -σ*Fx
    ηk = ρ2F0/(1+iter)^2
    φ(α) = norm(F(Fx, x.+α.*d))^nexp
    φ0 = fx
    α, φα = find_steplength(RNMS(), φ, φ0, fbar, ηk, τmin, τmax)
    s = α*d
    ρs = norm(s)
    x .= x.+s
    y .= -Fx
    Fx = F(Fx, x)
    y .+= Fx
    ρ2fx = norm(Fx, 2)
    ρfx = norm(Fx ,Inf)
    fx = ρ2fx^nexp
    σ = norm(s)^2/dot(s, y)
    σ = safeguard_σ(σ, σmin, σmax, Fx)
    # use sqrt(length(x))*abs or abstol?
    if ρfx < abstol + reltol*ρF0
      break
    end
  end
  ConvergenceInfo(method, (solution=x, best_residual=Fx, ρF0=ρF0, ρ2F0=ρ2F0, ρs=ρs, iter=iter, time=time()-t0), options)
end
