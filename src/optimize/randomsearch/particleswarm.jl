## References
# - [1] Zhan, Zhang, and Chung. Adaptive particle swarm optimization, IEEE Transactions on Systems, Man, and Cybernetics, Part B: CyberneticsVolume 39, Issue 6 (2009): 1362-1381
struct APSO{Tn, T}
  n_particles::Tn
  limit_search_space::Bool
  elitist_learning::Bool
  c₁::T
  c₂::T
  σmin::T
  σmax::T
end
# defaults for c's are from the flowchart of page 1370, σ's are from the bottom of 1370  
APSO(; n_particles=nothing, limit_search_space=false, elitist_learning=true, c₁=2.0, c₂=2.0, σmin=0.1, σmax=1.0) = 
  APSO(n_particles, limit_search_space, elitist_learning, c₁, c₂, σmin, σmax)
function solve(problem::OptimizationProblem, x0, method::APSO, options::OptimizationOptions)
    if !(mstyle(problem) === InPlace())
        throw(ErrorException("solve() not defined for OutOfPlace() with Anderson"))
    end
    t0 = time()
    n_particles = method.n_particles isa Nothing ? max(length(x0), 5) : method.n_particles
    lower, upper = lowerbounds(problem), upperbounds(problem)

    T, n = eltype(x0), length(x0)

    c₁, c₂, σmin, σmax = T(method.c₁), T(method.c₂), T(method.σmin), T(method.σmax)
    ω = T(9)/10 # see page 1368

    X, V = [copy(x0) for i = 1:n_particles], [x0*T(0) for i = 1:n_particles]
    X_best = [x0*T(0) for i = 1:n_particles]

    Fs, Fs_best = zeros(T, n_particles), zeros(T, n_particles)

    x = copy(x0)
    if method.elitist_learning
      x_learn = copy(x0)
    end
    current_state = 0

  # spread the initial population uniformly over the whole search space
  width = upper .- lower
  for i in 1:n_particles
    X[i] .= lower .+ width .* rand(T)
    X_best[i] .= X[i]
  end

  best_f = T(0)
  swarm_f = best_f
  f0 = swarm_f
  X_best[1] .= x0
  X[1] .= x0
  iter = 0
  while iter <= options.maxiter
    iter += 1
    limit_X!(X, lower, upper)
  	Fs = batched_value(problem, Fs, X)

    if iter == 1
      copyto!(Fs_best, Fs)
      best_f = Base.minimum(Fs)
    end
    best_f = housekeeping!(Fs, Fs_best, X, X_best, x, best_f)
    if iter == 1
        f0 = best_f
    end
    if method.elitist_learning # try to avoid non-global minima
      x_learn .= x
      # Perturb a random dimension and replace the current worst
      # solution in X with x_learn if x_learn presents the new
      # best solution. Else, discard x_learn.
      worst_Fs, i_worst = findmax(Fs)
      random_index = rand(1:n)

      # Eqn (14) - See Section VI-C for a discussion of max and min
      # Could allow for an annealing scheme. Maybe a SAMIN style?
      σlearn = σmax - (σmax - σmin) * iter / options.maxiter
      
      x_learn[random_index] = x_learn[random_index] + width[random_index]*σlearn*randn()
      x_learn[random_index] = max(lower[random_index], min(upper[random_index], x_learn[random_index]))
      
      Fs_learn = value(problem, x_learn)
      if Fs_learn < best_f
        X_best[i_worst] .= x_learn
        X[i_worst]      .= x_learn
        x .= x_learn

        Fs_best[i_worst] = Fs_learn
        Fs[i_worst] = Fs_learn
        best_f = Fs_learn
      end
    end

    # TODO find a better name for _f (look inthe paper, it might be called f there)
    current_state, swarm_f = get_swarm_state(X, Fs, x, current_state)
    ω, c₁, c₂ = update_swarm_params!(c₁, c₂, ω, current_state, swarm_f)
    update_swarm!(X, X_best, x, n, V, ω, c₁, c₂)
  end
  best_f, x
  ConvergenceInfo(method, (swarm_f=swarm_f, X=X, Fs=Fs, minimizer=x, minimum=best_f, f0=f0, iter=iter, time=time()-t0), options)
end

function update_swarm!(X, X_best, best_point, n, V, ω, c₁, c₂)

  Tx = eltype(first(X))
  # compute new positions for the swarm particles
  # FIXME vmax should be 20%
  for i in eachindex(X, X_best)
      for j in 1:n
          r1 = rand(Tx)
          r2 = rand(Tx)
          vx = X_best[i][j] - X[i][j]
          vg = best_point[j] - X[i][j]
          V[i][j] = V[i][j]*ω + c₁*r1*vx + c₂*r2*vg
          X[i][j] = X[i][j] + V[i][j]
      end
    end
end

function get_mu_1(f::Tx) where Tx
    if Tx(0) <= f <= Tx(4)/10
        return Tx(0)
    elseif Tx(4)/10 < f <= Tx(6)/10
        return Tx(5) * f - Tx(2)
    elseif Tx(6)/10 < f <= Tx(7)/10
        return Tx(1)
    elseif Tx(7)/10 < f <= Tx(8)/10
        return -Tx(10) * f + Tx(8)
    else
        return Tx(0)
    end
end

function get_mu_2(f::Tx) where Tx
    if Tx(0) <= f <= Tx(2)/10
        return Tx(0)
    elseif Tx(2)/10 < f <= Tx(3)/10
        return Tx(10) * f - Tx(2)
    elseif Tx(3)/10 < f <= Tx(4)/10
        return Tx(1)
    elseif Tx(4)/10 < f <= Tx(6)/10
        return -Tx(5) * f + Tx(3)
    else
        return Tx(0)
    end
end

function get_mu_3(f::Tx) where Tx
    if Tx(0) <= f <= Tx(1)/10
        return Tx(1)
    elseif Tx(1)/10 < f <= Tx(3)/10
        return -Tx(5) * f + Tx(3)/2
    else
        return Tx(0)
    end
end

function get_mu_4(f::Tx) where Tx
    if Tx(0) <= f <= Tx(7)/10
        return Tx(0)
    elseif Tx(7)/10 < f <= Tx(9)/10
        return Tx(5) * f - Tx(7)/2
    else
        return Tx(1)
    end
end

function get_swarm_state(X, Fs, best_point, previous_state)
    # swarm can be in 4 different states, depending on which
    # the weighing factors c₁ and c₂ are adapted.
    # New state is not only depending on the current swarm state,
    # but also from the previous
    n_particles = length(X)
    n = length(first(X))
    Tx = eltype(first(X))
    f_best, i_best = findmin(Fs)
    d = zeros(Tx, n_particles)
    for i in 1:n_particles
        dd = Tx(0)
        for k in 1:n_particles
            for dim in 1:n
                @inbounds ddd = (X[i][dim] - X[k][dim])
                dd += ddd * ddd
            end
        end
        d[i] = sqrt(dd)
    end
    dg = d[i_best]
    dmin = Base.minimum(d)
    dmax = Base.maximum(d)

    f = (dg - dmin) / max(dmax - dmin, sqrt(eps(Tx)))

    mu = zeros(Tx, 4)
    mu[1] = get_mu_1(f)
    mu[2] = get_mu_2(f)
    mu[3] = get_mu_3(f)
    mu[4] = get_mu_4(f)
    best_mu, i_best_mu = findmax(mu)
    current_state = 0

    if previous_state == 0
        current_state = i_best_mu
    elseif previous_state == 1
        if mu[1] > 0
            current_state = 1
        else
          if mu[2] > 0
              current_state = 2
          elseif mu[4] > 0
              current_state = 4
          else
              current_state = 3
          end
        end
    elseif previous_state == 2
        if mu[2] > 0
            current_state = 2
        else
          if mu[3] > 0
              current_state = 3
          elseif mu[1] > 0
              current_state = 1
          else
              current_state = 4
          end
        end
    elseif previous_state == 3
        if mu[3] > 0
            current_state = 3
        else
          if mu[4] > 0
              current_state = 4
          elseif mu[2] > 0
              current_state = 2
          else
              current_state = 1
          end
        end
    elseif previous_state == 4
        if mu[4] > 0
            current_state = 4
        else
            if mu[1] > 0
                current_state = 1
            elseif mu[2] > 0
                current_state = 2
            else
                current_state = 3
            end
        end
    end
    return current_state, f
end

function update_swarm_params!(c₁, c₂, ω, current_state, f::T) where T

    Δc₁ = T(5)/100 + rand(T) / T(20)
    Δc₂ = T(5)/100 + rand(T) / T(20)

    if current_state == 1
        c₁ += Δc₁
        c₂ -= Δc₂
    elseif current_state == 2
        c₁ += Δc₁ / 2
        c₂ -= Δc₂ / 2
    elseif current_state == 3
        c₁ += Δc₁ / 2
        c₂ += Δc₂ / 2
    elseif current_state == 4
        c₁ -= Δc₁
        c₂ -= Δc₂
    end

    if c₁ < T(3)/2
        c₁ = T(3)/2
    elseif c₁ > T(5)/2
        c₁ = T(5)/2
    end

    if c₂ < T(3)/2
        c₂ = T(5)/2
    elseif c₂ > T(5)/2
        c₂ = T(5)/2
    end

    if c₁ + c₂ > T(4)
        c_total = c₁ + c₂
        c₁ = c₁ / c_total * 4
        c₂ = c₂ / c_total * 4
    end

    ω = 1 / (1 + T(3)/2 * exp(-T(26)/10 * f))
    return ω, c₁, c₂
end

function housekeeping!(Fs, Fs_best, X, X_best, best_point, F)
    for i in eachindex(X, X_best)
        if Fs[i] <= Fs_best[i]
            Fs_best[i] = Fs[i]
            X_best[i] .= X[i]

            if Fs[i] <= F
              	best_point .= X[i]
              	F = Fs[i]
            end
        end
    end
    return F
end

function limit_X!(X, lower, upper)
    # limit X values to boundaries
    for x in X
        x .= min.(max.(x, lower), upper)
    end
    X
end