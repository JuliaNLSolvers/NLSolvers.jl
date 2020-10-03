struct SimulatedAnnealing{Tn, Ttemp} <: RandomSearch
    neighbor::Tn
    temperature::Ttemp
end
Base.summary(::SimulatedAnnealing) = "Simulated Annealing"

"""
# SimulatedAnnealing
## Constructor
```julia
SimulatedAnnealing(; neighbor = default_neighbor, temperature = log_temperature)
```

The constructor takes two keywords:
* `neighbor = a(x_now, [x_candidate])`, a function that generates a new iterate based on the current
* `temperature = b(iteration)`, a function of the current iteration that returns a temperature

## Description
Simulated Annealing is a derivative free method for optimization. It is based on the
Metropolis-Hastings algorithm that was originally used to generate samples from a
thermodynamics system, and is often used to generate draws from a posterior when doing
Bayesian inference. As such, it is a probabilistic method for finding the minimum of a
function, often over a quite large domains. For the historical reasons given above, the
algorithm uses terms such as cooling, temperature, and acceptance probabilities.
"""
SimulatedAnnealing(;neighbor = default_neighbor,
                    temperature = log_temperature) =
  SimulatedAnnealing(neighbor, temperature)

log_temperature(t) = 1 / log(t)^2

function default_neighbor(x_best)
  T = eltype(x_best)
  n = length(x_best)
  return x_best .+ T.(RandomNumbers.randn(n))
end

function solve(prob::OptimizationProblem, x0, method::SimulatedAnnealing, options::OptimizationOptions)
  T = eltype(x0)
  t0 = time()

  x_best = copy(x0)
  f_best = value(prob, x_best)
  x_now = copy(x0)
  f_now = f_best
  f0 = f_now
  temperature = f_best
  iter = 0
  is_converged = converged(method, f_now, options)
  while iter ≤ options.maxiter && !(is_converged)
    iter += 1
    # Determine the temperature for current iteration
    temperature = method.temperature(iter)

    # Randomly generate a neighbor of our current state
    x_candidate = method.neighbor(x_best)

    # Evaluate the cost function at the proposed state
    f_candidate = value(prob, x_candidate)

    if f_candidate <= f_now # this handles non-finite values as well
      # If proposal is superior, we always move to it
      x_now = copy(x_candidate)
      f_now = f_candidate

      # If the new state is the best state yet, keep a record of it
      if f_candidate < f_best
        x_best = copy(x_now)
        f_best = f_now
       end
    else
      # If proposal is inferior, we move to it with probability p
      p = exp(-(f_candidate - f_now) / temperature)
      if RandomNumbers.rand() <= p
        x_now = copy(x_candidate)
        f_now = f_candidate
      end
    end
    is_converged = converged(method, f_now, options)
  end

  ConvergenceInfo(method, (minimizer=x_best, minimum=f_best, f_now=f_now, x_now=x_now, temperature=temperature, f0=f0, iter=iter, time=time()-t0), options)
end
function converged(method::SimulatedAnnealing, fz, options)
  f_converged = false
  f_converged = f_converged || fz ≤ options.f_limit
  f_converged = f_converged || isnan(fz)
end