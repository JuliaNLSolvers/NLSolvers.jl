# make this keyworded?
init(::NEqProblem, ::LineSearch, x) = (z=copy(x), d=copy(x), Fx=copy(x), Jx=x*x')
# the bang is just potentially inplace x and state. nonbang copies these
function solve(problem::NEqProblem, x, method::LineSearch=LineSearch(Newton(), Static(1)), options=NEqOptions(), state=init(problem, method, x))
    t0 = time()

    # Unpack
    scheme, linesearch = modelscheme(method), algorithm(method)
    # Unpack important objectives
    F = problem.R.F
    FJ = problem.R.FJ
    # Unpack state
    z, d, Fx, Jx = state
    z .= x
    T = eltype(Fx)


    # Set up MeritObjective. This defines the least squares
    # objective for the line search.
    merit = MeritObjective(problem, F, FJ, Fx, Jx, d)
    meritproblem = OptimizationProblem(merit, nothing, Euclidean(0), nothing, mstyle(problem), nothing)

    # Evaluate the residual and Jacobian
    Fx, Jx = FJ(Fx, Jx, x)
    if !(scheme.reset_age === nothing)
        JFx = factorize(Jx)
    else
        JFx = nothing
    end
    ρF0, ρ2F0 = norm(Fx, Inf),  norm(Fx, 2)
    age, force_update = 1, false

    stoptol = T(options.f_abstol)# T(options.f_reltol)*ρF0 + T(options.f_abstol)
    if ρF0 < stoptol
        return ConvergenceInfo(method, (solution=x, best_residual=Fx, ρF0=ρF0, ρ2F0=ρ2F0, ρs=T(NaN), iter=0, time=time()-t0), options)
    end

    # Create variable for norms but keep the first ones for printing purposes.
    ρs, ρ2F = ρF0, ρ2F0

    iter = 1
    while iter ≤ options.maxiter
        # Shift z into x
        if mstyle(problem) isa InPlace
            x .= z
        else
            x = copy(z)
        end
        # Update the search direction
        if JFx === nothing
            if mstyle(problem) isa InPlace
                d = scheme.linsolve(d, Jx, -Fx)
            else
                d = scheme.linsolve(Jx, -Fx)
            end
        else
            d = JFx\-Fx
        end

        # Need to restrict to static and backtracking here because we don't allow
        # for methods that calculate the gradient of the line objective.
        #
        # For non-linear systems of equations we choose the sum-of-
        # squares merit function. Some useful things to remember is:
        #
        # f(y) = 1/2*|| F(y) ||^2 =>
        # ∇_df = -d'*J(x)'*F(x)
        #
        # where we remember the notation x means the current iterate and y is any
        # proposal. This means that if we step in the Newton direction such that d
        # is defined by
        #
        # J(x)*d = -F(x) => -d'*J(x)' = F(x)' =>
        # ∇_df = -F(x)'*F(x) = -f(x)*2
        #
        # φ = LineObjective!(F,        ∇fz, z, x, d, fx,        dot(∇fx, d))
        φ = LineObjective(meritproblem, Fx, z, x, d, (ρ2F^2)/2, -ρ2F^2)

        # Perform line search along d
        α, ϕ_out, ls_success = find_steplength(mstyle, linesearch, φ, T(1))
        # Step in the direction α*d
        z = retract(problem, z, x, d, α)

        # Update residual and jacobian
        Fx, Jx, JFx, age = update_model(problem, scheme, Fx, Jx, JFx, z, age, ϕ_out, ls_success, force_update)

        # Update 2-norm for line search conditions: ϕ(0) and ϕ'(0)
        ρ2F = norm(Fx, 2)
        ρF  = norm(Fx, Inf)

        # Update the largest successive change in the iterate
        ρs = mapreduce(x->abs(x[1]-x[2]), max, zip(x,z)) # norm(x.-z, Inf)

        if ρF < stoptol #|| ρs <= 1e-12
            break
        end
        iter += 1
    end
    return ConvergenceInfo(method, (solution=z, best_residual=Fx, ρF0=ρF0, ρ2F0=ρ2F0, ρs=ρs, iter=iter, time=time()-t0), options)
end

function update_model(problem::NEqProblem, scheme::Newton, F, J, JF, z, age, ϕ_out, ls_success, force_update)
    # Update F, J, JF, and age as necessary
    if scheme.reset_age === nothing
        F, J = problem.R.FJ(F, J, z)
        if scheme.factorizer === nothing
            JF = nothing
        else
            JF = scheme.factorizer(J)
        end
        age += 1
    else
        if scheme.reset_age <= age
            if ls_success
                F = ϕ_out.Fx
                J = problem.R.J(J, z)
            else                
                F, J = problem.R.FJ(F, J, z)
            end
            if scheme.factorizer === nothing
                # If we have a reset_age set, we will always want
                # to factorize, because that's what the age refers
                # to: the age of the factorization.
                JF = factorize(J)
            else
                # If the user specified a factorization
                JF = scheme.factorizer(J)
            end
            age = 1
        else
            if ls_success
                F = ϕ_out.Fx
            else
                F = problem.R.F(F, z)
            end
            age += 1
        end
    end
    return F, J, JF, age
end