# define initial simplex
abstract type AbstractSimplexer end
struct SSH{T} <: AbstractSimplexer
    L::T
end
function initial_simplex(is::SSH, x0)
    T = eltype(x0)
    L = T(is.L)
    n = length(x0)
    p = (n - 1 + sqrt(n + 1))/(n*sqrt(2))
    q = (sqrt(n + 1) - 1)/(n*sqrt(2))

    S = [copy(x0)]
    for i = 1:n
        vₙ = [j == i ?  x0[j] + L*p : x0[j] + L*q for j = 1:n]
        push!(S, vₙ)
    end
    S
end

struct ABA{T} <: AbstractSimplexer
    L::T
end
function initial_simplex(is::ABA{T}, x0) where T<:Real
    Tx = eltype(x0)
    iss = ABA(x0*Tx(0).+Tx(is.L))
    initial_simplex(iss, x0)
end
function initial_simplex(is::ABA, x0)
    T = eltype(x0)
    L = T.(is.L)
    n = length(x0)

    S = [copy(x0)]
    for i = 1:n
        vₙ = x0 .+ ((j == i)*L[j] for j = 1:n)
        push!(S, vₙ)
    end
    S
end

struct RB <: AbstractSimplexer
    lb
    ub
end
function simplex_rb(x0)
end
#
# struct LPS{A, R} <: AbstractSimplexer
#     abs::A
#     rel::R
# end
# function initial_simplex(is::LPS, x0)
#     T = eltype(x0)
#     small = abs.(x0) .<= sqrt(eps(T))
#
#     ...
# end

nmparameters(t) = t
struct GaoHan{Tx}
    n::Int
    T::Tx
end
function (gh::GaoHan)(t)
    n = gh.n
    T = gh.T
    nT = T(n)
    t .+ (0, 2/nT, inv(2*nT), 1/nT)
end

struct NelderMead{Ts, Tp, Tsi}
    initial_simplex::Ts
    parameters::Tp
    shrink_index::Tsi
end

"""
# NelderMead
## Constructor
```julia
NelderMead(; parameters = nmparameters,
             initial_simplex = AffineSimplex())
```

The constructor takes 2 keywords:

* `parameters`, an instance of either `AdaptiveParameters` or `FixedParameters`,
and is used to generate parameters for the Nelder-Mead Algorithm
* `initial_simplex`, an instance of `AffineSimplexer`

## Description
Our current implementation of the Nelder-Mead algorithm is based on [1] and [3].
Gradient-free methods can be a bit sensitive to starting values and tuning parameters,
so it is a good idea to be careful with the defaults provided in Optim.jl.

Instead of using gradient information, Nelder-Mead is a direct search method. It keeps
track of the function value at a number of points in the search space. Together, the
points form a simplex. Given a simplex, we can perform one of four actions: reflect,
expand, contract, or shrink. Basically, the goal is to iteratively replace the worst
point with a better point. More information can be found in [1], [2] or [3].

## References
- [1] Nelder, John A. and R. Mead (1965). "A simplex method for function minimization". Computer Journal 7: 308–313. doi:10.1093/comjnl/7.4.308
- [2] Lagarias, Jeffrey C., et al. "Convergence properties of the Nelder–Mead simplex method in low dimensions." SIAM Journal on Optimization 9.1 (1998): 112-147
- [3] Gao, Fuchang and Lixing Han (2010). "Implementing the Nelder-Mead simplex algorithm with adaptive parameters". Computational Optimization and Applications. doi:10.1007/s10589-010-9329-3
"""
function NelderMead()
    return NelderMead(0, x->(1.0, 2.0, 0.5, 0.5), nothing)
end
Base.summary(::NelderMead) = "Nelder-Mead"
function solve(prob::OptimizationProblem, x0, method::NelderMead, options::OptimizationOptions)
    solve(mstyle(prob), prob, x0, method, ABA(x0.*0 .+ 1), options)
end

# centroid except h-th vertex
function centroid!(c, simplex_vector, h=0)
    T = eltype(c)
    n = length(c)
    fill!(c, zero(T))
    @inbounds for i in 1:n+1
        if i != h
            xi = simplex_vector[i]
            c .+= xi
        end
    end
    c .= c ./ n
end

centroid(simplex_vector, h) = centroid!(similar(simplex_vector[1]), simplex_vector, h)
using Statistics
function nmobjective(y::Vector)
    a = sqrt(var(y) * (length(y) / (length(y) - 1)))
    return a
end
struct ValuedSimplex{TS, TV, TO}
    S::TS
    V::TV
    O::TO
end
ValuedSimplex(S, V) = ValuedSimplex(S, V, sortperm(V))

function solve(mstyle::InPlace, prob::OptimizationProblem, x0, method::NelderMead, as::AbstractSimplexer, options)
    simplex_vector = initial_simplex(as, x0)
    simplex_value = batched_value(prob, simplex_vector) # this could be batched
    order = sortperm(simplex_value)
    simplex = ValuedSimplex(simplex_vector, simplex_value, order)
    res = solve(mstyle, prob, simplex, method, options)
    x0 .= res.info.minimizer
    return res
end

function NMCaches(simplex)
    x_reflect = copy(first(simplex.S))
    x_cache = copy(first(simplex.S))
    x_centroid = copy(x_cache)
    return (x_reflect=x_reflect, x_cache=x_cache, x_centroid=x_centroid)
end

function solve(mstyle::InPlace, prob::OptimizationProblem, simplex::ValuedSimplex, method::NelderMead, options::OptimizationOptions, nmcache=NMCaches(simplex))
    t0 = time()
    simplex_vector, simplex_value, i_order = simplex.S, simplex.V, simplex.O
    f0 = minimum(simplex.V)
    n = length(first(simplex_vector))
    m = length(simplex_vector)

    # We only need two caches
    x_reflect = nmcache.x_reflect
    x_cache = nmcache.x_cache

    x_centroid = centroid!(nmcache.x_centroid, simplex_vector, i_order[end])

    α, β, γ, δ = method.parameters(n)
    step_type = "none"
    iter = 0
    nm_obj = f0
    is_converged = false
    while iter <= options.maxiter && !any(is_converged)
        iter += 1
        nm_obj, x_centroid = iterate!(prob, method, simplex_vector, simplex_value, i_order, x_cache, x_centroid, x_reflect, α, β, γ, δ)
        print_trace(method, options, iter, t0, simplex_value)
        if nm_obj ≤ options.nm_tol
            is_converged = true
        end
    end
    f_centroid_min = value(prob, x_centroid)
    f_min, i_f_min = findmin(simplex_value)
    x_min = simplex_vector[i_f_min]
    if f_centroid_min < f_min
        x_min = x_centroid
        f_min = f_centroid_min
    end
    ConvergenceInfo(method, (nm_obj=nm_obj, centroid=x_centroid, simplex=simplex, minimizer=x_min, minimum=f_min, f0=f0, iter=iter, time=time()-t0), options)
end
function print_trace(::NelderMead, options, iter, t0, simplex_value)
    if !isa(options.logger, NullLogger) 
    end
end

function iterate!(prob, method::NelderMead, simplex_vector, simplex_value, i_order, x_cache, x_centroid, x_reflect, α, β, γ, δ)
    # Augment the iteration counter
    shrink = false
    n = length(first(simplex_vector))
    m = length(simplex_vector)

    # Compute a reflection
    x_highest = simplex_vector[i_order[end]]
    @. x_reflect .= x_centroid + α * (x_centroid - x_highest)

    f_reflect = value(prob, x_reflect)
    if f_reflect < simplex_value[i_order[1]] # f_lowest(simplex)
        # Compute an expansion
        x_expand = x_cache
        @. x_expand .= x_centroid + β * (x_reflect - x_centroid)

        f_expand = value(prob, x_expand)

        if f_expand < f_reflect
            simplex_vector[i_order[end]] .= x_expand
            simplex_value[i_order[end]] = f_expand
            step_type = "expansion"
        else
            simplex_vector[i_order[end]] .= x_reflect
            simplex_value[i_order[end]] = f_reflect
            step_type = "reflection"
        end
    elseif f_reflect < simplex_value[i_order[end - 1]]  # f_second_highest(simplex)
        simplex_vector[i_order[end]] .= x_reflect
        simplex_value[i_order[end]] = f_reflect
        step_type = "reflection"
    else
        if f_reflect < simplex_value[i_order[end]] # f_highest(simplex)
            # Outside contraction
            x_contract = x_cache
            @. x_contract .= x_centroid + γ * (x_reflect-x_centroid)

            f_outside_contraction = value(prob, x_contract)
            if f_outside_contraction < f_reflect
                simplex_vector[i_order[end]] .= x_contract
                simplex_value[i_order[end]] = f_outside_contraction
                step_type = "outside contraction"
            else
                shrink = true
            end
        else # f_reflect > f_highest
            # Inside contraction
            x_inside_contract = x_cache
            @. x_inside_contract .= x_centroid - γ *(x_reflect - x_centroid)
            f_inside_contraction = value(prob, x_inside_contract)

            if f_inside_contraction < simplex_value[i_order[end]] #  f_highest(simplex)
                simplex_vector[i_order[end]] .= x_inside_contract
                simplex_value[i_order[end]] = f_inside_contraction
                step_type = "inside contraction"
            else
                shrink = true
            end
        end
    end

    if shrink
        if isa(method.shrink_index, Nothing)
            # shrink all vertices indexed 2 and up
            low_range_index = 2
        elseif isa(method.shrink_index, Number)
            # only shrink the vertices indexed shrink_index and up
            low_range_index = method.shrink_index
        else
            # shrink the indeces returned by shrink_index ( could be random)
            low_range_index = method.shrink_index(m)
        end
        shrink_range = low_range_index:m
        for i = shrink_range
            ord = i_order[i]
            x_lowest = simplex_vector[i_order[1]]
            @. simplex_vector[ord] .= x_lowest + δ*(simplex_vector[ord] - x_lowest)
            simplex_value[ord] = value(prob, simplex_vector[ord])
        end
        step_type = "shrink"
    end

    sortperm!(i_order, simplex_value)

    x_centroid = centroid!(x_centroid, simplex_vector, i_order[end])
    nm_obj = nmobjective(simplex_value)

    nm_obj, x_centroid
end

#####################
#    out-of-place   #
#####################
function solve(mstyle::OutOfPlace, prob::OptimizationProblem, x0, method::NelderMead, as::AbstractSimplexer, options::OptimizationOptions)
    simplex_vector = initial_simplex(as, copy(x0))
    simplex_value = batched_value(prob, simplex_vector) # this could be batched
    order = sortperm(simplex_value)
    simplex = ValuedSimplex(simplex_vector, simplex_value, order)
    res = solve(mstyle, prob, simplex, method, options)
    return res
end
function solve(mstyle::OutOfPlace, prob::OptimizationProblem, simplex::ValuedSimplex, method::NelderMead, options::OptimizationOptions)
    t0 = time()
    simplex_vector, simplex_value = simplex.S, simplex.V
    n = length(first(simplex_vector))
    m = length(simplex_vector)
    f0 = minimum(simplex.V)
    # Get the indices that correspond to the ordering of the f values
    # at the vertices. i_order[1] is the index in the simplex of the vertex
    # with the lowest function value, and i_order[end] is the index in the
    # simplex of the vertex with the highest function value
    i_order = sortperm(simplex_value)

    x_centroid = centroid(simplex_vector, i_order[end])

    α, β, γ, δ = method.parameters(n)
    step_type = "none"
    is_converged = false
    iter = 0
    nm_obj = f0

    while iter <= options.maxiter && !any(is_converged)
        iter += 1

        # Augment the iteration counter
        shrink = false

        # Compute a reflection
        x_highest = simplex_vector[i_order[end]]
        x_reflect = @. x_centroid + α * (x_centroid - x_highest)

        f_reflect = value(prob, x_reflect)
        if f_reflect < simplex_value[i_order[1]] # f_lowest(simplex)
            # Compute an expansion
            x_expand = @. x_centroid + β * (x_reflect - x_centroid)

            f_expand = value(prob, x_expand)

            if f_expand < f_reflect
                simplex_vector[i_order[end]] = x_expand
                simplex_value[i_order[end]] = f_expand
                step_type = "expansion"
            else
                simplex_vector[i_order[end]] = x_reflect
                simplex_value[i_order[end]] = f_reflect
                step_type = "reflection"
            end
        elseif f_reflect < simplex_value[i_order[end - 1]]  # f_second_highest(simplex)
            simplex_vector[i_order[end]] = x_reflect
            simplex_value[i_order[end]] = f_reflect
            step_type = "reflection"
        else
            if f_reflect < simplex_value[i_order[end]] # f_highest(simplex)
                # Outside contraction
                x_contract = @. x_centroid + γ * (x_reflect-x_centroid)

                f_outside_contraction = value(prob, x_contract)    
                if f_outside_contraction < f_reflect
                    simplex_vector[i_order[end]] = x_contract
                    simplex_value[i_order[end]] = f_outside_contraction
                    step_type = "outside contraction"
                else
                    shrink = true
                end
            else # f_reflect > f_highest
                # Inside contraction
                x_inside_contract = @. x_centroid - γ *(x_reflect - x_centroid)

                f_inside_contraction = value(prob, x_inside_contract)
                if f_inside_contraction < simplex_value[i_order[end]] #  f_highest(simplex)
                    simplex_vector[i_order[end]] = x_inside_contract
                    simplex_value[i_order[end]] = f_inside_contraction
                    step_type = "inside contraction"
                else
                    shrink = true
                end
            end
        end

        if shrink
            # only shirk
            if isa(method.shrink_index, Nothing)
                low_range_index = 2
            elseif isa(method.shrink_index, Number)
                low_range_index = method.shrink_index
            else
                low_range_index = method.shrink_index(m)
            end
            shrink_range = low_range_index:m
            for i = shrink_range
                ord = i_order[i]
                x_lowest = simplex_vector[i_order[1]]
                simplex_vector[ord] = @. x_lowest + δ*(simplex_vector[ord] - x_lowest)
            end
            step_type = "shrink"
        end

        sortperm!(i_order, simplex_value)

        x_centroid = centroid(simplex_vector, i_order[end])

        nm_obj = nmobjective(simplex_value)
        # if nm_x < 1e-18
        #     break
        # end
        # check conv
    end
    f_centroid_min = value(prob, x_centroid)
    f_min, i_f_min = findmin(simplex_value)
    x_min = simplex_vector[i_f_min]
    if f_centroid_min < f_min
        x_min = x_centroid
        f_min = f_centroid_min
    end
    ConvergenceInfo(method, (nm_obj=nm_obj, centroid=x_centroid, simplex=simplex, minimizer=x_min, minimum=f_min, f0=f0, iter=iter, time=time()-t0), options)
end
