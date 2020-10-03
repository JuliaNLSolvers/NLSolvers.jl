# should be constriants not projection, projection is a function on the contrsrinats
abstract type Constraints end
struct BoxConstraint{T} <: Constraints
    lower::T
    upper::T
end
abstract type Scaling end
struct NoScaling <: Scaling end
scale(S::NoScaling, gx) = gx
struct ConstantPDScaling{T} <: Scaling
    S::T
end
scale(S::ConstantPDScaling, gx) = S.S\gx

function update_xλ(prob::OptimizationProblem, S::Scaling, λ, x, gx)
    bounds = bounds(prob)
    min.(max.(bounds.lower, x .- λ.*scale(S, gx)), bounds.upper)
end
#update_xλ(p::BoxConstraint, λ, x, gx) = min.(max.(p.lower, x .- λ.*gx), p.upper)

# 5.13 in Kelley IMO
# what's it in Bertsekas?
function sufficient_decrease(fλ, fx, λ, α, x, xλ, gx)
    lhs = fλ
    # We use (22) from Bersekas (1976) to avoid dividing by λ
    rhs = fx - dot(gx, x.-xλ)*α
    lhs <= rhs
end
function sufficient_decrease(fλ, fx, λ, α, gx)
    lhs = fλ
    rhs = fx - norm(gx, 2)*α*λ
    lhs <= rhs
end


function projected_gradient(prob::OptimizationProblem, f, g, x0, sc::Union{Scaling, NoScaling}=NoScaling(); itermax, ls_itermax, β=0.5, α=1e-6, print_trace=true)

    f0, g0 = f(x0), g(x0)
    if norm(x0.-update_xλ(prob, sc, 1, x0, g0), Inf) <= 1e-6
      return x0, f0, g0
    end

    x = x0

	# 5.12 in Kelley IMO
	# what's it in Bertsekas?
	local fx, gx
    for i = 1:itermax
        fx, gx = f(x), g(x)
        if print_trace
	        println("i = $i    fx: $(fx)    |gx|: $(norm(gx,Inf))")
            println("x1 = $(x[1])")
            println("x2 = $(x[2])")
	    end
        if norm(x.-update_xλ(prob, sc, 1, x, gx)) <= 1e-6
            return x, fx, gx
        end
        for j = 1:ls_itermax
        	λ = β^j
        	xλ = update_xλ(P, sc, λ, x, gx)
            fλ = f(xλ)
            if sufficient_decrease(fλ, fx, λ, α, x, xλ, gx)
            	x = xλ
            	break
            end
        end
    end
    x, fx, gx
end