struct BrentMin{T}
	division::T
end
BrentMin(; division=(3-sqrt(5))/2) = BrentMin(division) 
function optimize(f, a, x::T, b, bm::BrentMin) where T
	t = 0.0000001
    c = bm.division
    v = w = x = a + c*(b-a)
    
    e = d = 0*x

    fv = fw = fx = f(x)
    for i = 1:1000
    	m = (a + b)/2
    	tol = eps(T)*abs(x) + t
    	if abs(x - m) > 2*tol - (b - a)/2 # stopping crit
    		p = q = r = 0
    		# fit parabola
    		if abs(e) > tol
    			r = (x - w)*(fx - fv)
    			q = (x - v)*(fx - fw)
    			p = (x - v)*q - (x - w)*r
    			q = 2*(q - r)
    			if q > 0
    			    p = -p
                else
    			    q = -q
    			end
    			r = e
    			e = d
    		end

    		if abs(p) < abs(q*r/2) && p < q*(a - x) && p < q*(b - x)
    		    # the do the parapolic interpolation
    		    d = p/q
    		    u = x + d
    		    if u - a < 2*tol || b - u < 2*tol
    		    	if x < m # this should just use sign
    		    	    d = tol
    		    	else
    		    	    d = -tol
   		    	    end
	    	    end
		    else
    			# do golden section
    			if x < m
    				e = b - x
				else
    				e = a - x
				end
				d = c * e
			end

			if abs(d) >= tol
			    u = x + d
		    else
		    	if d > 0  # this should just use sign
		    	    u = x + tol
	    	    else
		    		u = x - tol
	    		end
	    	end
	    	fu = f(u)
	    	if fu <= fx
	    		if u < x
	    			b = x
	    		else
	    		    a = x
			    end
		    	v = w
		    	fv = fw
		    	w = x
		    	fw = fx
		    	x = u
		    	fx = fu
	    	else
	    		if u < x
	    		    a = u
	    		else
	    			b = u
				end
				if fu <= fw || w == x
				    v = w
				    fv = fw
				    w = u
				    fw = fu
			    elseif fu <= fv || v == x || v == w
				    v = u
				    fv = fu
		    	end
	    	end
    	else
    		break
    	end
    end
    return x, fx
end

julia> @time optimize(x->-sign(x), -2.0, 0.0, 1.0, BrentMin())
  0.030817 seconds (62.15 k allocations: 3.751 MiB, 99.75% compilation time)
(0.9999998647435788, -1.0)

julia> @time optimize(x->sign(x), -2.0, 0.0, 1.0, BrentMin())
  0.028942 seconds (61.86 k allocations: 3.737 MiB, 99.71% compilation time)
(-0.8541018309932635, -1.0)

