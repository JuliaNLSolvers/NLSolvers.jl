struct BrentMin{T}
    division::T
    evaluate_bounds::Bool
    tiebreak_down::Bool
end
BrentMin(; division = (3 - sqrt(5)) / 2, evaluate_bounds = true, tiebreak_down = true) =
    BrentMin(division, evaluate_bounds, tiebreak_down)

function solve(
    problem::OptimizationProblem,
    approach::BrentMin,
    options::OptimizationOptions,
)
    _solve(problem, approach, options)
end

function _solve(prob, bm::BrentMin, options)
    time0 = time()
    a, b = bounds(prob)
    T = typeof(a)
    t = 1e-8
    c = bm.division
    v = w = x = a + c * (b - a)

    e = d = 0 * x

    # The following variable names are explained on (p. 73)
    #   - x is the point with the lowest objective value observed
    #       with a value of fx. It is possible that more than one value
    #       of the input has returned an objective equal to fx and
    #       in that case the (x, fx) pair is the most recent such pair.
    #   - w is the point with the next lowest objective value of fw
    #   - v is the previous value of w with objective fv
    #   - u is the last point at which the objective was evaluated
    #   - f0 is the first value observed
    fv = fw = fx = value(prob, x)

    # We modify the algorith slightly here, and evaluate the objetive
    # at the bounds initially. If this is not what the user wants, they
    # should set `evaluate_bounds = false`.
    if bm.evaluate_bounds
        fa = value(prob, a)
        fb = value(prob, b)
        if fa < fx || fb < fx
            if fa < fb
                x = a
                fx = fa
            else
                x = b
                fx = fb
            end
        end
    end

    # Set f0 to the (corrected) initial fx
    f0 = fx

    p = q = r = 0 * x
    brent_iter = 0

    while brent_iter < options.maxiter

        brent_iter += 1
        m = (a + b) / 2
        tol = eps(T) * abs(x) + t
        if abs(x - m) > 2 * tol - (b - a) / 2 # stopping crit
            # fit parabola
            if abs(e) > tol
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2 * (q - r)
                if q > 0
                    p = -p
                else
                    q = -q
                end
                r = e
                e = d
            end

            if abs(p) < abs(q * r / 2) && p < q * (a - x) && p < q * (b - x)
                # then do the parapolic interpolation
                d = p / q
                u = x + d
                if u - a < 2 * tol || b - u < 2 * tol
                    d = sign(m-x)*tol
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
                u = x + sign(d) * tol
            end

            fu = value(prob, u)

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
    if bm.evaluate_bounds
        # Even if we stopped it can still be the case that x is not a or b even if they're the lowest.
        # We check if fx is indeed the lowest point. If not, we return fa or fb. We tie break to exact
        # bound.
        if fx >= fa
            if fx >= fb
                if bm.tiebreak_down
                    x = a
                else
                    x = b
                end
            else
                x = a
            end
        elseif fx >= fb
            x = b
        end
    end
    return ConvergenceInfo(bm, (;x, f0, minimum=fx, time=time()-time0, iter=brent_iter), options)
end
