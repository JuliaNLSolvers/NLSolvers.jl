struct BrentMin{T}
    division::T
    detect_flatness::Bool
end
BrentMin(; division = (3 - sqrt(5)) / 2, detect_flatness = false) =
    BrentMin(division, detect_flatness)

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

    f0 = fv = fw = fx = value(prob, x)
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

    return ConvergenceInfo(bm, (;x, f0, minimum=fx, time=time()-time0, iter=brent_iter), options)
end
