function solve(prob::NEqProblem, x, method::Anderson, options::NEqOptions)
    if !(mstyle(prob) === InPlace())
        throw(ErrorException("solve() not defined for OutOfPlace() with Anderson"))
    end
    function fixedfromnleq(F, x)
        F .= value(prob, F, x) .+ x
    end
    fixedpoint!(fixedfromnleq, x, method;
                # kwargs
                Gx = similar(x),
                Fx = similar(x),
                f_abstol=sqrt(eps(eltype(x))),
                maxiter=options.maxiter)

end
