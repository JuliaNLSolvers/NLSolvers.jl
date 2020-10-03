# Dogleg is appropriate here because Jx'*Jx is positive definite, and in that
# case we only need to calculate one newton step. In the secular equation version
# we need repeaed factorizations, and that is not as easy to exploit (but maybe
# there's a good shifted one out there?)
struct NormedResiduals{Tx, Tfx, Tf}
    x::Tx
    Fx::Tfx
    F::Tf
end
function value(nr::NormedResiduals, x)
    Fx = nr.F.F(x, nr.Fx)
  
    # this is just to grab them outside, but this hsould come from the convergence info perhaps?
    nr.x .= x
    nr.Fx .= Fx

    f = (norm(Fx)^2)/2
    return f
end
function upto_gradient(nr::NormedResiduals, Fx, x)
    Fx, Jx = nr.F.FJ(Fx, Fx*x', x)

    # this is just to grab them outside, but this hsould come from the convergence info perhaps?
    nr.x .= x
    nr.Fx .= Fx

    f = (norm(Fx)^2)/2
    Fx .= Jx'*Fx
   return f, Fx
end
function upto_hessian(nr::NormedResiduals, Fx, Jx, x)  #Fx is the gradient and Jx is the Hessian
    Fx, Jx = nr.F.FJ(nr.Fx, Jx, x)

    # this is just to grab them outside, but this hsould come from the convergence info perhaps?
    nr.x .= x
    nr.Fx .= Fx

    f = (norm(Fx)^2)/2
    # this is the gradient
    Fx .= Jx'*Fx
    # As you may notice, this can be expensive... Because the gradient
    # is going to be very simple. May want to create a
    # special type or way to hook into trust regions here. We can exploit
    # that we only need the cauchy and the newton steps, not any shifted
    # systems. There is no need to get the full hessian. because these two
    # steps are don't need these multiplies
    # This is the Hessian
    Jx .= Jx'*Jx
    return f, Fx, Jx
end

function solve(prob::NEqProblem, x, approach::TrustRegion{<:Union{SR1, DBFGS, BFGS, Newton}, <:Any, <:Any}, options::NEqOptions)
    if !(mstyle(prob) === InPlace())
        throw(ErrorException("solve() not defined for OutOfPlace() with Trustregion for NEqProblem"))
    end
    F = prob.R
    # should we wrap a Fx here so we can log F0 info here?
    # and so we can extract it at the end as well?
    # xcache = copy(x).-1
    Fx_outer = copy(x)
    x_outer = copy(x)

    normed_residual = NormedResiduals(x_outer, Fx_outer, F)
    ρ2F0 = 2*value(normed_residual, x_outer)
    ρF0 = norm(normed_residual.Fx, Inf)
    td = OptimizationProblem(normed_residual)
    res = solve(td, x, approach, OptimizationOptions(maxiter=options.maxiter))
    # normed_residual to get Fx
    # f0*2 is wrong because it's 2norm
    newinfo = (zero=res.info.minimizer, best_residual=Fx_outer, ρF0=ρF0, ρ2F0=ρ2F0, time=res.info.time, iter=res.info.iter)
    return ConvergenceInfo(approach, newinfo, options)
end
