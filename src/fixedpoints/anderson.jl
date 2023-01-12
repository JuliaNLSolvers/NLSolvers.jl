#===============================================================================
  Anderson acceleration is a fixed point iteration acceleration method. A fixed
  point problem is one of finding a solution to g(x) = x or alternatively to
  find a solution to F(x) = g(x) - x = 0. To allow for easy switching between
  methods, we write the code in the F(x) = 0 form, but it really is of no impor-
  tance. This makes it a bit simpler to keep a consistent naming scheme as well,
  since convergence is measured in F(x) not g(x) directly.
===============================================================================#
struct Anderson{Ti,Tb,Td}
    delay::Ti
    memory::Ti
    beta::Tb
    droptol::Td
end
Anderson() = Anderson(0, 5, nothing, nothing)
function vv_shift!(G)
    for i = 1:length(G)-1
        G[i] = G[i+1]
    end
end

# args
function fixedpoint!(
    g,
    x,
    anderson::Anderson;
    # kwargs
    Gx = similar(x),
    Fx = similar(x),
    f_abstol = sqrt(eps(eltype(x))),
    maxiter = 100,
)
options = NEqOptions(maxiter=maxiter)
    #==============================================================
      Do initial function iterations; default is a delay of 0,
      but we always do at least one evaluation of G, to set up AA.

      Notation: AA solves g(x) - x = 0 or F(x) = 0
    ==============================================================#
    t0 = time()
    Gx = g(Gx, x)
    Fx .= Gx .- x
    ρF0 = norm(Fx, Inf)
    ρ2F0 = norm(Fx, 2)
    x .= Gx
    delay_iter = 0
    while delay_iter < anderson.delay
        delay_iter += 1
        Gx = g(Gx, x)
        Fx .= Gx .- x
        x .= Gx
        finite_check = isallfinite(x)
        if norm(Fx) < f_abstol || !finite_check
                return ConvergenceInfo(
                    anderson,
                    (
                        solution = x,
                        best_residual = Fx,
                        ρF0,
                        ρ2F0,
                        delay_iter=delay_iter,
                        iter=0,
                        time = time() - t0,
                    ),
                    options,
                )
        end
    end

    #==============================================================
      If we got this far, then the delay was not enough to con-
      verge. However, we now hope to have moved to a region where
      everything is well-behaved, and we start the acceleration.
    ==============================================================#

    n = length(x)
    memory = min(n, anderson.memory)

    Q = x * x[1:memory]'
    R = x[1:memory] * x[1:memory]'

    #==============================================================
      Start Anderson Acceleration. We use QR updates to add new
      successive changes in G to the system, and once the memory
      is exhausted, we use QR downdates to forget the oldest chan-
      ges we have stored.
    ==============================================================#
    effective_memory = 0
    beta = anderson.beta

    G = [copy(x) for i = 1:memory]
    Δg = copy(Gx)
    Δf = copy(Fx)

    Gold = copy(Gx)
    Fold = copy(Fx)
    γv0 = zeros(effective_memory)
    iter = 0
    while iter < options.maxiter
        iter += 1
        Gx = g(Gx, x)
        Fx .= Gx .- x
        x .= Gx

        # is this actually needed? I think we can avoid these
        @. Δg = Gx - Gold
        @. Δf = Fx - Fold

        Gold .= Gx
        Fold .= Fx

        effective_memory += 1
        # if we've exhausted the memory, downdate
        if effective_memory > memory
            vv_shift!(G)
            qrdelete!(Q, R, memory)
            effective_memory -= 1
        end

        # Add the latest change to G
        G[effective_memory] .= Δg

        # QR update
        qradd!(Q, R, vec(Δf), effective_memory)

        # Create views for the system depending on the effective memory counter
        Qv = view(Q, :, 1:effective_memory)
        Rv = UpperTriangular(view(R, 1:effective_memory, 1:effective_memory))

        # check condition number
        if !isa(anderson.droptol, Nothing)
            while cond(Rv) > anderson.droptol && effective_memory > 1
                qrdelete!(Q, R, effective_memory)

                effective_memory -= 1
                Qv = view(Q, :, 1:effective_memory)
                Rv = UpperTriangular(view(R, 1:effective_memory, 1:effective_memory))
            end
        end

        # solve least squares problem
        γv = view(γv0, 1:effective_memory)
        ldiv!(Rv, mul!(γv, Qv', vec(Fx)))

        # update next iterate
        for i = 1:effective_memory
            @. x -= γv[i] * G[i]
        end
        if !isa(beta, Nothing)
            x .= x .- (1 .- beta) .* (Fx .- Qv * Rv * γv) # this is suboptimal!
        end

        finite_check = isallfinite(x)
        if norm(Fx) < f_abstol || !finite_check
            break#return (x = x, Fx = Fx, acc_iter = 0, finite = finite_check)
        end
    end
    ConvergenceInfo(
        anderson,
        (
            solution = x,
            best_residual = Fx,
            ρF0,
            ρ2F0,
            iter = iter + delay_iter,
            time = time() - t0,
        ),
        options,
    )
end
