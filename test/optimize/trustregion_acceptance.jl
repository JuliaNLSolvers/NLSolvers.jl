using Test, NLSolvers, LinearAlgebra

# Self-contained Rosenbrock so the file can also be run standalone
_tra_f(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
function _tra_g!(∇f, x)
    ∇f[1] = -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1])
    ∇f[2] = 200 * (x[2] - x[1]^2)
    return ∇f
end
function _tra_h!(H, x)
    H[1, 1] = 1200 * x[1]^2 - 400 * x[2] + 2
    H[1, 2] = -400 * x[1]
    H[2, 1] = -400 * x[1]
    H[2, 2] = 200
    return H
end
function _tra_fg!(∇f, x)
    _tra_g!(∇f, x)
    return _tra_f(x), ∇f
end
function _tra_fgh!(∇f, H, x)
    _tra_g!(∇f, x)
    _tra_h!(H, x)
    return _tra_f(x), ∇f, H
end

_tra_obj() = ScalarObjective(f = _tra_f, g = _tra_g!, fg = _tra_fg!, fgh = _tra_fgh!, h = _tra_h!)
_tra_prob() = OptimizationProblem(_tra_obj(); inplace = true)

# cos objective: indefinite Hessian near the maximum, so a huge radius makes
# NTR take an enormous boundary step whose model decrease dwarfs the bounded
# actual decrease; R ≈ 0 and the step is rejected. Forces Newton rejections.
_trc_f(x) = cos(x[1]) + cos(x[2])
function _trc_g!(∇f, x)
    ∇f[1] = -sin(x[1])
    ∇f[2] = -sin(x[2])
    return ∇f
end
function _trc_h!(H, x)
    H[1, 2] = 0
    H[2, 1] = 0
    H[1, 1] = -cos(x[1])
    H[2, 2] = -cos(x[2])
    return H
end
function _trc_fg!(∇f, x)
    _trc_g!(∇f, x)
    return _trc_f(x), ∇f
end
function _trc_fgh!(∇f, H, x)
    _trc_g!(∇f, x)
    _trc_h!(H, x)
    return _trc_f(x), ∇f, H
end
_trc_prob() = OptimizationProblem(
    ScalarObjective(f = _trc_f, g = _trc_g!, fg = _trc_fg!, fgh = _trc_fgh!, h = _trc_h!);
    inplace = true,
)

# One iterate! against a huge radius: the full step overshoots massively, so f
# rises (or barely falls versus the model), and the step is rejected. Used by
# several testsets below.
function _tra_forced_rejection(approach, prob, x0)
    objvars = NLSolvers.prepare_variables(prob, approach, copy(x0), copy(x0), nothing)
    qnvars = NLSolvers.QNVars(objvars.z, objvars.z)
    p = copy(objvars.x)
    Bcache =
        NLSolvers.modelscheme(approach) isa NLSolvers.Newton ? copy(objvars.B) : nothing
    Bbefore = copy(objvars.B)
    Δk = 1e6
    out, Δkp1, reject, qn = NLSolvers.iterate!(
        p,
        objvars,
        Δk,
        approach,
        prob,
        NLSolvers.OptimizationOptions(),
        qnvars,
        Bcache,
        false,
    )
    return (
        x0 = x0,
        objvars = out,
        Δk = Δk,
        Δkp1 = Δkp1,
        reject = reject,
        Bbefore = Bbefore,
    )
end

@testset "trust region acceptance" begin
    @testset "tr_acceptance case table" begin
        η = 1e-4
        # regular ratios
        @test NLSolvers.tr_acceptance(1.0, 1.0, η) == (1.0, true)
        @test NLSolvers.tr_acceptance(η, 1.0, η) == (η, true)
        @test NLSolvers.tr_acceptance(η / 2, 1.0, η) == (η / 2, false)
        @test NLSolvers.tr_acceptance(0.0, 1.0, η) == (0.0, false)
        @test NLSolvers.tr_acceptance(-1.0, 1.0, η) == (-1.0, false)
        # non-finite actual reduction
        @test NLSolvers.tr_acceptance(NaN, 1.0, η)[2] == false
        @test NLSolvers.tr_acceptance(-Inf, 1.0, η)[2] == false
        # ratio-based semantics for degenerate model decreases: the sign
        # structure decides, including the deliberate nonmonotone acceptance
        # of uphill steps when the model also predicts an increase
        @test NLSolvers.tr_acceptance(1.0, 0.0, η) == (Inf, true)
        @test NLSolvers.tr_acceptance(0.0, 0.0, η)[2] == false # NaN ratio
        @test NLSolvers.tr_acceptance(-1.0, 0.0, η) == (-Inf, false)
        @test NLSolvers.tr_acceptance(1.0, -1.0, η) == (-1.0, false)
        @test NLSolvers.tr_acceptance(-1.0, -1.0, η) == (1.0, true) # uphill accept
        @test NLSolvers.tr_acceptance(NaN, 0.0, η)[2] == false
        @test NLSolvers.tr_acceptance(1.0, NaN, η)[2] == false
    end

    @testset "rejected step restores the iterate state" begin
        res = _tra_forced_rejection(
            TrustRegion(BFGS(Direct()), Dogleg()),
            _tra_prob(),
            [-1.2, 1.0],
        )
        @test res.reject
        @test res.objvars.z == res.objvars.x
        @test res.objvars.fz == res.objvars.fx
        @test res.objvars.∇fz == res.objvars.∇fx
        @test res.Δkp1 < res.Δk
    end

    @testset "update_reject policy" begin
        # default: the rejected trial still updates the approximation
        res = _tra_forced_rejection(
            TrustRegion(BFGS(Direct()), Dogleg()),
            _tra_prob(),
            [-1.2, 1.0],
        )
        @test res.reject
        @test res.objvars.B != res.Bbefore
        # opt out: B is untouched by a rejected step
        res = _tra_forced_rejection(
            TrustRegion(BFGS(Direct()), Dogleg(); update_reject = false),
            _tra_prob(),
            [-1.2, 1.0],
        )
        @test res.reject
        @test res.objvars.B == res.Bbefore
    end

    @testset "Newton restores the Hessian on rejection" begin
        res = _tra_forced_rejection(TrustRegion(Newton(), NTR()), _trc_prob(), [0.5, 0.5])
        @test res.reject
        Hx = _trc_h!(zeros(2, 2), res.x0)
        @test res.objvars.B == Hx
    end

    @testset "eval_f_first" begin
        fcount = Ref(0)
        gcount = Ref(0)
        cf(x) = (fcount[] += 1; _tra_f(x))
        cg(∇f, x) = (gcount[] += 1; _tra_g!(∇f, x))
        counting_prob() =
            OptimizationProblem(ScalarObjective(f = cf, g = cg); inplace = true)

        # a rejected iteration evaluates f once and g never
        prob = counting_prob()
        approach = TrustRegion(BFGS(Direct()), Dogleg(); eval_f_first = true)
        x0 = [-1.2, 1.0]
        objvars = NLSolvers.prepare_variables(prob, approach, copy(x0), copy(x0), nothing)
        qnvars = NLSolvers.QNVars(objvars.z, objvars.z)
        p = copy(objvars.x)
        Bbefore = copy(objvars.B)
        fcount[] = 0
        gcount[] = 0
        out, Δkp1, reject, qn = NLSolvers.iterate!(
            p,
            objvars,
            1e6,
            approach,
            prob,
            NLSolvers.OptimizationOptions(),
            qnvars,
            nothing,
            false,
        )
        @test reject
        @test fcount[] == 1
        @test gcount[] == 0
        @test out.B == Bbefore

        # a full solve still converges, and never calls g more than f
        fcount[] = 0
        gcount[] = 0
        res = solve(
            counting_prob(),
            [-1.2, 1.0],
            TrustRegion(BFGS(Direct()), Dogleg(); eval_f_first = true),
            OptimizationOptions(),
        )
        @test res.info.minimum < 1e-10
        @test gcount[] <= fcount[]

        # requires a standalone f
        fg_only = OptimizationProblem(ScalarObjective(fg = _tra_fg!); inplace = true)
        @test_throws ArgumentError solve(
            fg_only,
            [-1.2, 1.0],
            TrustRegion(BFGS(Direct()), Dogleg(); eval_f_first = true),
            OptimizationOptions(),
        )
    end

    @testset "integration: rejections occur and solves converge" begin
        saw_rejection = Ref(false)
        cb = info -> begin
            if info.state.rejected
                saw_rejection[] = true
            end
            false
        end
        res = solve(
            _tra_prob(),
            ([-1.2, 1.0], nothing),
            TrustRegion(BFGS(Direct()), Dogleg()),
            OptimizationOptions(callback = cb);
            initial_Δ = 1e6,
        )
        @test saw_rejection[]
        @test res.info.minimum < 1e-10

        for approach in (
            TrustRegion(Newton(), NTR()),
            TrustRegion(Newton(), NWI()),
            TrustRegion(BFGS(Direct()), Dogleg(); update_reject = false),
        )
            res = solve(
                _tra_prob(),
                ([-1.2, 1.0], nothing),
                approach,
                OptimizationOptions();
                initial_Δ = 1e6,
            )
            @test res.info.minimum < 1e-10
        end
    end
end
