module ForwardDiffExt

using NLSolvers, ForwardDiff

import NLSolvers: ScalarObjective, mutation_style
function NLSolvers.ScalarObjective(autodiff::Experimental.ForwardDiffAutoDiff, x, style=NLSolvers.mutation_style(x, nothing); f = nothing, g = nothing, fg = nothing, fgh = nothing, h = nothing, hv = nothing, batched_f = nothing, param = nothing)
    sco = ScalarObjective(f, g, fg, fgh, h, hv, batched_f, param)
    _autodiff(autodiff, x, sco, style)
end
function _autodiff(fdad::Experimental.ForwardDiffAutoDiff, x, so::ScalarObjective, style)
    g_f = (g,x) -> let f = so.f
        chunk = something(fdad.chunk, ForwardDiff.Chunk(x))
        gcfg = ForwardDiff.GradientConfig(f, x, chunk)
        gr_res = DiffResults.DiffResult(zero(eltype(x)), g)

        ForwardDiff.gradient!(gr_res, f, x, gcfg)
        g
    end

    fg_f = (g, x) -> let f = so.f
        chunk = something(fdad.chunk, ForwardDiff.Chunk(x))
        gcfg = ForwardDiff.GradientConfig(f, x, chunk)
        gr_res = DiffResults.DiffResult(zero(eltype(x)), g)

        ForwardDiff.gradient!(gr_res, f, x, gcfg)
        DiffResults.value(gr_res), g
    end

    h_f = (H, x) -> let f = so.f
        chunk = something(fdad.chunk, ForwardDiff.Chunk(x))
            
        H_res = DiffResults.DiffResult(zero(eltype(x)), x*0, H)
        hcfg = ForwardDiff.HessianConfig(f, H_res, x, chunk)
        ForwardDiff.hessian!(H_res, f, x, hcfg)
        H
    end

    fgh_f = (G, H, x) -> let f = so.f
        chunk = something(fdad.chunk, ForwardDiff.Chunk(x))
        
        H_res = DiffResults.DiffResult(zero(eltype(x)), G, H)
        hcfg = ForwardDiff.HessianConfig(f, H_res, x, chunk)
        ForwardDiff.hessian!(H_res, f, x, hcfg)
        
        DiffResults.value(H_res), G, H
    end

    hv_f = (w, x, v) -> let f = so.f, g = g_f
        #=chunk = something(fdad.chunk, ForwardDiff.Chunk(x))
        gcfg = ForwardDiff.GradientConfig(x->sum(g(x*0, x)'*v), x, chunk)
        gr_res = DiffResults.DiffResult(zero(eltype(x)), x*0)

        w .= ForwardDiff.gradient!(gr_res, f, x, gcfg)=#
        w .= ForwardDiff.gradient(x->sum(g_f(x*0, x)'*v), x)
    end

    ScalarObjective(f = so.f, g = g_f, fg = fg_f, h = h_f, fgh = fgh_f, hv = hv_f)
end
end