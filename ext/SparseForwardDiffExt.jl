module SparseForwardDiffExt

using NLSolvers, ForwardDiff, SparseDiffTools, Symbolics

import NLSolvers: ScalarObjective, mutation_style
function NLSolvers.ScalarObjective(autodiff::Experimental.SparseForwardDiff, x, style=NLSolvers.mutation_style(x, nothing); f = nothing, g = nothing, fg = nothing, fgh = nothing, h = nothing, hv = nothing, batched_f = nothing, param = nothing)
    sco = ScalarObjective(f, g, fg, fgh, h, hv, batched_f, param)
    _autodiff(autodiff, x, sco, style)
end
function _autodiff(fdad::Experimental.SparseForwardDiff, x, so::ScalarObjective, style)
    g_f = (g,x) -> let f = so.f
        chunk = something(fdad.chunk, ForwardDiff.Chunk(x))
        g .= ForwardDiff.gradient(f, x)
    end

    fg_f = (g, x) -> let f = so.f
        # brug diffresults her
        chunk = something(fdad.chunk, ForwardDiff.Chunk(x))
        gcfg = ForwardDiff.GradientConfig(f, x, chunk)
        gr_res = DiffResults.DiffResult(zero(eltype(x)), g)

        ForwardDiff.gradient!(gr_res, f, x, gcfg)
        DiffResults.value(gr_res), g
    end

    sparsity_pattern = Symbolics.hessian_sparsity(so.f, x)
    hes = eltype(x).(sparsity_pattern)
    colorvec = matrix_colors(hes)
    h_f = (H, x) -> let f = so.f, sparsity_pattern = sparsity_pattern, hes = hes, colorvec = colorvec
        hescache = ForwardColorHesCache(f, x, colorvec, sparsity_pattern)
        numauto_color_hessian!(H, f, x, hescache)
        H
    end

    fgh_f = (G, H, x) -> let f = so.f
        # brug diffresults her
        chunk = something(fdad.chunk, ForwardDiff.Chunk(x))
        
        H_res = DiffResults.DiffResult(zero(eltype(x)), G, H)
        hcfg = ForwardDiff.HessianConfig(f, H_res, x, chunk)
        ForwardDiff.hessian!(H_res, f, x, hcfg)
        
        DiffResults.value(H_res), G, H
    end

    hv_f = (w, x, v) -> let f = so.f, g = g_f
    numauto_hesvec!(w,f,x,v,
                 ForwardDiff.GradientConfig(f,v),
                 similar(v),
                 similar(v))
    end

    ScalarObjective(f = so.f, g = g_f, fg = fg_f, h = h_f, fgh = fgh_f, hv = hv_f)
end
end