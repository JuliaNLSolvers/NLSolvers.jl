module AbstractDifferentiationExt

using NLSolvers
import AbstractDifferentiation as AD

import NLSolvers: ScalarObjective, mutation_style
function NLSolvers.ScalarObjective(autodiff::AD.AbstractBackend, x, style=NLSolvers.mutation_style(x, nothing); f = nothing, g = nothing, fg = nothing, fgh = nothing, h = nothing, hv = nothing, batched_f = nothing, param = nothing)
    sco = ScalarObjective(f, g, fg, fgh, h, hv, batched_f, param)
    _autodiff(autodiff, x, sco, style)
end
function _autodiff(fdad::AD.AbstractBackend, x, so::ScalarObjective, style)
    g_f = (g,x) -> let f = so.f
        g .= AD.gradient(fdad, f, x)[1]
        g
    end

    fg_f = (g, x) -> let f = so.f
        v, G = AD.value_and_gradient(fdad, f, x)
        g .= G[1] # why is G a tuple...
        v, g
    end

    h_f = (H, x) -> let f = so.f
        h = AD.hessian(fdad, f, x)
        H .= h[1]
        H
    end

    fgh_f = (g, h, x) -> let f = so.f    
        v, G, H = AD.value_gradient_and_hessian(fdad, f, x)
        g .= G[1]
        h .= H[1]
        v, g, h
    end

    hv_f = (w, x, v) -> let f = so.f, g = g_f
        h = AD.hessian(fdad, f, x)
        w .= h[1]*v
    end

    ScalarObjective(f = so.f, g = g_f, fg = fg_f, h = h_f, fgh = fgh_f, hv = hv_f, param=so.param)
end
end