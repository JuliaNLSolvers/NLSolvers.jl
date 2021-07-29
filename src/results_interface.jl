solution(conv::ConvergenceInfo) = _solution(conv.solver, conv.info, conv.options)
_solution(solver, info, options) =  info.solution