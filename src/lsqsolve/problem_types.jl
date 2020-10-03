"""
  LsqProblem(residuals)

An LsqProblem ([Non-linear] Least squares Problem), is used to represent the
mathematical problem of finding the smallest possible sum of squared values of
the elements of the residual function. The problem is defined by `residual` which is
an appropriate objective type (for example `NonDiffed`, `OnceDiffed`, ...) for the
types of algorithm to be used.

Options are stored in `options` and are of the `NLsqOptions` type. See more information
about options using `?NLsqOptions`.

The package NLSolversAD.jl adds automatic conversion of problems to match algorithms
that require higher order derivates than provided by the user. It also adds AD
constructors for a target number of derivatives.
"""
struct LsqProblem{R<:ObjWrapper, B, M, C}
  residuals::R
  bounds::B
  manifold::M
  constraints::C
end

struct LsqOptions{Tmi}
  maxiter::Tmi
end
