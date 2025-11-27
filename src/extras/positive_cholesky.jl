function positive_linsolve(d, B, ∇f)
    cholesky!(Positive, B)
    Bchol = Cholesky(B, 'L', 0)
    d .= Bchol \ ∇f
end
function positive_linsolve(B, ∇f)
    Bchol = cholesky(Positive, B)
    Bchol \ ∇f
end
function positive_factorize(B)
    cholesky!(Positive, B)
    Cholesky(B, 'L', 0)
end

Base.summary(::Newton{<:Direct,typeof(positive_linsolve)}) =
    "Newton's method with PositiveFactorizations.jl linsolve"
