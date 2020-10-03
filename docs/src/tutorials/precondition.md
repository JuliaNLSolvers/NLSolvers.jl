# Preconditioners

It is possible to apply preconditioners to improve convergence on ill-conditioned problems. The interface across solvers is the same, but in different contexts, it means slightly different things.

As mentioned above, preconditioning is used to improve the conditioning of the problem at hand. A useful way to think about it is as a change of variables according to `x = S*y`, see [HZSurvey], where `S` is an invertible matrix. Writing the conjugate gradient descent, `ConjugateGradient` in this package, in the variable `y`, and changing it back to `x`, you obtain the equations
```
x' = x + α*d
d' = P*g' + β*d
```
where `P = S*S'` is the preconditioner we will provide as the end-user, and `g` and `d` in the update formulae for β are replaced by `S'g` and `inv(S)*d` respectively. For this to be effective, `P` itself should represent the inverse of the actual Hessian of the objective function. 

The methods that accept preconditioners accept a `P` keyword in their constructors, for example
```julia
function precon(x, P=nothing)
  if P isa Nothing
    return InvDiagPrecon([1.0, 1.0, 1.0])
  else
    P.diag .= [1.0, 1.0, 1.0]
    return P
  end 
end
BFGS(; P=precon)
```
Here, we're using the `InvDiagPrecon` preconditioner provided by this package. The preconditioner is applied using `ldiv!(Pg, P, g)` for in-place algorithms, and
`Pg = Pg\g` for out-of-oplace algorithms. Otherwise, unless a custom type is provided and methods are defined, it is applied using `ldiv!(Pg, factorize(P), g)` for in-place algorithms. Additionally, `ConjugateGradient` requires `dot(x, P, y)` to be defined for the preconditioner type as well. 