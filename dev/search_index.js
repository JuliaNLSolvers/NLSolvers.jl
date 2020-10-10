var documenterSearchIndex = {"docs":
[{"location":"tutorials/precondition/#Preconditioners","page":"Preconditioners","title":"Preconditioners","text":"","category":"section"},{"location":"tutorials/precondition/","page":"Preconditioners","title":"Preconditioners","text":"It is possible to apply preconditioners to improve convergence on ill-conditioned problems. The interface across solvers is the same, but in different contexts, it means slightly different things.","category":"page"},{"location":"tutorials/precondition/","page":"Preconditioners","title":"Preconditioners","text":"As mentioned above, preconditioning is used to improve the conditioning of the problem at hand. A useful way to think about it is as a change of variables according to x = S*y, see [HZSurvey], where S is an invertible matrix. Writing the conjugate gradient descent, ConjugateGradient in this package, in the variable y, and changing it back to x, you obtain the equations","category":"page"},{"location":"tutorials/precondition/","page":"Preconditioners","title":"Preconditioners","text":"x' = x + α*d\nd' = P*g' + β*d","category":"page"},{"location":"tutorials/precondition/","page":"Preconditioners","title":"Preconditioners","text":"where P = S*S' is the preconditioner we will provide as the end-user, and g and d in the update formulae for β are replaced by S'g and inv(S)*d respectively. For this to be effective, P itself should represent the inverse of the actual Hessian of the objective function. ","category":"page"},{"location":"tutorials/precondition/","page":"Preconditioners","title":"Preconditioners","text":"The methods that accept preconditioners accept a P keyword in their constructors, for example","category":"page"},{"location":"tutorials/precondition/","page":"Preconditioners","title":"Preconditioners","text":"function precon(x, P=nothing)\n  if P isa Nothing\n    return InvDiagPrecon([1.0, 1.0, 1.0])\n  else\n    P.diag .= [1.0, 1.0, 1.0]\n    return P\n  end \nend\nBFGS(; P=precon)","category":"page"},{"location":"tutorials/precondition/","page":"Preconditioners","title":"Preconditioners","text":"Here, we're using the InvDiagPrecon preconditioner provided by this package. The preconditioner is applied using ldiv!(Pg, P, g) for in-place algorithms, and Pg = Pg\\g for out-of-oplace algorithms. Otherwise, unless a custom type is provided and methods are defined, it is applied using ldiv!(Pg, factorize(P), g) for in-place algorithms. Additionally, ConjugateGradient requires dot(x, P, y) to be defined for the preconditioner type as well. ","category":"page"},{"location":"#NLSolvers.jl","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"","category":"section"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"Univariate and multivariate optimization and equation solving in Julia.","category":"page"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"NLSolvers.jl is the backend code for Optim.jl v2.0.0 and higher.","category":"page"},{"location":"#How","page":"NLSolvers.jl","title":"How","text":"","category":"section"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"The package is a registered package, and can be installed with Pkg.add.","category":"page"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"julia> using Pkg; Pkg.add(\"OptimNLSolvers\")","category":"page"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"or through the pkg REPL mode by typing","category":"page"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"] add NLSolvers","category":"page"},{"location":"#What","page":"NLSolvers.jl","title":"What","text":"","category":"section"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"NLSolvers.jl is a Julia package for optimization, curve fitting and systems of nonlinear equations. The package requires full specification of the problem, so no smart constructors, automatic differentiation, or other user friendly features are present. If the focus is on ease of use, Optim.jl is the package to use.","category":"page"},{"location":"#Citing-the-package","page":"NLSolvers.jl","title":"Citing the package","text":"","category":"section"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"If you use NLSolvers.jl or Optim.jl in your work, please cite consider citing our paper in the Journal of Open Source Software. Citations give us the possibility to document the usage of the package, but it also gives us a way of following all the exciting ways in which NLSolvers.jl and Optim.jl are used in many fields fields including, but not limited to, optimization methods and software, economics, optics, physics, machine learning and more.","category":"page"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"(Image: JOSS) (Image: SCHOLAR)","category":"page"},{"location":"","page":"NLSolvers.jl","title":"NLSolvers.jl","text":"@article{mogensen2018optim,\n  title={Optim: A mathematical optimization package for Julia},\n  author={Mogensen, Patrick Kofod and Riseth, Asbj{\\o}rn Nilsen},\n  journal={Journal of Open Source Software},\n  volume={3},\n  number={24},\n  year={2018},\n  publisher={Open Journals}\n}","category":"page"}]
}
