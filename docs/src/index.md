# NLSolvers.jl

Univariate and multivariate optimization and equation solving in Julia.

NLSolvers.jl is the backend code for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) v2.0.0 and higher.

## How

The package is a registered package, and can be installed with `Pkg.add`.

```julia
julia> using Pkg; Pkg.add("OptimNLSolvers")
```
or through the `pkg` REPL mode by typing
```
] add NLSolvers
```

## What
NLSolvers.jl is a Julia package for optimization, curve fitting and
systems of nonlinear equations. The package requires full specification
of the problem, so no smart constructors, automatic differentiation, or
other user friendly features are present. If the focus is on ease of use,
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) is the package to
use.

## Citing the package
If you use NLSolvers.jl or Optim.jl in your work, please cite
consider citing our paper in the Journal of Open Source Software. Citations
give us the possibility to document the usage of the package, but it also
gives us a way of following all the exciting ways in which NLSolvers.jl and
Optim.jl are used in many fields fields including, but not limited to, optimization
methods and software, economics, optics, physics, machine learning and more.

[![JOSS](http://joss.theoj.org/papers/10.21105/joss.00615/status.svg)](https://doi.org/10.21105/joss.00615) [![SCHOLAR](https://img.shields.io/badge/google-scholar-informational)](https://scholar.google.com/scholar?cluster=8284109979069908974&hl=en&as_sdt=2005)

```
@article{mogensen2018optim,
  title={Optim: A mathematical optimization package for Julia},
  author={Mogensen, Patrick Kofod and Riseth, Asbj{\o}rn Nilsen},
  journal={Journal of Open Source Software},
  volume={3},
  number={24},
  year={2018},
  publisher={Open Journals}
}
```
