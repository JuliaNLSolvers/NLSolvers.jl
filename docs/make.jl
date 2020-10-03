using Documenter, NLSolvers

makedocs(
    doctest = false,
    sitename = "NLSolvers.jl"
)

#deploydocs(
#    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-windmill"),
#    repo = "github.com/JuliaNLSolvers/Optim.jl.git",
#    julia = "1.0"
#)