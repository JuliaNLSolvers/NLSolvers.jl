using Documenter, NLSolvers

makedocs(
    doctest = false,
    sitename = "NLSolvers.jl"
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-windmill"),
    repo = "github.com/JuliaNLSolvers/NLSolvers.jl.git",
)
