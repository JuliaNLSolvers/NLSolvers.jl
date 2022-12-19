using Documenter, NLSolvers

makedocs(
    doctest = false,
    sitename = "NLSolvers.jl",
    pages = ["index.md", "optimization.md",]
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-windmill"),
    repo = "github.com/JuliaNLSolvers/NLSolvers.jl.git",
)
