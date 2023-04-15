using Documenter, DocumenterMarkdown, NLSolvers

makedocs(
    doctest = false,
    sitename = "NLSolvers.jl",
    pages = ["index.md", "optimization.md",],
    format = Markdown(),
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-windmill"),
    repo = "github.com/JuliaNLSolvers/NLSolvers.jl.git",
    make = () -> run(`mkdocs build`),
    target = "site",
)
