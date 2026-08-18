using Documenter, NLSolvers

makedocs(
    sitename = "NLSolvers.jl",
    modules = [NLSolvers],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://julianlsolvers.github.io/NLSolvers.jl/dev/",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Minimizing a function" => "optimization.md",
            "Solving non-linear equations" => "nonlineareq.md",
            "Callbacks" => "callbacks.md",
        ],
    ],
    doctest = false,
    checkdocs = :none,
)

deploydocs(repo = "github.com/JuliaNLSolvers/NLSolvers.jl.git", devbranch = "master")
