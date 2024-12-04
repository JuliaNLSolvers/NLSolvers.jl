@testset "param test" begin
    N_test = 9
    w = rand(N_test)
    function obji(x, data)
        s = sum((x .- data) .^ 2)
        s
    end
    function g!(G, x, data)
        G .= 2 * (x .- data)
        G
    end
    function h!(H, x, data)
        H .= 2 * I
        H
    end
    param1 = [1, 2, 3]
    sc = ScalarObjective(; f = obji, g = g!, h = h!, param = param1)
    op = OptimizationProblem(sc)
    sol1 = solve(op, [3.0, 4.0, 4.0], BFGS(), OptimizationOptions())
    @test sol1.info.solution ≈ param1

    param2 = [10, 20, 30]
    sc = ScalarObjective(; f = obji, g = g!, h = h!, param = param2)
    op = OptimizationProblem(sc)
    sol2 = solve(op, [3.0, 4.0, 4.0], BFGS(), OptimizationOptions())
    @test sol2.info.solution ≈ param2
end
