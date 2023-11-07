N_test = 9
w = rand(N_test)
obj(x, data) = sum((x .- data) .^ 2)
function g!(G, x, data)
    G .= 2 * (x .- data)
    G
end
g(x) = 2x
function h!(H, x, data)
    H .= x * x' + 2 * I
    H
end
param = [1, 2, 3]

sc = ScalarObjective(; f = obj, g = g!, h = h!, param = param)
op = OptimizationProblem(sc)
solve(op, [3.0, 4.0, 4.0], BFGS(), OptimizationOptions())
