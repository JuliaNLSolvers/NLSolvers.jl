using GeometryTypes
@testset "GeometryTypes" begin
    function fu(G, x)
        fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

        if !(G == nothing)
            G1 = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
            G2 = 200.0 * (x[2] - x[1]^2)
            gx = Point(G1, G2)

            return fx, gx
        else
            return fx
        end
    end
    fu(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    f_obj = OptimizationProblem(
        ScalarObjective(fu, nothing, fu, nothing, nothing, nothing, nothing, nothing);
        inplace = false,
    )
    res = solve(
        f_obj,
        Point(1.3, 1.3),
        LineSearch(GradientDescent(Inverse())),
        OptimizationOptions(),
    )
    @test res.info.minimum < 1e-9
    res = solve(f_obj, Point(1.3, 1.3), LineSearch(BFGS(Inverse())), OptimizationOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3, 1.3), LineSearch(DFP(; inverse = true, scaling = OrenLuenberger())), OptimizationOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3, 1.3), LineSearch(SR1(Inverse())), OptimizationOptions())
    @test res.info.minimum < 1e-4

    res = solve(
        f_obj,
        Point(1.3, 1.3),
        LineSearch(GradientDescent(Direct())),
        OptimizationOptions(),
    )
    @test res.info.minimum < 1e-9
    res = solve(f_obj, Point(1.3, 1.3), LineSearch(BFGS(Direct())), OptimizationOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3, 1.3), LineSearch(DFP(; inverse = false, scaling = OrenLuenberger())), OptimizationOptions())
    @test res.info.minimum < 1e-10
    # TODO: Look into this. Maybe SR1 updates are just not PSD and thus inappropriate with line search
    res = solve(f_obj, Point(1.3, 1.3), LineSearch(SR1(Direct())), OptimizationOptions())
    @test_broken res.info.minimum < 1e-10
end
