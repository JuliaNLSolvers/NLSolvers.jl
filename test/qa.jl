using NLSolvers
using Test
import Aqua
import ExplicitImports
import JET

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(NLSolvers)
    end

    @testset "ExplicitImports" begin
        # No implicit imports (`using XY`)
        @test ExplicitImports.check_no_implicit_imports(
            NLSolvers
        ) === nothing

        # All explicit imports (`using XY: Z`) are loaded via their owners
        @test ExplicitImports.check_all_explicit_imports_via_owners(
            NLSolvers;
            ignore = (
                # ExplicitImports does currently not ignore non-public names of main package in extensions
                # Ref https://github.com/JuliaTesting/ExplicitImports.jl/issues/92
                :LinearAlgebra,
            ),
        ) === nothing

        # No explicit imports (`using XY: Z`) that are not used
        @test ExplicitImports.check_no_stale_explicit_imports(
            NLSolvers;
        ) === nothing

        # Nothing is accessed via modules other than its owner
        @test ExplicitImports.check_all_qualified_accesses_via_owners(NLSolvers) === nothing

        # NLSolvers currently accesses many non-public names
        @test ExplicitImports.check_all_qualified_accesses_are_public(NLSolvers) === nothing

        # No self-qualified accesses
        @test ExplicitImports.check_no_self_qualified_accesses(NLSolvers) === nothing
    end

    @testset "JET" begin
        # Check that there are no undefined global references and undefined field accesses
        res = JET.report_package(NLSolvers; target_defined_modules = true, mode = :typo, toplevel_logger = nothing)
        reports = JET.get_reports(res)
        @test isempty(reports)
        @test length(reports) == 0


        # Analyze methods based on their declared signature
        res = JET.report_package(NLSolvers; target_defined_modules = true, toplevel_logger = nothing)
        reports = JET.get_reports(res)
        @test_broken isempty(reports)
        @test length(reports) <= 1
    end
 end 