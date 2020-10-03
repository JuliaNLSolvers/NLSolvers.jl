using NLSolvers, StaticArrays, Test
@testset "mixed optimization problems" begin
function theta(x)
   if x[1] > 0
       return atan(x[2] / x[1]) / (2.0 * pi)
   else
       return (pi + atan(x[2] / x[1])) / (2.0 * pi)
   end
end
f(x) = 100.0 * ((x[3] - 10.0 * theta(x))^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

function f∇f!(x, ∇f)
    if !(∇f==nothing)
        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
        end
        ∇f[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
        ∇f[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
        ∇f[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
    end

    fx = f(x)
    objective_return(fx, ∇f)
end

function f∇f(x, ∇f)
    if !(∇f == nothing)
        ∇f = similar(x)
    end
    fx, ∇f = f∇f!(gx, x)
    objective_return(fx, ∇f)
end
function f∇fs(x, ∇f)
    if !(∇f == nothing)
        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
        end

        s1 = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 )
        s2 = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 )
        s3 = 200.0*(x[3]-10.0*theta(x)) + 2.0*x[3]
        ∇f = @SVector [s1, s2, s3]
    end
    objective_return(f(x), ∇f)
end
obj_inplace = OnceDiffed(f∇f!)
obj_outofplace = OnceDiffed(f∇f)
obj_static = OnceDiffed(f∇fs)

x0 = [-1.0, 0.0, 0.0]
xopt = [1.0, 0.0, 0.0]
x0s = @SVector [-1.0, 0.0, 0.0]

println("Starting  from: ", x0)
println("Targeting     : ", xopt)


res = minimize(obj_inplace, copy(x0), NelderMead(), MinOptions())
print("NN  $(summary(NelderMead()))         ")
@printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.nm_obj, Inf), res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), SimulatedAnnealing(), MinOptions())
print("NN  $(summary(SimulatedAnnealing()))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=HZ()), MinOptions())
print("NN  $(summary(ConjugateGradient(update=HZ())))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=CD()), MinOptions())
print("NN  $(summary(ConjugateGradient(update=CD())))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=HS()), MinOptions())
print("NN  $(summary(ConjugateGradient(update=HS())))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=FR()), MinOptions())
print("NN  $(summary(ConjugateGradient(update=FR())))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=PRP()), MinOptions())
print("NN  $(summary(ConjugateGradient(update=PRP(;plus=false))))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=PRP(plus=true)), MinOptions())
print("NN  $(summary(ConjugateGradient(update=PRP(;plus=true))))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=VPRP()), MinOptions())
print("NN  $(summary(ConjugateGradient(update=VPRP())))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=LS()), MinOptions())
print("NN  $(summary(ConjugateGradient(update=LS())))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(update=DY()), MinOptions())
print("NN  $(summary(ConjugateGradient(update=DY())))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)

for _method in (GradientDescent, LBFGS, BFGS, DBFGS, DFP, SR1)
# for _method in (GradientDescent, BFGS, DBFGS, DFP, SR1)
    methodtxt = summary(_method()) 
    for m in (Inverse(), Direct())
        mtxt = m isa Inverse ? "(inverse): " : "(direct):  "
        if _method == LBFGS && m isa Inverse
            res = minimize!(obj_inplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN! $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
        elseif _method !== LBFGS
            res = minimize(obj_inplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN  $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            res = minimize!(obj_inplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN! $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            res = minimize(obj_static, x0s, LineSearch(_method(m)), MinOptions())
            print("NN  $_method(S) $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            if m isa Direct && !(_method == GradientDescent)
                println("Trust region: NWI")
                res = minimize!(obj_inplace, copy(x0), TrustRegion(_method(Direct()), NWI()), MinOptions())
                print("NN! $_method    $mtxt")
                @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
                println("Trust region: NTR")
                res = minimize!(obj_inplace, copy(x0), TrustRegion(_method(Direct()), NTR()), MinOptions())
                print("NN! $_method    $mtxt")
                @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            end
        end
    end
end
println()

res = minimize(obj_inplace, x0, LineSearch(BFGS(Inverse())), MinOptions())
@test res.info.iter == 30
@printf("NN  BFGS    (inverse): %2.2e  %2.2e %2.2e %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
res = minimize!(obj_inplace, copy(x0), LineSearch(BFGS(Inverse()), Backtracking()), MinOptions())
@test res.info.iter == 30
@printf("NN! BFGS    (inverse): %2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
res = minimize!(obj_inplace, copy(x0), LineSearch(BFGS(Inverse()), Backtracking(interp=FFQuadInterp())), MinOptions())
@test res.info.iter == 30
@printf("NN! BFGS    (inverse, quad): %2.2e %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
res = minimize(obj_static, x0s, LineSearch(BFGS(Inverse())), MinOptions())
@test res.info.iter == 30
@printf("NN  BFGS(S) (inverse): %2.2e %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
res = minimize(obj_static, x0s, LineSearch(BFGS(Inverse()), Backtracking(interp=FFQuadInterp())), MinOptions())
@test res.info.iter == 30
@printf("NN  BFGS(S) (inverse, quad): %2.2e %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)

x0 = rand(3)
x0s = SVector{3}(x0)
println("\nFrom a random point: ", x0)
res = minimize(obj_inplace, copy(x0), NelderMead(), MinOptions())
print("NN  $(summary(NelderMead()))         ")
@printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.nm_obj, Inf), res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), SimulatedAnnealing(), MinOptions())
print("NN  $(summary(SimulatedAnnealing()))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)
res = minimize(obj_inplace, copy(x0), ConjugateGradient(), MinOptions())
print("NN  $(summary(ConjugateGradient()))         ")
@printf("%2.2e  %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf),  res.info.minimum, res.info.iter)

for _method in (GradientDescent, LBFGS, BFGS, DBFGS, DFP, SR1)
    methodtxt = summary(_method()) 
    for m in (Inverse(), Direct())
        mtxt = m isa Inverse ? "(inverse): " : "(direct):  "
        if _method == LBFGS && m isa Inverse
            res = minimize!(obj_inplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN! $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
        elseif _method !== LBFGS
            res = minimize(obj_inplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN  $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            res = minimize!(obj_inplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN! $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            res = minimize(obj_static, x0s, LineSearch(_method(m)), MinOptions())
            print("NN  $_method(S) $mtxt")
            @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            if m isa Direct && !(_method == GradientDescent)
                println("Trust region: NWI")
                res = minimize!(obj_inplace, copy(x0), TrustRegion(_method(Direct()), NWI()), MinOptions())
                print("NN! $_method    $mtxt")
                @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
                println("Trust region: NTR")
                res = minimize!(obj_inplace, copy(x0), TrustRegion(_method(Direct()), NTR()), MinOptions())
                print("NN! $_method    $mtxt")
                @printf("%2.2e  %2.2e %2.2e %d\n",  norm(res.info.minimizer-xopt,Inf), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            end
        end
    end
end
println()

function himmelblau!(x, ∇f)
    if !(∇f == nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    objective_return(fx, ∇f)
end


function himmelblaus(x, ∇f)
    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    if !(∇f == nothing)
        ∇f1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        ∇f = @SVector([∇f1, ∇f2])
    end
    objective_return(fx, ∇f)
end

function himmelblau(x, ∇f)
    g = ∇f == nothing ? ∇f : similar(x)

    return himmelblau!(x, g)
end

him_inplace = OnceDiffed(himmelblau!)
him_static = OnceDiffed(himmelblaus)
him_outofplace = OnceDiffed(himmelblau)

println("\nHimmelblau function")
x0 = [3.0, 1.0]
x0s = SVector{2}(x0)
minimizers = [[3.0,2.0],[-2.805118,3.131312],[-3.779310,-3.283186],[3.584428,-1.848126]]
for _method in (GradientDescent, LBFGS, BFGS, DBFGS, DFP, SR1)
# for _method in (GradientDescent, BFGS, DBFGS, DFP, SR1)
    methodtxt = summary(_method()) 
    for m in (Inverse(), Direct())
        mtxt = m isa Inverse ? "(inverse): " : "(direct):  "
        if _method == LBFGS && m isa Inverse
            res = minimize!(him_inplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN! $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
        elseif _method !== LBFGS
            res = minimize(him_outofplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN  $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            res = minimize!(him_inplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN! $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            res = minimize(him_static, x0s, LineSearch(_method(m)), MinOptions())
            print("NN  $_method(S) $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            if m isa Direct && !(_method == GradientDescent)
                println("Trust region: NWI")
                res = minimize!(him_inplace, copy(x0), TrustRegion(_method(Direct()), NWI()), MinOptions())
                print("NN! $_method    $mtxt")
                @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
                println("Trust region: NTR")
                res = minimize!(him_inplace, copy(x0), TrustRegion(_method(Direct()), NTR()), MinOptions())
                print("NN! $_method    $mtxt")
                @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            end
        end
    end
end
println()
println()
xrand = rand(2)
xrands = SVector{2}(xrand)
println("\nFrom a random point: ", xrand)

for _method in (GradientDescent, LBFGS, BFGS, DBFGS, SR1)
# for _method in (GradientDescent, BFGS, DBFGS, DFP, SR1)
    methodtxt = summary(_method()) 
    for m in (Inverse(), Direct())
        mtxt = m isa Inverse ? "(inverse): " : "(direct):  "
        if _method == LBFGS && m isa Inverse
            res = minimize!(him_inplace, copy(x0), LineSearch(_method(m)), MinOptions())
            print("NN! $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
        elseif _method !== LBFGS
            res = minimize(him_outofplace, copy(xrand), LineSearch(_method(m)), MinOptions())
            print("NN  $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            res = minimize!(him_inplace, copy(xrand), LineSearch(_method(m)), MinOptions())
            print("NN! $_method    $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            res = minimize(him_static, xrands, LineSearch(_method(m)), MinOptions())
            print("NN  $_method(S) $mtxt")
            @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            if m isa Direct && !(_method == GradientDescent)
                println("Trust region: NWI")
                res = minimize!(him_inplace, copy(xrand), TrustRegion(_method(Direct()), NWI()), MinOptions())
                print("NN! $_method    $mtxt")
                @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
                println("Trust region: NTR")
                res = minimize!(him_inplace, copy(xrand), TrustRegion(_method(Direct()), NTR()), MinOptions())
                print("NN! $_method    $mtxt")
                @printf("%2.2e  %2.2e %2.2e  %d\n", minimum([norm(res.info.minimizer-xopt,Inf) for xopt in minimizers]), norm(res.info.∇fz, Inf), res.info.minimum, res.info.iter)
            end
        end
    end
end
println()

end
