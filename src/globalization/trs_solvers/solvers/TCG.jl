struct TCG <: TRSPSolver
end
summary(::TCG) = "Steihaug-Toint Truncated CG"

function (ms::TCG)(∇f, H, Δ::T, s, scheme, λ0=0; abstol=1e-10, maxiter=min(5,length(s)), κeasy=T(1)/10, κhard=T(2)/10) where T
	# We only know that TCG has its properties if we start with s = 0
	s .= T(0)
	# g is the gradient of the quadratic model with the step α*p
	g = copy(∇f)
	v = apply_inverse_preconditioner(InPlace(), P, copy(g), g)
	p = -v
	for i = 1:maxiter
		# Check for negative curvature
		κ = dot(p, H*p)
		if κ ⩽ T(0)
			# positive_root_of
			σ = positive_root_of()
			s .= s .+ σ.*p
		end
		α = dot(g, v)/κ
		step_length = M_norm(s .+ α.*p)
		# If next step is outside, step to the boundary
		if step_length ⫺ Δ
			σ = positive_root_of()
			s .= s .+ σ.*p
		end
		s .= s + α*p
		den = dot(g, v)
		g .= g .+ α.*(H*p)
		v = apply_inverse_preconditioner(InPlace(), P, v, g)
		β = dot(g, v)/den
		p .= -v .+ β.*p
	end
end