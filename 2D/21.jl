using LinearAlgebra
using Dierckx
using Plots


struct SETUP

	L::Int
	cutoff::Int
	e2::Float64

end


function g_summand(self::SETUP, n1, n2)

	#if abs(n1+n2) <= self.cutoff
	if true
		denom = 4.0 * (π^2) * (n1^2 + n2^2 + self.e2)
		return 1.0 / denom
	else
		return 0.0
	end
end

# calculates the lattice coupling
function get_g(self::SETUP)

	out = 0.0
	cutoff = self.cutoff

	for i in -cutoff:cutoff
		for j in -cutoff:cutoff
			out += g_summand(self, i, j)
		end
	end

	denom = 1.0 / out
	return denom
end

function diag(f)
	λ = []
	for i in 1:size(f)[1]
		#A = f[i,:,:]
		A = Symmetric(f[i,:,:])
		push!(λ, eigvals(A))
	end

	return transpose(hcat(λ...))
end


function construct_matrix(f, vals, cutoff, E, g, L)
	## parallization is super slow
	#f = SharedArray{Float64}(size(f))
	#@sync @distributed for p in vals
	W = indexify(L)
	for p1x in vals
		ip1x = p1x + cutoff + 1
		for p1y in vals
			ip1y = p1y + cutoff + 1
			for p2x in vals
				ip2x = p2x + cutoff + 1
				for p2y in vals
					ip2y = p2y + cutoff + 1
					if p1x == p2x && p1y == p2y
						continue
					end

					for kx in vals
						for ky in vals
							# move this statement up one
							if p1x + p2x + kx != 0 || p1y + p2y + ky != 0
								continue
							end

							ip1 = W[ip1x, ip1y]
							ip2 = W[ip2x, ip2y]

							T = (p1x^2 + p1y^2 + p2x^2 +
								p2y^2 + kx^2 + ky^2)/2.

							f[:, ip1, ip1] += 1 ./(T .+ E)
							f[:, ip1, ip2] -= 1 ./(T .+ E)
						end
					end
				end
			end
		end
	end

	f = convert(Array{Float64}, f)
	return f = f * (g /(4 * pi^2))
end


function solve(L, e2, alpha)

	cutoff::Int = div(L - 1,2)
	s = SETUP(L, cutoff, e2)
	g = get_g(s)
	E = alpha * e2
	vals = -cutoff:cutoff

	f0 = zeros(size(alpha)[1], L^2, L^2)
	f = construct_matrix(f0, vals, cutoff, E, g, L)
	"""
	println("diagonalization time")
	"""
	λ = diag(f)
	return λ

end

function indexify(L)

	# indexing scheme to devectorize the 2-D momentum
	# [px = 1:py = 1], [px = 1: py = 2] etc...
	# for [px = x and py = y W[px, py] returns appropriate index in 1-D vector
	W = zeros(Int, L, L)
	index::Int = 0
	for i in 1:L
		for j in 1:L
			index += 1
			W[i,j] = index
		end
	end
	return W

end


function get_eigen_alpha(λ, alpha, L)
	eigen_alpha = []
	for i in 1:size(λ)[2]
		try
			spl = Dierckx.Spline1D(alpha, λ[:,i] .- 1.0)
			push!(eigen_alpha, roots(spl)[1])
		catch
			continue
		end
	end
	return eigen_alpha
end


function main()

	L = 23
	e2::Float64 = 7.5
	#e2 = exp10.(range(-1; length=200, stop=5.0))

	alpha = collect(range(-.2; length=100, stop=-.1))
	# animation code
	"""
	anim = @animate for i in e2
	    plot(alpha, solve(L, i, alpha), title=string("e3 = ", i))
		xlabel!("E / e3")
		ylims!(.9,1.1)
	end
	mp4(anim, "e2_anim_11.mp4", fps = 30)
	"""
	λ = solve(L, e2, alpha)
	#png(pl, "pl.png")
	roots = get_eigen_alpha(λ, alpha, L)
	println("The eigen energy ratios are: ")
	println(roots)
	plot(alpha, λ)
	ylims!(.9,1.1)
	#scatter(roots[end:-1:1])
end

@time main()
