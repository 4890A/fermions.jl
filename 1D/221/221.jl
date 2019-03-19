using LinearAlgebra
using DelimitedFiles
using ProgressMeter
using Dierckx
using Plots

struct SETUP

	L::Int
	cutoff::Int
	e3::Float64

end


function get_g(self::SETUP)
	cutoff = self.cutoff
	out = 0.

	for i in -cutoff:cutoff
		for j in -cutoff:cutoff

			out += g_summand(self, i, j)
		end
	end
	return 1/out
end


function g_summand(self::SETUP, n1, n2)

	cutoff = self.cutoff
	if abs(n1+n2) <= cutoff;
		denom = 4 .* pi^2 * (n1^2 + n2^2 + n1*n2 + self.e3)
		return 1.0/ denom
	else
		return 0.0
	end
end


function diag(f)
	λ = []
	for i in 1:size(f)[1]
		A = Symmetric(f[i,:,:])
		push!(λ, eigvals(A))
	end

	return transpose(hcat(λ...))
end

function indexify(L)
	# vectorize a function of two arguments into an function f of a single argument
	W = zeros(Int, L, L)
	index::Int = 0
	for i in 1:L
		for j in 1:L
			index += 1
			W[i, j] = index
		end
	end
	return W
end

function construct_matrix(f, vals, cutoff, E, g, L)
	#f = SharedArray{Float64}(size(f))
	#@sync @distributed for p in vals
	W = indexify(L)
	for p1 in vals
		for k1 in vals

			for p2 in vals
				if p2 == p1
					continue
				end
				for k2 in vals
					if k2 == k1
						continue
					end

					ς::Int = - (p1 + k1 + p2 + k2)
					if ~ (ς in vals)
						continue
					end

					ip1, ik1, ip2, ik2 = map(x -> x + cutoff + 1,
											 [p1, k1, p2, k2])
					p1k1 = W[ip1, ik1]
					p2k1 = W[ip2, ik1]
					p2k2 = W[ip2, ik2]
					p1k2 = W[ip1, ik2]

					T = sum(map(x -> x^2 , [p1, p2, k1, k2, ς]))/2.
					G = 1 ./(T .+ E)

					f[:, p1k1, p1k1] += G
					f[:, p1k1, p2k1] -= G
					f[:, p1k1, p2k2] += G
					f[:, p1k1, p1k2] -= G
				end
			end
		end
	end

	f = convert(Array{Float64}, f)
	return f = f * (g /(4. * pi^2))
end


function solve(L, e3, alpha)

	cutoff = div(L - 1,2)
	s = SETUP(L, cutoff, e3)
	g = get_g(s)
	E = alpha * e3
	vals = -cutoff:cutoff

	f0 = zeros(size(alpha)[1], L^2, L^2)
	f = construct_matrix(f0, vals, cutoff, E, g, L)

	println("diagonalization time")
	λ = @time diag(f)
	return λ

end

function get_eigen_alpha(λ, alpha, L)
	eigen_alpha :: Array{Float64} = []

	for i in 1:L^2
		try
			spl = Dierckx.Spline1D(alpha, λ[:,i] .- 1.0)
			root_list = roots(spl)
			push!(eigen_alpha, root_list...)
		catch
			continue
		end
	end

	return eigen_alpha
end

function find_energies(from, to, density)
	L::Int = 13
	e3 = 2.0
	alpha = collect(range(from; length=density, stop= to))


	# animation code
	"""
	anim = @animate for i in L
	    plot(alpha, solve(i, e3, alpha))
		ylims!(.9,1.1)
	end
	mp4(anim, "anim_fps15.mp4", fps = 15)
	plot(alpha, solve(L[end], e3, alpha))
	ylims!(.9,1.1)
	"""

	λ = solve(L, e3, alpha)
	roots =  get_eigen_alpha(λ, alpha, L)
	println(roots)
	return roots, alpha, λ

end

function main(start, stop, root_grid_step=.001)

	all_roots =  []
	search_range = start:root_grid_step:stop
	alpha_grid_density = 31

	@showprogress 1 "Computing..." for start in search_range
		stop = start + root_grid_step
		all_roots = vcat(find_energies(start, stop,
				alpha_grid_density)[1], all_roots)
	end
	println(all_roots)
	# write roots to csv
	writedlm( string("raw_roots", search_range, ".csv"),  all_roots, ',')
	# graph roots on λ curves for debugging
	graph_data = find_energies(search_range[1], search_range[end], 503)
	plot(graph_data[2], graph_data[3], legend = false)
	scatter!(all_roots, ones(length(all_roots)))
	ylims!(.9, 1.1)
	png(string("raw_roots", search_range, ".png"))

end

find_energies(0,1.0,100)
