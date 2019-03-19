#using Distributed
#using SharedArrays
#addprocs(Sys.CPU_THREADS)
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
		for p2 in vals
			for p3 in vals
				if p1 == p2 || p1 == p3 || p2 == p3
					continue
				end
				for k in vals
					for q in vals
						if p1 + p2 + p3 + k + q != 0
							continue
						end

						ip1, ip2, ip3 = map(x -> x + cutoff + 1, [p1, p2, p3])

						p1p2 = W[ip1, ip2]
						p2p1 = W[ip2, ip1]
						p2p3 = W[ip2, ip3]
						p3p2 = W[ip3, ip2]
						p3p1 = W[ip3, ip1]
						p1p3 = W[ip1, ip3]

						T = sum(map(x -> x^2 , [p1, p2, p3, k, q]))/2.

						f[:, p1p2, p1p2] += 1 ./(T .+ E)
						f[:, p1p2, p2p1] -= 1 ./(T .+ E)
						f[:, p1p2, p2p3] += 1 ./(T .+ E)
						f[:, p1p2, p3p2] -= 1 ./(T .+ E)
						f[:, p1p2, p3p1] += 1 ./(T .+ E)
						f[:, p1p2, p1p3] -= 1 ./(T .+ E)
					end
				end
			end
		end
	end

	f = convert(Array{Float64}, f/2.)
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

	λ = diag(f)
	return λ

end

function get_eigen_alpha(λ, alpha, L)
	eigen_alpha :: Array{Float64} = []

	for i in 1:L^2
		try
			spl = Dierckx.Spline1D(alpha, λ[:,i] .- 1.0)
			root_list = roots(spl)
			for root_check in root_list
				if derivative(spl, root_check) < 3000
					push!(eigen_alpha, root_check)
				end
			end
		catch
			continue
		end
	end

	return eigen_alpha
end

function find_energies(from, to, density)
	L = 15
	e3 = 7.5
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
	ylims!(-3., 3.0)
	png(string("raw_roots", search_range, ".png"))

end

main(-4.6, -4.2, .001)
