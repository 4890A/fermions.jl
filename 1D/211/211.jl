#!etc/julia

#using Distributed
#using SharedArrays
#addprocs(Sys.CPU_THREADS)

using ProgressMeter
using DelimitedFiles

using LinearAlgebra
using Dierckx

using Plots
gr()

struct SETUP

	L::Int
	cutoff::Int
	e3::Float64

end

# calculate the lattice renomalization
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


# carry out the sum required for the lattice renomalization
function g_summand(self::SETUP, n1, n2)

	cutoff = self.cutoff
	if abs(n1+n2) <= cutoff;
		denom = 4 .* pi^2 * (n1^2 + n2^2 + n1*n2 + self.e3)
		return 1.0/ denom
	else
		return 0.0
	end
end


# diagonlize the matrix for constructing f from itself. eigenvalue =1 implies Wf = f, and
# f satisfies the recursive definition
function diag(f)
	λ = []
	for i in 1:size(f)[1]
		A = Symmetric(f[i,:,:])
		push!(λ, eigvals(A))
	end

	return transpose(hcat(λ...))
end

# construct matrix W that constructs f from itself Wf = f
function construct_matrix(f, vals, cutoff, E, g)
	#f = SharedArray{Float64}(size(f))

	#@sync @distributed for p in vals
	for p in vals
		ip = p + cutoff + 1

		for x in vals
			if p == x
				continue
			end
			ix = x + cutoff + 1

			for k in vals
				for q in vals

					if p + x + q + k != 0
						continue
					end

					T = (p^2 + x^2 + k^2 + q^2)/2.
					f[:, ip , ip] += 1 ./(T .+ E)
					f[:, ip , ix] -= 1 ./(T .+ E)
				end
			end
		end
	end

	f = convert(Array{Float64}, f)
	return f = f * (g /(4 * pi^2))
end


function solve(L, e3, alpha)

	cutoff = div(L - 1,2)
	s = SETUP(L, cutoff, e3)
	g = get_g(s)
	E = alpha * e3
	vals = -cutoff:cutoff

	f0 = zeros(size(alpha)[1], 2*cutoff + 1, 2*cutoff + 1)
	f = construct_matrix(f0, vals, cutoff, E, g)

	#println("diagonalization time")
	λ = diag(f)
	return λ

end


# find roots (alpha) of Wf =  λf eigenvalues such that λ = 1 and Wf = f
function get_eigen_alpha(λ, alpha, L)
	eigen_alpha :: Array{Float64} = []
	for i in 1:L
		try
			spl = Dierckx.Spline1D(alpha, λ[:,i] .- 1.0)
			root = roots(spl)[1]
			# exclude roots that occur due to connecting sides of poles
			if abs(derivative(spl, root)) < 3000
				push!(eigen_alpha, root)
			end
		catch
			continue
		end
	end
	return eigen_alpha
end

function find_energies(from, to, density)
	L = 31
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
	ylims!(0.0, 3.0)
	png(string("raw_roots", search_range, ".png"))

end
