#!etc/julia

#using Distributed
#using SharedArrays
#addprocs(Sys.CPU_THREADS)

using ProgressMeter
using NPZ

using LinearAlgebra
using Dierckx
using FFTW

using PyPlot
pygui(true)

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
function diag(f, vec = false)
	if vec == false
		λ = []
		for i in 1:size(f)[1]
			A = Symmetric(f[i,:,:])
			push!(λ, eigvals(A))
		end

	else
		for i in 1:size(f)[1]
			A = Symmetric(f[i,:,:])
			return eigen(A)
		end

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


function solve(L, e3, alpha, vec = false)
		cutoff = div(L - 1,2)
		s = SETUP(L, cutoff, e3)
		g = get_g(s)
		E = alpha * e3
		vals = -cutoff:cutoff

		f0 = zeros(size(alpha)[1], 2*cutoff + 1, 2*cutoff + 1)
		f = construct_matrix(f0, vals, cutoff, E, g)

		#println("diagonalization time")
	if vec == false
		λ = diag(f)
		return λ
	else
		λ = diag(f, vec)
		return λ
	end

end


# find roots (alpha) of Wf =  λf eigenvalues such that λ = 1 and Wf = f
function get_eigen_alpha(λ, alpha, L)
	eigen_alpha :: Array{Float64} = []
	for i in 1:L
		try
			spl = Dierckx.Spline1D(alpha, λ[:,i] .- 1.0)
			root = roots(spl)[1]
			# exclude roots that occur due to connecting sides of poles
			if true
				push!(eigen_alpha, root)
			end
		catch
			continue
		end
	end
	return eigen_alpha
end

function find_energies(from, to, density, e3, n)
	L = 71
	alpha = collect(range(from; length=density, stop= to))

	λ = solve(L, e3, alpha)
	roots =  get_eigen_alpha(λ, alpha, L)
	ground_alpha = [roots[end - n]]
	println(ground_alpha)

	eval, evec = solve(L, e3, ground_alpha, true)
	evec_index = findall(x->abs(x-1)<.001,eval)
	f = evec[:, evec_index[1]]

	ϕ = find_phi(f, L, ground_alpha[1],e3)
	return ϕ

end

function find_phi(f, L, ground_alpha, e3)
	E = ground_alpha * e3
	ϕ = zeros(L, L, L, L)
	cutoff = div(L - 1,2)
	vals = -cutoff:cutoff

	for p in vals
		ip = p + cutoff + 1

		for x in vals

			if x == p
				continue
			end

			ix = x + cutoff + 1

			for k in vals
				for q in vals

					if (p + x + k + q) != 0
						continue
					end
					ik = k + cutoff + 1
					iq = q + cutoff + 1

					T = (p^2 + x^2 + k^2 + q^2)/2.
					ϕ[ip, ix, ik, iq] = (f[ip] - f[ix])/(T + E)
				end
			end
		end
	end


	return ϕ
end

function xdensity(A)
	l = size(A)[1]
	B = zeros(l,l)
	center = cld(l,2)
	for i in 1:l
		B[:,:] += A[:,:,center,i]
	end
	return B
end

function main(start, stop, density)
	println("trimer binding energy e3 = ?")
	e3 = parse(Float64,readline(stdin))
	println("excited state? 0 for ground state")
	n = parse(Int,readline(stdin))

	ϕ = find_energies(start, stop, density, e3, n)

	ψ = fft(ϕ)
	ψ2 = abs.(ψ)
	# npzwrite("fft.npz", ψ2)
	# npzwrite("pwf.npz",abs.(ϕ))

	B = xdensity(ψ2)
	imshow(B[:,end:-1:1],cmap="jet",interpolation="bilinear")

end
