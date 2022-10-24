using LinearAlgebra
#Parm(t_i, a_u, a_d, Pr, mu, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ)
struct Parm
    t_i::Float64
    t_b::Float64
    Δ::Float64
    mu::Float64
    eta::Float64
    K_SIZE::Int
    dk2::Float64
end

function set_parm(arg::Array{String,1})
    t_i = parse(Float64,arg[1])
    t_b = parse(Float64,arg[2])
    Δ = parse(Float64,arg[3])
    mu0 = parse(Float64,arg[4])
    eta = parse(Float64,arg[5])
    K_SIZE = parse(Int,arg[6])
    dk2 = 1.0/(K_SIZE^2)

    return t_i, t_b, Δ, mu0, eta, K_SIZE, dk2
end

#a1 = [1.0, 0.0]
#a2 = [-0.5, sqrt(3.0)/2]
#a3 = [0.5, sqrt(3.0)/2]

sigma = [[1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0], [0.0 -1.0im; 1.0im 0.0], [1.0 0.0; 0.0 -1.0]]

function get_kk(K_SIZE::Int)
    kk::Vector{Vector{Float64}} = []
    dk = 2pi/(K_SIZE)
    for x in collect(-pi+dk:dk:pi)
        for y in collect(-pi+dk:dk:pi)
            k0 = [x, y]
            push!(kk,k0)
        end
    end
    return kk
end

function set_H(k::Vector{Float64},p::Parm)
    eps0::Float64 = p.t_i*cos(k[2]) + p.mu
    t_x::Float64 = p.t_i*cos(k[1])
    t_y::Float64 = p.t_b*sin(k[1])
    t_z::Float64 = p.Δ 
    ee = [eps0, t_x, t_y, t_z]
    H = ee' * sigma
    return H
end

function Disp_HSL(p::Parm)
    E = zeros(Float64, 4*p.K_SIZE, 2)

    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [KK, 0.0]
        e, v = eigen(set_H(kk, p))
        E[K0,:] = real.(e)
    end
    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [pi-KK, KK]
        e, v = eigen(set_H(kk, p))
        E[K0+p.K_SIZE,:] = real.(e)
    end
    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [0.0, pi-KK]
        e, v = eigen(set_H(kk, p))
        E[K0+2p.K_SIZE,:] = real.(e)
    end
    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [KK, KK]
        e, v = eigen(set_H(kk, p))
        E[K0+3p.K_SIZE,:] = real.(e)
    end
    p1 = plot(real.(E), label="e",xlabel="HSL",xticks=([0, p.K_SIZE, 2p.K_SIZE, 3p.K_SIZE, 4p.K_SIZE],["Γ", "X", "X'", "Γ", "M"]),ylabel="E",title="Dispersion", width=3.0)
    savefig(p1,"./free_disp.png")

    return nothing
end

#=
function Spectral_HSL(p::Parm, w_mesh::Vector{Float64}, sigma_w::Vector{ComplexF64})
    E = zeros(Float64, 4*p.K_SIZE, length(sigma_w))

    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [KK, 0.0]
        E[K0,:] = -imag.(1.0 ./( w_mesh .- set_H(kk, p) .- sigma_w))
    end
    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [pi-KK, KK]
        E[K0+p.K_SIZE,:] = -imag.(1.0 ./( w_mesh .- set_H(kk, p) .- sigma_w))
    end
    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [0.0, pi-KK]
        E[K0+2p.K_SIZE,:] = -imag.(1.0 ./( w_mesh .- set_H(kk, p) .- sigma_w))
    end
    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [KK, KK]
        E[K0+3p.K_SIZE,:] = -imag.(1.0 ./( w_mesh .- set_H(kk, p) .- sigma_w))
    end
    q = 1:4p.K_SIZE
    p1 = plot(q, w_mesh, E,st=:heatmap, label="e",xlabel="HSL",xticks=([0, p.K_SIZE, 2p.K_SIZE, 3p.K_SIZE, 4p.K_SIZE],["Γ", "X", "X'", "Γ", "M"]),ylabel="E",title="Dispersion", width=3.0)
    savefig(p1,"./disp.png")

    return nothing
end=#

#=
function check_DOS(p::Parm, ws::Int, bw::Float64)
    w_mesh = range(-bw, bw, length=ws)
    kk = get_kk(p.K_SIZE)
    Aw = zeros(Float64, ws)
    for w in 1:ws
        for k in 1:size(kk)[1]
            GA0 = Matrix{Complex{Float64}}((w_mesh[w] - 1.0im*p.eta)*I,2,2) - set_H(kk[k],p)
            GA = inv(GA0)
            Aw[w] += p.dk2 * imag(tr(GA))/pi
        end
    end
    return w_mesh, Aw
end


using Plots
function main(arg::Vector{String})
    p = Parm(set_parm(arg)...)
    kk = get_kk(p.K_SIZE)
    println(length(kk))
    e = Disp_HSL(p)
    w_mesh, Aw = check_DOS(p, 400, 2.0)
    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)
    p2 = plot(w_mesh, Aw, linewidth=2.0)
    savefig(p2,"./Aw.png")


    
end

@time main(ARGS)=#
