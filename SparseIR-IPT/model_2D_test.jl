using LinearAlgebra
#Parm(t_i, a_u, a_d, Pr, mu, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ)
struct Parm
    t_i::Float64
    Pr::Float64
    mu::Float64
    eta::Float64
    K_SIZE::Int
    dk2::Float64
end

function set_parm(arg::Array{String,1})
    t_i = parse(Float64,arg[1])
    Pr = parse(Float64,arg[2])
    mu0 = parse(Float64,arg[3])
    eta = parse(Float64,arg[4])
    K_SIZE = parse(Int,arg[5])
    dk2 = 1.0/(K_SIZE^2)

    return t_i, Pr, mu0, eta, K_SIZE, dk2
end

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
    eps::Float64 = 2.0p.t_i*(cos(k[1]) + cos(k[2])) + p.mu
    #(p.Pr*cos(k'*a1) + cos(k'*a2) + cos(k'*a3)) + p.mu
    return eps
end

function Disp_HSL(p::Parm)
    E = zeros(Float64, 4*p.K_SIZE)

    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [KK, 0.0]
        E[K0] = set_H(kk, p)
    end
    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [pi-KK, KK]
        E[K0+p.K_SIZE] = set_H(kk, p)
    end
    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [0.0, pi-KK]
        E[K0+2p.K_SIZE] = set_H(kk, p)
    end
    for K0 in 1:p.K_SIZE
        KK = pi*K0/p.K_SIZE
        kk = [KK, KK]
        E[K0+3p.K_SIZE] = set_H(kk, p)
    end
    p1 = plot(E, label="e",xlabel="HSL",xticks=([0, p.K_SIZE, 2p.K_SIZE, 3p.K_SIZE, 4p.K_SIZE],["Γ", "X", "X'", "Γ", "M"]),ylabel="E",title="Dispersion", width=3.0)
    savefig(p1,"./free_disp.png")

    return nothing
end


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
end
#=

using Plots
function main(arg::Vector{String})
    p = Parm(set_parm(arg)...)
    kk = get_kk(p.K_SIZE)
    println(length(kk))
    e = Disp_HSL(p)
    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)

    
end

@time main(ARGS)=#
