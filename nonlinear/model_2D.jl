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
    dk2 = 2.0/(3*sqrt(3.0)*K_SIZE^2)

    return t_i, Pr, mu0, eta, K_SIZE, dk2
end

a1 = [1.0, 0.0]
a2 = [-0.5, sqrt(3.0)/2]
a3 = [0.5, sqrt(3.0)/2]


sigma = [[1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0], [0.0 -1.0im; 1.0im 0.0], [1.0 0.0; 0.0 -1.0]]

function get_kk(K_SIZE::Int)
    kk::Vector{Vector{Float64}} = []
    dk = 4pi/(3K_SIZE)
    #dk2 = 2.0/(3*sqrt(3.0)*K_SIZE*K_SIZE)
    for i in collect(dk:dk:4pi/3)
        
        for j in collect(0:dk:4pi/3)
            k = j*a1 + i*a2
            push!(kk,k)
        end
        
        for j in collect(dk:dk:4pi/3)
            if (i+j) < (4pi/3+dk)
                k = -j*a1 + i*a3
                push!(kk,k)
            end
        end
    end
    l = length(kk)
    for i in 1:l
        k0 = kk[i]
        k0 = -1 .* k0
        push!(kk,k0)
    end
    for i in collect(-4pi/3:dk:4pi/3)
        k = i*a1
        push!(kk,k)
    end
    return kk
end

function set_H(k::Vector{Float64},p::Parm)
    eps::Float64 = 2.0p.t_i*(p.Pr*cos(k'*a1) + cos(k'*a2) + cos(k'*a3)) + p.mu
    return eps
end

function Disp_HSL(p::Parm)
    E = zeros(Float64, 4*p.K_SIZE)

    for K0 in 1:p.K_SIZE
        KK = 4pi/3*K0/p.K_SIZE
        k = 4pi/3*a1 + KK*a2
        kk = [k[1], k[2]]
        E[K0] = set_H(kk, p)
    end
    for K0 in 1:p.K_SIZE
        KK = 2pi/3*K0/p.K_SIZE
        k = 4pi/3*a3 - KK*a1
        kk = [k[1], k[2]]
        E[K0+p.K_SIZE] = set_H(kk, p)
    end
    for K0 in 1:p.K_SIZE
        KK = 4pi/3*K0/p.K_SIZE
        k = 4pi/3*a3 -2pi/3*a1 + KK*(0.5*a1 - a3) 
        kk = [k[1], k[2]]
        E[K0+2p.K_SIZE] = set_H(kk, p)
    end
    for K0 in 1:p.K_SIZE
        KK = 4pi/3*K0/p.K_SIZE
        k = KK*(a1)
        kk = [k[1], k[2]]
        E[K0+3p.K_SIZE] = set_H(kk, p)
    end
    p1 = plot(E, label="e",xlabel="HSL",xticks=([0, p.K_SIZE/2, p.K_SIZE, 2p.K_SIZE, 3p.K_SIZE, 4p.K_SIZE],["K", "M", "K'", "M'", "Γ", "K"]),ylabel="E",title="Dispersion", width=3.0)
    savefig(p1,"./free_disp.png")

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
