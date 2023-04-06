#using Distributed
#addprocs(30)
using LinearAlgebra
#Parm(t_i, a_u, a_d, Pr, mu, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ)
struct Parm
    td::Float64
    tp::Float64
    td_AB::Float64
    tp_AB::Float64
    t0_AB::Float64
    μd::Float64
    μp::Float64
    μf::Float64
    eta::Float64
    T::Float64
    K_SIZE::Int
    W_MAX::Float64
    W_in::Float64
    W_SIZE::Int
    abc::Vector{Int}
    #α::Char
    #β::Char
    #γ::Char
    H_size::Int
end

function set_parm(arg::Array{String,1})
    td = parse(Float64,arg[1])
    tp = parse(Float64,arg[2])
    td_AB = parse(Float64,arg[3])
    tp_AB = parse(Float64,arg[4])
    t0_AB = parse(Float64,arg[5])
    μd = parse(Float64,arg[6])
    μp = parse(Float64,arg[7])
    μf = parse(Float64,arg[8])
    eta = parse(Float64,arg[9])
    T = parse(Float64,arg[10])
    K_SIZE = parse(Int,arg[11])
    W_MAX = parse(Float64,arg[12])
    W_in = parse(Float64,arg[13])
    W_SIZE = parse(Int,arg[14])
    abc::Vector{Int} = [parse(Int,arg[15]), parse(Int,arg[16]), parse(Int,arg[17])]
    #α = (arg[15])[1]
    #β = (arg[16])[1]
    #γ = (arg[17])[1]

    #return t_i, a_u, a_d, Pr, mu0, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ, 4
    return td, tp, td_AB, tp_AB, t0_AB, μd, μp, μf, eta, T, K_SIZE, W_MAX, W_in, W_SIZE, abc, 4
end
#=
function set_parm_mudep(arg::Array{String,1}, mu::Float64)
    t_i = parse(Float64,arg[1])
    a_u = parse(Float64,arg[2])
    a_d = parse(Float64,arg[3])
    Pr = parse(Float64,arg[4])
    mu0 = mu
    eta = parse(Float64,arg[6])
    T = parse(Float64,arg[7])
    hx = parse(Float64,arg[8])
    hy = parse(Float64,arg[9])
    hz = parse(Float64,arg[10])
    K_SIZE = parse(Int,arg[11])
    W_MAX = parse(Float64,arg[12])
    W_in = parse(Float64,arg[13])
    W_SIZE = parse(Int,arg[14])
    α = (arg[15])[1]
    β = (arg[16])[1]
    γ = (arg[17])[1]

    return t_i, a_u, a_d, Pr, mu0, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ
end

function set_parm_etadep(arg::Array{String,1}, eta0::Float64)
    t_i = parse(Float64,arg[1])
    a_u = parse(Float64,arg[2])
    a_d = parse(Float64,arg[3])
    Pr = parse(Float64,arg[4])
    mu0 = parse(Float64,arg[5])
    eta = eta0
    T = parse(Float64,arg[7])
    hx = parse(Float64,arg[8])
    hy = parse(Float64,arg[9])
    hz = parse(Float64,arg[10])
    K_SIZE = parse(Int,arg[11])
    W_MAX = parse(Float64,arg[12])
    W_in = parse(Float64,arg[13])
    W_SIZE = parse(Int,arg[14])
    α = (arg[15])[1]
    β = (arg[16])[1]
    γ = (arg[17])[1]

    return t_i, a_u, a_d, Pr, mu0, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ
end

function set_parm_Tdep(arg::Array{String,1}, T0::Float64)
    t_i = parse(Float64,arg[1])
    a_u = parse(Float64,arg[2])
    a_d = parse(Float64,arg[3])
    Pr = parse(Float64,arg[4])
    mu0 = parse(Float64,arg[5])
    eta = parse(Float64,arg[6])
    T = T0
    hx = parse(Float64,arg[8])
    hy = parse(Float64,arg[9])
    hz = parse(Float64,arg[10])
    K_SIZE = parse(Int,arg[11])
    W_MAX = parse(Float64,arg[12])
    W_in = parse(Float64,arg[13])
    W_SIZE = parse(Int,arg[14])
    α = (arg[15])[1]
    β = (arg[16])[1]
    γ = (arg[17])[1]

    return t_i, a_u, a_d, Pr, mu0, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ
end=#

function set_parm_Wdep(arg::Array{String,1}, Win::Float64)
    td = parse(Float64,arg[1])
    tp = parse(Float64,arg[2])
    td_AB = parse(Float64,arg[3])
    tp_AB = parse(Float64,arg[4])
    t0_AB = parse(Float64,arg[5])
    μd = parse(Float64,arg[6])
    μp = parse(Float64,arg[7])
    μf = parse(Float64,arg[8])
    eta = parse(Float64,arg[9])
    T = parse(Float64,arg[10])
    K_SIZE = parse(Int,arg[11])
    W_MAX = parse(Float64,arg[12])
    W_in = Win
    W_SIZE = parse(Int,arg[14])
    abc::Vector{Int} = [parse(Int,arg[15]), parse(Int,arg[16]), parse(Int,arg[17])]
    #α = (arg[15])[1]
    #β = (arg[16])[1]
    #γ = (arg[17])[1]

    #return t_i, a_u, a_d, Pr, mu0, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ, 4
    return td, tp, td_AB, tp_AB, t0_AB, μd, μp, μf, eta, T, K_SIZE, W_MAX, W_in, W_SIZE, abc, 4
end


#a1 = [1.0, 0.0]
#a2 = [-0.5, sqrt(3.0)/2]
#a3 = [0.5, sqrt(3.0)/2]

#[τ0*s0, τx*s0, τz*sx, τz*sy, τz*s0]
#sigma = [[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0], [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0; 1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0], [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 -1.0; 0.0 0.0 -1.0 0.0], [0.0 -1.0im 0.0 0.0; 1.0im 0.0 0.0 0.0; 0.0 0.0 0.0 1.0im; 0.0 0.0 -1.0im 0.0], [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 -1.0 0.0; 0.0 0.0 0.0 -1.0]]
function generate_M2(H_size::Int)
    A::Vector{Hermitian{ComplexF64, Matrix{ComplexF64}}} =[]
    for i in 1:H_size
        for j in i:H_size
            if(i==j)
                #l = 2(d+1-i)*(i-1) + (i-1)^2 + 1
                B = zeros(ComplexF64,H_size,H_size)
                B[i,j] = 1.0
                B = Hermitian(B)
                push!(A,B)
                #A[l,:,:] = B[:,:]
            else
                #l = 2(d+1-i)*(i-1) + (i-1)^2 + 2*(j-i)
                B = zeros(ComplexF64,H_size,H_size)
                B[i,j] = 1.0
                B[j,i] = 1.0
                B = Hermitian(B)
                B2 = zeros(ComplexF64,H_size,H_size)
                B2[i,j] = -1.0im
                B2[j,i] = 1.0im
                B2 = Hermitian(B2)
                #A[l,:,:] = B[:,:]
                #A[l+1,:,:] = B2[:,:]
                push!(A,B)
                push!(A,B2)
            end
        end
    end
    return A
end
sigma = generate_M2(4)

R_ad = [-0.25, 0.35]
R_bd = [0.25, -0.35]
R_ap = [-0.25, -0.11]
R_bp = [0.25, 0.11]


#functions to set Hamiltonian and Velocity operators
function set_H(k::Vector{Float64},p::Parm)
    ϵd = p.μd + 2p.td*cos(k[2]) - p.μf
    ϵp = p.μp + 2p.tp*cos(k[2]) - p.μf
    V1_re = p.td_AB*(cos(-k'*(R_bd-R_ad)+k[2])+cos(-k'*(R_bd-R_ad)+k[2]-k[1]))
    V1_im = p.td_AB*(sin(k'*(R_bd-R_ad)-k[2])+sin(k'*(R_bd-R_ad)-k[2]+k[1]))
    V2_re = p.t0_AB*(cos(-k'*(R_bp-R_ad)) - cos(-k'*(R_bp-R_ad)-k[1]))
    V2_im = p.t0_AB*(sin(k'*(R_bp-R_ad)) - sin(k'*(R_bp-R_ad)+k[1]))
    V3_re = -V2_re
    V3_im = -V2_im
    V4_re = p.tp_AB*(cos(-k'*(R_bp-R_ap))+cos(-k'*(R_bp-R_ap)-k[1]))
    V4_im = p.tp_AB*(sin(k'*(R_bp-R_ap))+sin(k'*(R_bp-R_ap)+k[1]))
    gg = [ϵd, 0.0, 0.0, V1_re, V1_im, V2_re, V2_im, ϵp, V3_re, V3_im, V4_re, V4_im, ϵd, 0.0, 0.0, ϵp]
    H::Array{ComplexF64,2} =  sigma' * gg
    return H
end

function get_kk(K_size::Int)
    kk = []
    k = collect(-pi:2pi/K_size:pi)
    for kx in k, ky in k
        push!(kk, [kx,ky])
    end
    return kk
end

#If you want to use ForwardDiff
function set_H_v(k,p::Parm)
    ϵd = p.μd + 2p.td*cos(k[2]) - p.μf
    ϵp = p.μp + 2p.tp*cos(k[2]) - p.μf
    V1_re = p.td_AB*(cos(-k'*(R_bd-R_ad)+k[2])+cos(-k'*(R_bd-R_ad)+k[2]-k[1]))
    V1_im = p.td_AB*(sin(k'*(R_bd-R_ad)-k[2])+sin(k'*(R_bd-R_ad)-k[2]+k[1]))
    V2_re = p.t0_AB*(cos(-k'*(R_bp-R_ad)) - cos(-k'*(R_bp-R_ad)-k[1]))
    V2_im = p.t0_AB*(sin(k'*(R_bp-R_ad)) - sin(k'*(R_bp-R_ad)+k[1]))
    V3_re = -V2_re
    V3_im = -V2_im
    V4_re = p.tp_AB*(cos(-k'*(R_bp-R_ap))+cos(-k'*(R_bp-R_ap)-k[1]))
    V4_im = p.tp_AB*(sin(k'*(R_bp-R_ap))+sin(k'*(R_bp-R_ap)+k[1]))
    gg = [ϵd, 0.0, 0.0, V1_re, V1_im, V2_re, V2_im, ϵp, V3_re, V3_im, V4_re, V4_im, ϵd, 0.0, 0.0, ϵp]
    return gg
end
