#using Distributed
#addprocs(30)

#Parm(t_i, a_u, a_d, Pr, mu, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ)
struct Parm
    t::Float64
    tl::Float64
    αr::Float64
    αd::Float64
    mu::Float64
    eta::Float64
    T::Float64
    hx::Float64
    Δz::Float64 #imaginary potential 
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
    t = parse(Float64,arg[1])
    tl = parse(Float64,arg[2])
    αr = parse(Float64,arg[3])
    αd = parse(Float64,arg[4])
    mu0 = parse(Float64,arg[5])
    eta = parse(Float64,arg[6])
    T = parse(Float64,arg[7])
    hx = parse(Float64,arg[8])
    Δz = parse(Float64,arg[9])
    K_SIZE = parse(Int,arg[10])
    W_MAX = parse(Float64,arg[11])
    W_in = parse(Float64,arg[12])
    W_SIZE = parse(Int,arg[13])
    abc::Vector{Int} = [parse(Int,arg[14]), parse(Int,arg[15]), parse(Int,arg[16])]
    #α = (arg[15])[1]
    #β = (arg[16])[1]
    #γ = (arg[17])[1]

    #return t_i, a_u, a_d, Pr, mu0, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ, 4
    return t, tl, αr, αd, mu0, eta, T, hx, Δz, K_SIZE, W_MAX, W_in, W_SIZE, abc, 4
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
    t = parse(Float64,arg[1])
    tl = parse(Float64,arg[2])
    αr = parse(Float64,arg[3])
    αd = parse(Float64,arg[4])
    mu0 = parse(Float64,arg[5])
    eta = parse(Float64,arg[6])
    T = parse(Float64,arg[7])
    hx = parse(Float64,arg[8])
    Δz = parse(Float64,arg[9])
    K_SIZE = parse(Int,arg[10])
    W_MAX = parse(Float64,arg[11])
    W_in = Win
    W_SIZE = parse(Int,arg[13])
    abc::Vector{Int} = [parse(Int,arg[14]), parse(Int,arg[15]), parse(Int,arg[16])]
    #α = (arg[15])[1]
    #β = (arg[16])[1]
    #γ = (arg[17])[1]

    #return t_i, a_u, a_d, Pr, mu0, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ, 4
    return t, tl, αr, αd, mu0, eta, T, hx, Δz, K_SIZE, W_MAX, W_in, W_SIZE, abc, 4
end


#a1 = [1.0, 0.0]
#a2 = [-0.5, sqrt(3.0)/2]
#a3 = [0.5, sqrt(3.0)/2]

sigma = [[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0], [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0; 1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0], [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 -1.0; 0.0 0.0 -1.0 0.0], [0.0 -1.0im 0.0 0.0; 1.0im 0.0 0.0 0.0; 0.0 0.0 0.0 1.0im; 0.0 0.0 -1.0im 0.0], [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 -1.0 0.0; 0.0 0.0 0.0 -1.0]]


#functions to set Hamiltonian and Velocity operators
function set_H(k::Vector{Float64},p::Parm)
    eps = -p.t*(cos(k[1])+cos(k[2])) + p.mu
    Vab = -2p.tl*cos(k[1]/2)*cos(k[2]/2)
    g_x = (p.αd-p.αr)*sin(k[2]) + p.hx
    g_y = (p.αd+p.αr)*sin(k[2])
    g_z = 1.0im*p.Δz
    gg = [eps, Vab, g_x, g_y, g_z]
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
    eps = -p.t*(cos(k[1])+cos(k[2])) + p.mu
    Vab = -2p.tl*cos(k[1]/2)*cos(k[2]/2)
    g_x = (p.αd-p.αr)*sin(k[2]) + p.hx
    g_y = (p.αd+p.αr)*sin(k[2])

    #Here, we set g_z = 0.0 because ForwardDiff cannot do auto-derivative of comeplex number. We suppose the non-Hermitian term is independent of k. 
    g_z = 0.0
    #1.0im*p.Δz
    gg = [eps, Vab, g_x, g_y, g_z]
    return gg
end
