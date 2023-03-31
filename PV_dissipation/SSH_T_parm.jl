#using Distributed
#addprocs(30)

#Parm(t_i, a_u, a_d, Pr, mu, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ)
struct Parm
    μ::Float64
    t1::Float64
    t2::Float64
    Q::Float64
    J::Float64
    m::Float64
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
    μ = parse(Float64,arg[1])
    t1 = parse(Float64,arg[2])
    t2 = parse(Float64,arg[3])
    Q = pi/4
    J = parse(Float64,arg[4])
    m = parse(Float64,arg[5])
    eta = parse(Float64,arg[6])
    T = parse(Float64,arg[7])
    K_SIZE = parse(Int,arg[8])
    W_MAX = parse(Float64,arg[9])
    W_in = parse(Float64,arg[10])
    W_SIZE = parse(Int,arg[11])
    abc::Vector{Int} = [parse(Int,arg[12]), parse(Int,arg[13]), parse(Int,arg[14])]
    #α = (arg[15])[1]
    #β = (arg[16])[1]
    #γ = (arg[17])[1]

    #return μ, t1, t2, J, m, eta, T, hx, Δz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ
    return μ, t1, t2, Q, J, m, eta, T, K_SIZE, W_MAX, W_in, W_SIZE, abc, 2
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
    μ = parse(Float64,arg[1])
    t1 = parse(Float64,arg[2])
    t2 = parse(Float64,arg[3])
    Q = pi/4
    J = parse(Float64,arg[4])
    m = parse(Float64,arg[5])
    eta = parse(Float64,arg[6])
    T = parse(Float64,arg[7])
    K_SIZE = parse(Int,arg[8])
    W_MAX = parse(Float64,arg[9])
    W_in = Win
    W_SIZE = parse(Int,arg[11])
    abc::Vector{Int} = [parse(Int,arg[12]), parse(Int,arg[13]), parse(Int,arg[14])]
    #α = (arg[15])[1]
    #β = (arg[16])[1]
    #γ = (arg[17])[1]

    #return μ, t1, t2, J, m, eta, T, hx, Δz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ
    return μ, t1, t2, Q, J, m, eta, T, K_SIZE, W_MAX, W_in, W_SIZE, abc, 2
end


#a1 = [1.0, 0.0]
#a2 = [-0.5, sqrt(3.0)/2]
#a3 = [0.5, sqrt(3.0)/2]

sigma = [[1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0], [0.0 -1.0im; 1.0im 0.0], [1.0 0.0; 0.0 -1.0]]
#[[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0], [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0; 1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0], [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 -1.0; 0.0 0.0 -1.0 0.0], [0.0 -1.0im 0.0 0.0; 1.0im 0.0 0.0 0.0; 0.0 0.0 0.0 1.0im; 0.0 0.0 -1.0im 0.0], [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 -1.0 0.0; 0.0 0.0 0.0 -1.0]]


function get_kk(K_size::Int)
    kk::Vector{Vector{Float64}} = []
    k = collect(-pi:2pi/K_size:pi)
    for kx in k#, ky in k
        #In order to use ForwardDiff.jacobian when calculating the velocity operators, we set k as the vector  
        push!(kk, [kx,0])
    end
    return kk
end

#functions to set Hamiltonian and Velocity operators
function set_H(k::Vector{Float64},p::Parm)
    eps = -p.t1*(cos(k[1]-p.Q/2)+cos(k[1]+p.Q/2))-p.t2*(cos(2*(k[1]-p.Q/2))+cos(2*(k[1]+p.Q/2)))+p.μ
    g_x = -p.J*sqrt(1.0-p.m^2)
    g_y = 0.0
    g_z = -p.J*p.m -p.t1*(cos(k[1]-p.Q/2)-cos(k[1]+p.Q/2))-p.t2*(cos(2*(k[1]-p.Q/2))-cos(2*(k[1]+p.Q/2)))
    gg = [eps, g_x, g_y, g_z]
    H::Array{ComplexF64,2} =  sigma' * gg
    return H
end


#If you want to use ForwardDiff
function set_H_v(k,p::Parm)
    eps = -p.t1*(cos(k[1]-p.Q/2)+cos(k[1]+p.Q/2))-p.t2*(cos(2*(k[1]-p.Q/2))+cos(2*(k[1]+p.Q/2)))+p.μ
    g_x = -p.J*sqrt(1.0-p.m^2)
    g_y = 0.0
    g_z = -p.J*p.m -p.t1*(cos(k[1]-p.Q/2)-cos(k[1]+p.Q/2))-p.t2*(cos(2*(k[1]-p.Q/2))-cos(2*(k[1]+p.Q/2)))
    gg = [eps, g_x, g_y, g_z]
    return gg
end
