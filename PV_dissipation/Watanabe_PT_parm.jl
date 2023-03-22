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
#=
function set_vx(k::Vector{Float64},p::Parm)
    eps_vx::Float64 = 2.0p.t_i*(-p.Pr*sin(k'*a1) + 0.5sin(k'*a2) - 0.5sin(k'*a3))
    gx_vx::Float64 = p.a_u*(cos(k'*a3) - cos(k'*a2))/4 
    gy_vx::Float64 = -p.a_u * (cos(k'*a1) + 0.5*(cos(k'*a3) + cos(k'*a2))/2)/sqrt(3.0)
    gz_vx::Float64 = 2p.a_d*(cos(k'*a1) - 0.5cos(k'*a2) - 0.5cos(k'*a3))/(3.0*sqrt(3.0))
    gg_x = [eps_vx, gx_vx, gy_vx, gz_vx]
    Vx::Array{ComplexF64,2} = gg_x' * sigma
    return Vx
end

function set_vy(k::Vector{Float64},p::Parm)
    eps_vy::Float64 = sqrt(3.0)*p.t_i*(-sin(k'*a2) - sin(k'*a3))
    gx_vy::Float64 = sqrt(3.0)*p.a_u*(cos(k'*a3) + cos(k'*a2))/4 
    gy_vy::Float64 = -p.a_u * ((cos(k'*a3) - cos(k'*a2))/4)
    gz_vy::Float64 = p.a_d*(cos(k'*a2) - cos(k'*a3))/(3.0)
    gg_y = [eps_vy, gx_vy, gy_vy, gz_vy]
    Vy::Array{ComplexF64,2} = gg_y' * sigma
    return Vy
end

function set_vxx(k::Vector{Float64},p::Parm)
    eps_vxx::Float64 = 2.0p.t_i*(-p.Pr*cos(k'*a1) - 0.25cos(k'*a2) - 0.25cos(k'*a3))
    gx_vxx::Float64 = p.a_u*(-sin(k'*a3) - sin(k'*a2))/8 
    gy_vxx::Float64 = -p.a_u * (-sin(k'*a1) + 0.25*(-sin(k'*a3) + sin(k'*a2))/2)/sqrt(3.0)
    gz_vxx::Float64 = 2p.a_d*(-sin(k'*a1) - 0.25sin(k'*a2) + 0.25sin(k'*a3))/(3.0*sqrt(3.0))
    gg_xx = [eps_vxx, gx_vxx, gy_vxx, gz_vxx]
    Vxx::Array{ComplexF64,2} = gg_xx' * sigma
    return Vxx
end

function set_vxy(k::Vector{Float64},p::Parm)
    eps_vxy::Float64 = sqrt(3.0)*p.t_i*(cos(k'*a2) - cos(k'*a3))/2
    gx_vxy::Float64 = sqrt(3.0)*p.a_u*(-sin(k'*a3) + sin(k'*a2))/8 
    gy_vxy::Float64 = -p.a_u * ((-sin(k'*a3) - sin(k'*a2))/8)
    gz_vxy::Float64 = p.a_d*(sin(k'*a2) + sin(k'*a3))/(6.0)
    gg_xy = [eps_vxy, gx_vxy, gy_vxy, gz_vxy]
    Vxy::Array{ComplexF64,2} = gg_xy' * sigma
    return Vxy
end

function set_vyy(k::Vector{Float64},p::Parm)
    eps_vyy::Float64 = 1.5*p.t_i*(-cos(k'*a2) - cos(k'*a3))
    gx_vyy::Float64 = 3.0*p.a_u*(-sin(k'*a3) - sin(k'*a2))/8 
    gy_vyy::Float64 = -p.a_u * sqrt(3.0) * ((-sin(k'*a3) + sin(k'*a2))/8)
    gz_vyy::Float64 = p.a_d*(-sin(k'*a2) + sin(k'*a3))/(2.0*sqrt(3.0))
    gg_yy = [eps_vyy, gx_vyy, gy_vyy, gz_vyy]
    Vyy::Array{ComplexF64,2} = gg_yy' * sigma
    return Vyy
end

function set_vxxx(k::Vector{Float64},p::Parm)
    eps::Float64 = 2.0p.t_i*(p.Pr*sin(k'*a1) - 0.125sin(k'*a2) + 0.125sin(k'*a3))
    gx::Float64 = p.a_u*(-cos(k'*a3) + cos(k'*a2))/16 
    gy::Float64 = -p.a_u * (-cos(k'*a1) + 0.125*(-cos(k'*a3) - cos(k'*a2))/2)/sqrt(3.0)
    gz::Float64 = 2p.a_d*(-cos(k'*a1) + 0.125cos(k'*a2) + 0.125cos(k'*a3))/(3.0*sqrt(3.0))
    gg_xxx = [eps, gx, gy, gz]
    Vxxx::Array{ComplexF64,2} = gg_xxx' * sigma
    return Vxxx
end

function set_vxxy(k::Vector{Float64},p::Parm)
    eps::Float64 = sqrt(3.0)*p.t_i*(sin(k'*a2) + sin(k'*a3))/4
    gx::Float64 = sqrt(3.0)*p.a_u*(-cos(k'*a3) - cos(k'*a2))/16 
    gy::Float64 = -p.a_u * ((-cos(k'*a3) + cos(k'*a2))/16)
    gz::Float64 = p.a_d*(-cos(k'*a2) + cos(k'*a3))/(12.0)
    gg_xxy = [eps, gx, gy, gz]
    Vxxy::Array{ComplexF64,2} = gg_xxy' * sigma
    return Vxxy
end

function set_vxyy(k::Vector{Float64},p::Parm)
    eps::Float64 = 0.75*p.t_i*(-sin(k'*a2) + sin(k'*a3))
    gx::Float64 = 3.0*p.a_u*(-cos(k'*a3) + cos(k'*a2))/16 
    gy::Float64 = -p.a_u * sqrt(3.0) * ((-cos(k'*a3) - cos(k'*a2))/16)
    gz::Float64 = p.a_d*(cos(k'*a2) + cos(k'*a3))/(4.0*sqrt(3.0))
    gg_xyy = [eps, gx, gy, gz]
    Vxyy::Array{ComplexF64,2} = gg_xyy' * sigma
    return Vxyy
end

function set_vyyy(k::Vector{Float64},p::Parm)
    eps::Float64 = 0.75*p.t_i*(sin(k'*a2) + sin(k'*a3))*sqrt(3.0)
    gx::Float64 = 3.0*sqrt(3.0)*p.a_u*(-cos(k'*a3) - cos(k'*a2))/16 
    gy::Float64 = -p.a_u * 3.0 * ((-cos(k'*a3) + cos(k'*a2))/16)
    gz::Float64 = p.a_d*(-cos(k'*a2) + cos(k'*a3))/(4.0)
    gg_yyy = [eps, gx, gy, gz]
    Vyyy::Array{ComplexF64,2} = gg_yyy' * sigma
    return Vyyy
end

# set Hamiltoniann and velocity operator
function HandV(k0::NTuple{2, Float64},p::Parm)
    k = [k0[1], k0[2]]

    H = set_H(k,p)

    if(p.α == 'X')
        Va = set_vx(k,p)
        if(p.β == 'X')
            Vb = set_vx(k,p)
            Vab = set_vxx(k,p)
            if(p.γ == 'X')
                Vc = set_vx(k,p)
                Vbc = set_vxx(k,p)
                Vca = set_vxx(k,p)
                Vabc = set_vxxx(k,p)
            elseif(p.γ == 'Y')
                Vc = set_vy(k,p)
                Vbc = set_vxy(k,p)
                Vca = set_vxy(k,p)
                Vabc = set_vxxy(k,p)
            end
        elseif(p.β == 'Y')
            Vb = set_vy(k,p)
            Vab = set_vxy(k,p)
            if(p.γ == 'X')
                Vc = set_vx(k,p)
                Vbc = set_vxy(k,p)
                Vca = set_vxx(k,p)
                Vabc = set_vxxy(k,p)
            elseif(p.γ == 'Y')
                Vc = set_vy(k,p)
                Vbc = set_vyy(k,p)
                Vca = set_vxy(k,p)
                Vabc = set_vxyy(k,p)
            end
        end
    elseif(p.α == 'Y')
        Va = set_vy(k,p)
        if(p.β == 'X')
            Vb = set_vx(k,p)
            Vab = set_vxy(k,p)
            if(p.γ == 'X')
                Vc = set_vx(k,p)
                Vbc = set_vxx(k,p)
                Vca = set_vxy(k,p)
                Vabc = set_vxxy(k,p)
            elseif(p.γ == 'Y')
                Vc = set_vy(k,p)
                Vbc = set_vxy(k,p)
                Vca = set_vyy(k,p)
                Vabc = set_vxyy(k,p)
            end
        elseif(p.β == 'Y')
            Vb = set_vy(k,p)
            Vab = set_vyy(k,p)
            if(p.γ == 'X')
                Vc = set_vx(k,p)
                Vbc = set_vxy(k,p)
                Vca = set_vxy(k,p)
                Vabc = set_vxyy(k,p)
            elseif(p.γ == 'Y')
                Vc = set_vy(k,p)
                Vbc = set_vyy(k,p)
                Vca = set_vyy(k,p)
                Vabc = set_vyyy(k,p)
            end
        end
    end
    
    E::Array{ComplexF64,1} = zeros(2)

    return H, Va, Vb, Vc, Vab, Vbc, Vca, Vabc, E 
end=#

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


#=
using ForwardDiff


function set_vx_fd(k,p::Parm)
    m(k) = set_H_v(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    #gg = (ForwardDiff.jacobian(k -> set_H_v(k,p), k)[1])[:,1]
    #you can also write as below
    #k0 = k[1]
    #gg = ForwardDiff.jacobian(k0 -> set_H_v([k0,k[2],k[3]],p), k0)[1]
    #V::Array{ComplexF64,2} = gg' * sigma
    return gg
end

function set_vy_fd(k,p::Parm)
    m(k) = set_H_v(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,2]
    #gg = (ForwardDiff.jacobian(k -> set_H_v(k,p), k)[1])[:,2]
    #you can also write as below
    #k0 = k[2]
    #gg = ForwardDiff.jacobian(k0 -> set_H_v([k[1],k0,k[3]],p), k0)[1]
    #V::Array{ComplexF64,2} = gg' * sigma
    return gg
end

function set_vxx_fd(k,p::Parm)
    m(k) = set_vx_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    #gg = ForwardDiff.jacobian(k -> set_vx_fd(k,p), k)[1]
    #you can also write as below
    #k0 = k[1]
    #gg = ForwardDiff.jacobian(k0 -> set_vx_fd([k0,k[2],k[3]],p), k0)[1]
    #V::Array{ComplexF64,2} = gg' * sigma
    return gg
end

function set_vxy_fd(k,p::Parm)
    m(k) = set_vy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    #gg = (ForwardDiff.jacobian(k -> set_vy_fd(k,p), k)[1])[:,1]
    #you can also write as below
    #k0 = k[1]
    #gg = ForwardDiff.jacobian(k0 -> set_vy_fd([k0,k[2],k[3]],p), k0)[1]
    #V::Array{ComplexF64,2} = gg' * sigma
    return gg
end

function set_vyy_fd(k,p::Parm)
    m(k) = set_vy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,2]
    #gg = (ForwardDiff.jacobian(k -> set_vy_fd(k,p), k)[1])[:,2]
    #k0 = k[2]
    #gg = ForwardDiff.jacobian(k0 -> set_vy_fd([k[1],k0,k[3]],p), k0)[1]
    #V::Array{ComplexF64,2} = gg' * sigma
    return gg
end

function set_vxxx_fd(k,p::Parm)
    m(k) = set_vxx_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    #gg = (ForwardDiff.jacobian(k -> set_vxx_fd(k,p), k)[1])[:,1]
    #k0 = k[1]
    #gg = ForwardDiff.jacobian(k0 -> set_vxx_fd([k0,k[2],k[3]],p), k0)[1]
    #V::Array{ComplexF64,2} = gg' * sigma
    return gg
end

function set_vxxy_fd(k,p::Parm)
    m(k) = set_vxy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    #gg = (ForwardDiff.jacobian(k -> set_vxy_fd(k,p), k)[1])[:,1]
    #you can also write as below
    #k0 = k[1]
    #gg = ForwardDiff.jacobian(k0 -> set_vxy_fd([k0,k[2],k[3]],p), k0)[1]
    #V::Array{ComplexF64,2} = gg' * sigma
    return gg
end

function set_vxyy_fd(k,p::Parm)
    m(k) = set_vyy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    #gg = (ForwardDiff.jacobian(k -> set_vyy_fd(k,p), k)[1])[:,1]
    #k0 = k[1]
    #gg = ForwardDiff.jacobian(k0 -> set_vyy_fd([k0,k[2],k[3]],p), k0)[1]
    #V::Array{ComplexF64,2} = gg' * sigma
    return gg
end

function set_vyyy_fd(k,p::Parm)
    m(k) = set_vyy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,2]
    #gg = (ForwardDiff.jacobian(k -> set_vyy_fd(k,p), k)[1])[:,2]
    #you can also write as below
    #k0 = k[2]
    #gg = ForwardDiff.jacobian(k0 -> set_vx_fd([k[1],k0,k[3]],p), k0)[1]
    #V::Array{ComplexF64,2} = gg' * sigma
    return gg
end

function VtoM(v::Vector{Float64})
    M::Array{ComplexF64,2} = v' * sigma 
    return M
end

function HandV_fd(k0::NTuple{2, Float64},p::Parm)
    k = [k0[1], k0[2]]

    H = set_H(k,p)

    if(p.α == 'X')
        Va = VtoM(set_vx_fd(k,p))#set_vx(k,p)
        if(p.β == 'X')
            Vb = VtoM(set_vx_fd(k,p))
            Vab = VtoM(set_vxx_fd(k,p))
            if(p.γ == 'X')
                Vc = VtoM(set_vx_fd(k,p))
                Vbc = VtoM(set_vxx_fd(k,p))
                Vca = VtoM(set_vxx_fd(k,p))
                Vabc = VtoM(set_vxxx_fd(k,p))
            elseif(p.γ == 'Y')
                Vc = VtoM(set_vy_fd(k,p))
                Vbc = VtoM(set_vxy_fd(k,p))
                Vca = VtoM(set_vxy_fd(k,p))
                Vabc = VtoM(set_vxxy_fd(k,p))
            end
        elseif(p.β == 'Y')
            Vb = VtoM(set_vy_fd(k,p))
            Vab = VtoM(set_vxy_fd(k,p))
            if(p.γ == 'X')
                Vc = VtoM(set_vx_fd(k,p))
                Vbc = VtoM(set_vxy_fd(k,p))
                Vca = VtoM(set_vxx_fd(k,p))
                Vabc = VtoM(set_vxxy_fd(k,p))
            elseif(p.γ == 'Y')
                Vc = VtoM(set_vy_fd(k,p))
                Vbc = VtoM(set_vyy_fd(k,p))
                Vca = VtoM(set_vxy_fd(k,p))
                Vabc = VtoM(set_vxyy_fd(k,p))
            end
        end
    elseif(p.α == 'Y')
        Va = VtoM(set_vy_fd(k,p))#set_vx(k,p)
        if(p.β == 'X')
            Vb = VtoM(set_vx_fd(k,p))
            Vab = VtoM(set_vxy_fd(k,p))
            if(p.γ == 'X')
                Vc = VtoM(set_vx_fd(k,p))
                Vbc = VtoM(set_vxx_fd(k,p))
                Vca = VtoM(set_vxy_fd(k,p))
                Vabc = VtoM(set_vxxy_fd(k,p))
            elseif(p.γ == 'Y')
                Vc = VtoM(set_vy_fd(k,p))
                Vbc = VtoM(set_vxy_fd(k,p))
                Vca = VtoM(set_vyy_fd(k,p))
                Vabc = VtoM(set_vxyy_fd(k,p))
            end
        elseif(p.β == 'Y')
            Vb = VtoM(set_vy_fd(k,p))
            Vab = VtoM(set_vyy_fd(k,p))
            if(p.γ == 'X')
                Vc = VtoM(set_vx_fd(k,p))
                Vbc = VtoM(set_vxy_fd(k,p))
                Vca = VtoM(set_vxy_fd(k,p))
                Vabc = VtoM(set_vxyy_fd(k,p))
            elseif(p.γ == 'Y')
                Vc = VtoM(set_vy_fd(k,p))
                Vbc = VtoM(set_vyy_fd(k,p))
                Vca = VtoM(set_vyy_fd(k,p))
                Vabc = VtoM(set_vyyy_fd(k,p))
            end
        end
    end
    
    E::Array{ComplexF64,1} = zeros(2)

    return H, Va, Vb, Vc, Vab, Vbc, Vca, Vabc, E 
end
=#