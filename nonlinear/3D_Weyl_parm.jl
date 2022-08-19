using Distributed
#addprocs(30)

#Parm(t_i, a_u, a_d, Pr, mu, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_SIZE)
@everywhere struct Parm
    t1::Float64
    t2::Float64
    delta::Float64
    mu::Float64
    kw::Float64
    eta::Float64
    T::Float64
    K_SIZE::Int
    W_MAX::Float64
    W_in::Float64
    W_SIZE::Int
    α::Char
    β::Char
    γ::Char
end

@everywhere function set_parm(arg::Array{String,1})
    t1 = parse(Float64,arg[1])
    t2 = parse(Float64,arg[2])
    delta = parse(Float64,arg[3])
    mu = parse(Float64,arg[4])
    kw = parse(Float64,arg[5])
    eta = parse(Float64,arg[6])
    T = parse(Float64,arg[7])
    K_SIZE = parse(Int,arg[8])
    W_MAX = parse(Float64,arg[9])
    W_in = parse(Float64,arg[10])
    W_SIZE = parse(Int,arg[11])
    α = parse(Char, arg[12])
    β = parse(Char, arg[13])
    γ = parse(Char, arg[14])

    return t1, t2, delta, mu, kw, eta, T, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ
end

@everywhere function set_parm_mudep(arg::Array{String,1}, mu0::Float64)
    t1 = parse(Float64,arg[1])
    t2 = parse(Float64,arg[2])
    delta = parse(Float64,arg[3])
    mu = mu0
    kw = parse(Float64,arg[5])
    eta = parse(Float64,arg[6])
    T = parse(Float64,arg[7])
    K_SIZE = parse(Int,arg[8])
    W_MAX = parse(Float64,arg[9])
    W_in = parse(Float64,arg[10])
    W_SIZE = parse(Int,arg[11])
    α = parse(Char, arg[12])
    β = parse(Char, arg[13])
    γ = parse(Char, arg[14])

    return t1, t2, delta, mu, kw, eta, T, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ
end

@everywhere function set_parm_etadep(arg::Array{String,1}, eta0::Float64)
    t1 = parse(Float64,arg[1])
    t2 = parse(Float64,arg[2])
    delta = parse(Float64,arg[3])
    mu = parse(Float64,arg[4])
    kw = parse(Float64,arg[5])
    eta = eta0
    T = parse(Float64,arg[7])
    K_SIZE = parse(Int,arg[8])
    W_MAX = parse(Float64,arg[9])
    W_in = parse(Float64,arg[10])
    W_SIZE = parse(Int,arg[11])
    α = parse(Char, arg[12])
    β = parse(Char, arg[13])
    γ = parse(Char, arg[14])

    return t1, t2, delta, mu, kw, eta, T, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ
end

@everywhere function set_parm_Wdep(arg::Array{String,1}, Win::Float64)
    t1 = parse(Float64,arg[1])
    t2 = parse(Float64,arg[2])
    delta = parse(Float64,arg[3])
    mu = parse(Float64,arg[4])
    kw = parse(Float64,arg[5])
    eta = parse(Float64,arg[6])
    T = parse(Float64,arg[7])
    K_SIZE = parse(Int,arg[8])
    W_MAX = parse(Float64,arg[9])
    W_in = Win
    W_SIZE = parse(Int,arg[11])
    α = parse(Char, arg[12])
    β = parse(Char, arg[13])
    γ = parse(Char, arg[14])

    return t1, t2, delta, mu, kw, eta, T, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ
end


@everywhere a1 = [1.0, 0.0]
@everywhere a2 = [-0.5, sqrt(3.0)/2]
@everywhere a3 = [0.5, sqrt(3.0)/2]
@everywhere sigma = [[1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0], [0.0 -1.0im; 1.0im 0.0], [1.0 0.0; 0.0 -1.0]]

#functions to set Hamiltonian and Velocity operators
@everywhere function set_H(k::Vector{Float64},p::Parm)
    e0 = p.t2*cos(k[1]+k[2]) + p.delta*cos(k[1]-k[2]) + p.mu
    ex = p.t1*(cos(p.kw)-cos(k[2])) + p.delta*(1.0-cos(k[3]))
    ey = p.t1*sin(k[3])
    ez = p.t1*(cos(p.kw)-cos(k[1])) + p.delta*(1.0-cos(k[3]))
    ee = [e0, ex, ey, ez]
    H::Array{ComplexF64,2} = ee' * sigma
    return H
end

@everywhere function set_vx(k::Vector{Float64},p::Parm)
    eps_vx::Float64 = -p.t2*sin(k[1]+k[2]) - p.delta*sin(k[1]-k[2])
    gx_vx::Float64 = 0.0
    gy_vx::Float64 = 0.0
    gz_vx::Float64 = p.t1*sin(k[1])
    gg_x = [eps_vx, gx_vx, gy_vx, gz_vx]
    Vx::Array{ComplexF64,2} = gg_x' * sigma
    return Vx
end

@everywhere function set_vy(k::Vector{Float64},p::Parm)
    eps_vy::Float64 = -p.t2*sin(k[1]+k[2]) + p.delta*sin(k[1]-k[2])
    gx_vy::Float64 = p.t1*sin(k[2])
    gy_vy::Float64 = 0.0
    gz_vy::Float64 = 0.0
    gg_y = [eps_vy, gx_vy, gy_vy, gz_vy]
    Vy::Array{ComplexF64,2} = gg_y' * sigma
    return Vy
end

@everywhere function set_vz(k::Vector{Float64},p::Parm)
    eps_vz::Float64 = 0.0
    gx_vz::Float64 = p.delta*sin(k[3])
    gy_vz::Float64 = p.t1*cos(k[3])
    gz_vz::Float64 = p.delta*sin(k[3])
    gg_z = [eps_vz, gx_vz, gy_vz, gz_vz]
    Vz::Array{ComplexF64,2} = gg_z' * sigma
    return Vz
end

@everywhere function set_vxx(k::Vector{Float64},p::Parm)
    Vxx::Array{ComplexF64,2} = (-p.t2*cos(k[1]+k[2]) - p.delta*cos(k[1]-k[2]))*sigma[1] + (p.t1*cos(k[1]))*sigma[4]
    return Vxx
end

@everywhere function set_vxy(k::Vector{Float64},p::Parm)
    Vxy::Array{ComplexF64,2} = (-p.t2*cos(k[1]+k[2]) + p.delta*cos(k[1]-k[2]))*sigma[1]
    return Vxy
end

@everywhere function set_vyy(k::Vector{Float64},p::Parm)
    Vyy::Array{ComplexF64,2} = (-p.t2*cos(k[1]+k[2]) - p.delta*cos(k[1]-k[2]))*sigma[1] + (p.t1*cos(k[2]))*sigma[2]
    return Vyy
end

@everywhere function set_vzx(k::Vector{Float64},p::Parm)
    Vzx = zeros(ComplexF64,2,2)
    return Vzx
end
@everywhere function set_vyz(k::Vector{Float64},p::Parm)
    Vyz = zeros(ComplexF64,2,2)
    return Vyz
end
@everywhere function set_vzz(k::Vector{Float64},p::Parm)
    Vzz::Array{ComplexF64,2} = p.delta*cos(k[3])*(sigma[2]+sigma[4]) - p.t1*sin(k[3])*sigma[3]
    return Vzz
end

@everywhere function set_vxxx(k::Vector{Float64},p::Parm)
    Vxxx::Array{ComplexF64,2} = (p.t2*sin(k[1]+k[2]) + p.delta*sin(k[1]-k[2]))*sigma[1] - (p.t1*sin(k[1]))*sigma[4]
    return Vxxx
end

@everywhere function set_vxxy(k::Vector{Float64},p::Parm)
    Vxxy::Array{ComplexF64,2} = (p.t2*sin(k[1]+k[2]) - p.delta*sin(k[1]-k[2]))*sigma[1]
    return Vxxy
end

@everywhere function set_vxxz(k::Vector{Float64},p::Parm)
    Vxxz = zeros(ComplexF64,2,2)
    return Vxxz
end

@everywhere function set_vxyy(k::Vector{Float64},p::Parm)
    Vxyy::Array{ComplexF64,2} = (p.t2*sin(k[1]+k[2]) + p.delta*sin(k[1]-k[2]))*sigma[1]
    return Vxyy
end
@everywhere function set_vxyz(k::Vector{Float64},p::Parm)
    Vxyz = zeros(ComplexF64,2,2)
    return Vxyz
end
@everywhere function set_vxzz(k::Vector{Float64},p::Parm)
    Vxzz = zeros(ComplexF64,2,2)
    return Vxzz
end
@everywhere function set_vyyy(k::Vector{Float64},p::Parm)
    Vyyy::Array{ComplexF64,2} = (p.t2*sin(k[1]+k[2]) - p.delta*sin(k[1]-k[2]))*sigma[1] - (p.t1*sin(k[2]))*sigma[2]
    return Vyyy
end
@everywhere function set_vyyz(k::Vector{Float64},p::Parm)
    Vyyz = zeros(ComplexF64,2,2)
    return Vyyz
end
@everywhere function set_vyzz(k::Vector{Float64},p::Parm)
    Vyzz = zeros(ComplexF64,2,2)
    return Vyzz
end
@everywhere function set_vzzz(k::Vector{Float64},p::Parm)
    Vzzz::Array{ComplexF64,2} = -p.delta*sin(k[3])*(sigma[2]+sigma[4]) - p.t1*cos(k[3])*sigma[3]
    return Vxzz
end


# set Hamiltoniann and velocity operator

# set Hamiltoniann and velocity operator
@everywhere function HandV(k0::NTuple{3, Float64},p::Parm)
    k = [k0[1], k0[2], k0[3]]

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
            else
                Vc = set_vz(k,p)
                Vbc = set_vzx(k,p)
                Vca = set_vzx(k,p)
                Vabc = set_vxxz(k,p)
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
            else
                Vc = set_vz(k,p)
                Vbc = set_vyz(k,p)
                Vca = set_vzx(k,p)
                Vabc = set_vxyz(k,p)
            end
        else
            Vb = set_vz(k,p)
            Vab = set_vzx(k,p)
            if(p.γ == 'X')
                Vc = set_vx(k,p)
                Vbc = set_vzx(k,p)
                Vca = set_vxx(k,p)
                Vabc = set_vxxz(k,p)
            elseif(p.γ == 'Y')
                Vc = set_vy(k,p)
                Vbc = set_vyz(k,p)
                Vca = set_vxy(k,p)
                Vabc = set_vxyz(k,p)
            else
                Vc = set_vz(k,p)
                Vbc = set_vzz(k,p)
                Vca = set_vzx(k,p)
                Vabc = set_vxzz(k,p)
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
            else
                Vc = set_vz(k,p)
                Vbc = set_vzx(k,p)
                Vca = set_vyz(k,p)
                Vabc = set_vxyz(k,p)
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
            else
                Vc = set_vz(k,p)
                Vbc = set_vyz(k,p)
                Vca = set_vyz(k,p)
                Vabc = set_vyyz(k,p)
            end
        else
            Vb = set_vz(k,p)
            Vab = set_vzy(k,p)
            if(p.γ == 'X')
                Vc = set_vx(k,p)
                Vbc = set_vzx(k,p)
                Vca = set_vxy(k,p)
                Vabc = set_vxyz(k,p)
            elseif(p.γ == 'Y')
                Vc = set_vy(k,p)
                Vbc = set_vyz(k,p)
                Vca = set_vyy(k,p)
                Vabc = set_vyyz(k,p)
            else
                Vc = set_vz(k,p)
                Vbc = set_vzz(k,p)
                Vca = set_vyz(k,p)
                Vabc = set_vyzz(k,p)
            end
        end
    else
        Va = set_vz(k,p)
        if(p.β == 'X')
            Vb = set_vx(k,p)
            Vab = set_vzx(k,p)
            if(p.γ == 'X')
                Vc = set_vx(k,p)
                Vbc = set_vxx(k,p)
                Vca = set_vzx(k,p)
                Vabc = set_vxxz(k,p)
            elseif(p.γ == 'Y')
                Vc = set_vy(k,p)
                Vbc = set_vxy(k,p)
                Vca = set_vyz(k,p)
                Vabc = set_vxyz(k,p)
            else
                Vc = set_vz(k,p)
                Vbc = set_vzx(k,p)
                Vca = set_vzz(k,p)
                Vabc = set_vxzz(k,p)
            end
        elseif(p.β == 'Y')
            Vb = set_vy(k,p)
            Vab = set_vyz(k,p)
            if(p.γ == 'X')
                Vc = set_vx(k,p)
                Vbc = set_vxy(k,p)
                Vca = set_vzx(k,p)
                Vabc = set_vxyz(k,p)
            elseif(p.γ == 'Y')
                Vc = set_vy(k,p)
                Vbc = set_vyy(k,p)
                Vca = set_vyz(k,p)
                Vabc = set_vyyz(k,p)
            else
                Vc = set_vz(k,p)
                Vbc = set_vyz(k,p)
                Vca = set_vzz(k,p)
                Vabc = set_vyzz(k,p)
            end
        else
            Vb = set_vz(k,p)
            Vab = set_vzz(k,p)
            if(p.γ == 'X')
                Vc = set_vx(k,p)
                Vbc = set_vzx(k,p)
                Vca = set_vzx(k,p)
                Vabc = set_vxzz(k,p)
            elseif(p.γ == 'Y')
                Vc = set_vy(k,p)
                Vbc = set_vyz(k,p)
                Vca = set_vyz(k,p)
                Vabc = set_vyzz(k,p)
            else
                Vc = set_vz(k,p)
                Vbc = set_vzz(k,p)
                Vca = set_vzz(k,p)
                Vabc = set_vzzz(k,p)
            end
        end
    end
    
    E::Array{ComplexF64,1} = zeros(2)

    return H, Va, Vb, Vc, Vab, Vbc, Vca, Vabc, E 
end

#create mesh over the BZ
@everywhere function get_kk(K_SIZE::Int)
    kk = Vector{NTuple{2, Float64}}(undef,0)
    dk = 4pi/(3K_SIZE)
    #dk2 = 2.0/(3*sqrt(3.0)*K_SIZE*K_SIZE)
    for i in collect(dk:dk:4pi/3)
        
        for j in collect(0:dk:4pi/3)
            k = j*a1 + i*a2
            push!(kk,(k[1],k[2]))
        end
        
        
        for j in collect(dk:dk:4pi/3)
            if (i+j) < (4pi/3+dk)
                k = -j*a1 + i*a3
                push!(kk,(k[1],k[2]))
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
        push!(kk,(k[1],k[2]))
    end
    return kk
end