using LinearAlgebra
using DataFrames
using CSV
#Parm(t_i, a_u, a_d, Pr, mu, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE, α, β, γ)
struct Parm
    t_i::Float64
    t_b::Float64
    Δ::Float64
    mu::Float64
    eta::Float64
    K_SIZE::Int
    dk2::Float64
    Ω::Float64
    T::Float64
    Σw::Vector{Matrix{ComplexF64}}
    γ1::Vector{Matrix{ComplexF64}}
    γ2::Vector{Matrix{ComplexF64}}
    W_max::Float64
    w_size::Int
    w_mesh::Vector{Float64}
    dw::Float64
    α::Char
    β::Char
    γ::Char
end

function set_parm_DC(arg::Array{String,1})
    t_i = parse(Float64,arg[1])
    t_b = parse(Float64,arg[2])
    Δ = parse(Float64,arg[3])
    mu0 = parse(Float64,arg[4])
    eta = parse(Float64,arg[5])
    K_SIZE = parse(Int,arg[6])
    dk2 = 1.0/(K_SIZE^2)
    Ω::Float64 = 0.0
    T = parse(Float64,arg[7])

    SS = Matrix(CSV.read(arg[8], DataFrame))
    w_mesh::Vector{Float64} = SS[:,1]
    
    
    W_MAX::Float64 = abs(w_mesh[1])
    w_size::Int = length(w_mesh)
    dw::Float64 = w_mesh[2]-w_mesh[1]
    Σw::Vector{Matrix{ComplexF64}} = []
    for ww in 1:w_size
        Sm = zeros(ComplexF64, 2, 2)
        for it in 1:4 
            Sm += (1.0im*SS[ww, 2it] + SS[ww, 2it+1]) * sigma[it]
        end
        push!(Σw,Sm)
    end
    γ1::Vector{Matrix{ComplexF64}}=[]
    γ2::Vector{Matrix{ComplexF64}}=[]
    for ww in 1:w_size
        if(ww == w_size)
            g1 = (Σw[ww]-Σw[ww-1])/dw
        else
            g1 = (Σw[ww+1]-Σw[ww])/dw
        end
        push!(γ1, g1)
    end
    for ww in 1:w_size
        if(ww == w_size)
            g2 = (γ1[ww]-γ1[ww-1])/dw
        else
            g2 = (γ1[ww+1]-γ1[ww])/dw
        end
        push!(γ2, g2)
    end
    α = (arg[10])[1]
    β = (arg[11])[1]
    γ = (arg[12])[1]

    return t_i, t_b, Δ, mu0, eta, K_SIZE, dk2, Ω, T, Σw, γ1, γ2, W_MAX, w_size, w_mesh, dw, α, β, γ
end

function set_parm_Wdep(arg::Array{String,1}, W::Float64)
    t_i = parse(Float64,arg[1])
    t_b = parse(Float64,arg[2])
    Δ = parse(Float64,arg[3])
    mu0 = parse(Float64,arg[4])
    eta = parse(Float64,arg[5])
    K_SIZE = parse(Int,arg[6])
    dk2 = 1.0/(K_SIZE^2)
    Ω = W
    T = parse(Float64,arg[7])

    SS = Matrix(CSV.read(arg[8], DataFrame))

    w_mesh::Vector{Float64} = SS[:,1]
    W_MAX::Float64 = abs(w_mesh[1])
    w_size::Int = length(w_mesh)

    Σw::Vector{Matrix{ComplexF64}} = []
    for ww in 1:w_size
        Sm = zeros(ComplexF64, 2, 2)
        for it in 1:4 
            Sm += (1.0im*SS[ww, 2it] + SS[ww, 2it+1]) * sigma[it]
        end
        push!(Σw,Sm)
    end
    γ1::Vector{Matrix{ComplexF64}}=[]
    γ2::Vector{Matrix{ComplexF64}}=[]
    dw::Float64 = w_mesh[2]-w_mesh[1]

    α = (arg[10])[1]
    β = (arg[11])[1]
    γ = (arg[12])[1]

    return t_i, t_b, Δ, mu0, eta, K_SIZE, dk2, Ω, T, Σw, γ1, γ2, W_MAX, w_size, w_mesh, dw, α, β, γ
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

function set_H_v(k,p::Parm)
    eps0 = p.t_i*cos(k[2]) + p.mu
    t_x = p.t_i*cos(k[1])
    t_y = p.t_b*sin(k[1])
    t_z = p.Δ 
    H = [eps0, t_x, t_y, t_z]
    return H
end

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

function HandV_fd(k::Vector{Float64},p::Parm)
    #k = [k0[1], k0[2]]

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

#=
function main(arg::Array{String,1})
    p = Parm(set_parm_Wdep(arg, 0.2)...)
    println(p.Σw[1])
end

@time main(ARGS)
=#