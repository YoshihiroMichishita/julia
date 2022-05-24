using Distributed
addprocs(30)

#Parm(t_i, t_e, Delta, mu, eta, Ni, U, T, hx, hy, hz, K_SIZE, W_MAX, W_SIZE, W_in)
@everywhere struct Parm
    t_i::Float64
    t_e::Float64
    Delta::Float64
    mu::Float64
    eta::Float64
    Ni::Float64
    U::Float64
    T::Float64
    hx::Float64
    hy::Float64
    hz::Float64
    K_SIZE::Int
    W_MAX::Float64
    W_SIZE::Int
    W_in::Float64
end

@everywhere mutable struct Hamiltonian
    Hk::Array{ComplexF64,2}
    Vx::Array{ComplexF64,2}
    Vxx::Array{ComplexF64,2}
    E::Array{ComplexF64,1}
end

@everywhere mutable struct Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    GRmA::Array{ComplexF64,2}
    GRp::Array{ComplexF64,2}
    GAp::Array{ComplexF64,2}
    GRm::Array{ComplexF64,2}
    GAm::Array{ComplexF64,2}
end

@everywhere using LinearAlgebra

@everywhere function Gk(w::Float64, Ham::Hamiltonian, p::Parm, eta::Float64)
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,2,2) + eta*Matrix{Complex{Float64}}(1.0im*I,2,2)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    GRmA::Array{ComplexF64,2} = GR - GA

    GRp0::Array{ComplexF64,2} = -Ham.Hk + (w+p.W_in)*Matrix{Complex{Float64}}(I,2,2) + eta*Matrix{Complex{Float64}}(1.0im*I,2,2)
    GRp::Array{ComplexF64,2} = inv(GRp0)
    GAp::Array{ComplexF64,2} = GRp'

    GRm0::Array{ComplexF64,2} = -Ham.Hk + (w-p.W_in)*Matrix{Complex{Float64}}(I,2,2) + eta*Matrix{Complex{Float64}}(1.0im*I,2,2)
    GRm::Array{ComplexF64,2} = inv(GRm0)
    GAm::Array{ComplexF64,2} = GRm'
    #dGR::Array{ComplexF64,2} = - GR * GR
    #dGA::Array{ComplexF64,2} = - GA * GA
    #ddGR::Array{ComplexF64,2} = 2.0* GR * GR * GR
    
    return GR, GA, GRmA, GRp, GAp, GRm, GAm
end

@everywhere function G_M(m::Int, p::Parm, Ham::Hamiltonian, eta::Float64)
    #Green関数のinverse
    wm = pi*(2m+1)*p.T
    GR0::Array{ComplexF64,2} = -Ham.Hk + wm*Matrix{Complex{Float64}}(1.0im*I,2,2) + eta*Matrix{Complex{Float64}}(1.0im*I,2,2)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA0::Array{ComplexF64,2} = -Ham.Hk + wm*Matrix{Complex{Float64}}(1.0im*I,2,2) - eta*Matrix{Complex{Float64}}(1.0im*I,2,2)
    GA::Array{ComplexF64,2} = inv(GA0)
    GRmA::Array{ComplexF64,2} = GR - GA
    dGR::Array{ComplexF64,2} = - GR * GR
    dGA::Array{ComplexF64,2} = - GA * GA
    ddGR::Array{ComplexF64,2} = 2.0* GR * GR * GR
    
    return GR, GA, GRmA, dGR, dGA, ddGR
end

@everywhere sigma = [[1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0], [0.0 -1.0im; 1.0im 0.0], [1.0 0.0; 0.0 -1.0]]


@everywhere function HandV(k0,p::Parm)
    
    eps::Float64 = -p.mu
    g_x::Float64 = p.t_i*cos(k0) - p.hx
    g_y::Float64 = p.t_e * sin(k0) - p.hy
    g_z::Float64 = p.Delta - p.hz
    gg = [eps, g_x, g_y, g_z]
    H::Array{ComplexF64,2} =  gg' * sigma

    eps_vx::Float64 = 0.0
    gx_vx::Float64 = -p.t_i*sin(k0) 
    gy_vx::Float64 = p.t_e * cos(k0)
    gz_vx::Float64 = 0.0
    gg_x = [eps_vx, gx_vx, gy_vx, gz_vx]
    Vx::Array{ComplexF64,2} = gg_x' * sigma

    eps_vxx::Float64 = 0.0
    gx_vxx::Float64 = -p.t_i*cos(k0) 
    gy_vxx::Float64 = -p.t_e * sin(k0)
    gz_vxx::Float64 = 0.0
    gg_xx = [eps_vxx, gx_vxx, gy_vxx, gz_vxx]
    Vxx::Array{ComplexF64,2} = gg_xx' * sigma

    E::Array{ComplexF64,1} = zeros(2)

    return H, Vx, Vxx, E 
end

@everywhere function HV_BI(H::Hamiltonian)

    H.E, BI::Array{ComplexF64,2} = eigen(H.Hk)
    H.Hk = [H.E[1] 0.0; 0.0 H.E[2]]
    Vx_BI::Array{ComplexF64,2} = BI' * H.Vx * BI
    Vxx_BI::Array{ComplexF64,2} = BI' * H.Vxx * BI
    

    H.Vx = Vx_BI
    H.Vxx = Vxx_BI
end

@everywhere f(e::Float64,T::Float64) = 1.0/(1.0+exp(e/T))
@everywhere df(e::Float64,T::Float64) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T

@everywhere function PV_calcu_ver(p::Parm, k::Float64, eta::Float64)
    dk::Float64 = 2pi/p.K_SIZE
    dw::Float64 = 2*p.W_MAX/p.W_SIZE
    Hk = Hamiltonian(HandV(k,p)...)
    Jxxx::Float64 = 0.0
    for q in collect(-pi:dk:pi-dk)
        Hq = Hamiltonian(HandV(k+q,p)...)
        for w in collect(-p.W_MAX:dw:p.W_MAX)
            Gkw = Green(Gk(w,Hk,p,eta)...)
            Gqw = Green(Gk(w,Hq,p,eta)...)
            Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GR * Gqw.GR * Hq.Vx * (Gqw.GRp + Gqw.GRm) * Hq.Vx * (Gqw.GR * Gkw.GR - Gqw.GA * Gkw.GA))) * f(w,p.T)
            Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRm * Gqw.GRm * Hq.Vx * Gqw.GRmA * Hq.Vx * Gqw.GAm * Gkw.GAm)) * f(w,p.T)
            Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRp * Gqw.GRp * Hq.Vx * Gqw.GRmA * Hq.Vx * Gqw.GAp * Gkw.GAp)) * f(w,p.T)
            Jxxx += dk * dw * imag(tr(Hk.Vx * (Gkw.GR * Gqw.GR - Gkw.GA * Gqw.GA) * Hq.Vx * (Gqw.GAp+Gqw.GAm) * Hq.Vx * Gqw.GA * Gkw.GA)) * f(w,p.T)
        end
    end
    return p.U*p.U*p.Ni*Jxxx/(4*pi*pi)
end

@everywhere function PV_calcu_simple(p::Parm, k::Float64, eta::Float64)
    dk::Float64 = 2pi/p.K_SIZE
    dw::Float64 = 2*p.W_MAX/p.W_SIZE
    Hk = Hamiltonian(HandV(k,p)...)
    Jxxx::Float64 = 0.0
    for w in collect(-p.W_MAX:dw:p.W_MAX)
        Gkw = Green(Gk(w,Hk,p,eta)...)
        Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GR * Hk.Vx * (Gkw.GRp + Gkw.GRm) * Hk.Vx * Gkw.GRmA)) * f(w,p.T)
        Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRm * Hk.Vx * Gkw.GRmA * Hk.Vx * Gkw.GAm)) * f(w,p.T)
        Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRp * Hk.Vx * Gkw.GRmA * Hk.Vx * Gkw.GAp)) * f(w,p.T)
        Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRmA * Hk.Vx * (Gkw.GAp+Gkw.GAm) * Hk.Vx * Gkw.GA)) * f(w,p.T)
    end
    return Jxxx/(4*pi*pi)
end

@everywhere function calcu_imp_scattering(p::Parm)
    dk::Float64 = 2pi/p.K_SIZE
    dw::Float64 = 2*p.W_MAX/p.W_SIZE
    n::Float64 = 0.0
    for kk in collect(-pi:dk:pi-dk)
        Hk = Hamiltonian(HandV(kk,p)...)
        for ww in collect(-p.W_MAX:dw:p.W_MAX)
            Gkw = Green(Gk(ww,Hk,p,p.eta)...)
            n += dw * dk * imag(tr(Gkw.GRmA)) * f(ww, p.T)
        end
    end
    return p.U*p.U*p.Ni*n
end

using DataFrames
using CSV
using Plots

function main(arg::Array{String,1})
    K_SIZE = parse(Int,arg[6])
    dk::Float64 = 2pi/K_SIZE
    kk = collect(-pi:dk:pi-dk)

    mu0 = collect(0.0:0.05:1.0)
    #collect(0.005:0.005:0.15)
    #[0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
    #collect(0.005:0.005:0.1)

    PV_XXX_ver_mu = zeros(Float64,length(mu0))
    PV_XXX_mu = zeros(Float64,length(mu0))
    eta_mu = zeros(Float64,length(mu0))

    for j in 1:length(mu0)
        #Parm(t_i, t_e, Delta, mu, eta, Ni, U, T, hx, hy, hz, K_SIZE, W_MAX, W_SIZE, W_in)
        p = Parm(parse(Float64,arg[1]), parse(Float64,arg[2]), parse(Float64,arg[3]), mu0[j], 0.02, 1.0, parse(Float64,arg[4]),parse(Float64,arg[5]), 0.0, 0.0, 0.0, parse(Int,arg[6]), 1.5, parse(Int,arg[7]),parse(Float64,arg[8]))
        eta_mu[j] = calcu_imp_scattering(p)
        new_eta::Float64 = eta_mu[j] + p.eta
        PV_XXX_ver_mu[j], PV_XXX_mu[j] = @distributed (+) for i in 1:p.K_SIZE 
            a1 = dk * PV_calcu_ver(p, kk[i], new_eta) 
            a2 = dk * PV_calcu_simple(p, kk[i], new_eta)
            [a1, a2]
        end
    end


    
    e1_k = zeros(Float64,length(kk))
    e2_k = zeros(Float64,length(kk))

    p = Parm(parse(Float64,arg[1]), parse(Float64,arg[2]), parse(Float64,arg[3]), 0.0, 0.02, 1.0, parse(Float64,arg[4]),parse(Float64,arg[5]), 0.0, 0.0, 0.0, parse(Int,arg[6]),
         1.5, parse(Int,arg[7]),parse(Float64,arg[8]))
    for i in 1:length(kk)
        Hk = Hamiltonian(HandV(kk[i],p)...)
        HV_BI(Hk)
        e1_k[i] = Hk.E[1]
        e2_k[i] = Hk.E[2]
    end

    save_data1 = DataFrame(k=kk, e1=e1_k, e2=e2_k)
    CSV.write("./disp_T002.csv", save_data1)
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data2 = DataFrame(mu=mu0, PV_simple=PV_XXX_mu, PV_ver=PV_XXX_ver_mu, imp_scatt=eta_mu)
    CSV.write("./PV_mudep_XXX_T002.csv", save_data2)

    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)
    p1 = plot(kk, e1_k, label="e1",xlabel="kx",ylabel="e",title="dispersion", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(kk, e2_k, width=4.0, marker=:circle, markersize = 4.8)
    savefig(p1,"./disp.png")

    p2 = plot(mu0, PV_XXX_mu, label="w/o vertex",xlabel="mu",ylabel="PV",title="μ-dependence", width=4.0, marker=:circle, markersize = 4.8)
    p2 = plot!(mu0, PV_XXX_ver_mu, width=4.0, marker=:circle, markersize = 4.8)
    savefig(p2,"./mudep_PV_XXX.png")
end

@time main(ARGS)