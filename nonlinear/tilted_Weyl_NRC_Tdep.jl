#Parm: m, lamda, hx, hy, hz, mu, eta, T, K_MAX, K_SIZE, W_in, W_MAX, W_SIZE
#p = Parm(ARGS[1], ARGS[2], 0.0, 0.0, 0.0, ARGS[3], T0, ARGS[4], 1.0, ARGS[5], ARGS[6] , ARGS[7], ARGS[8])
# ARGS =(m(tilting), lamda, eta, mu, K_SIZE, W_in, W_MAX, W_SIZE)

#import Pkg
#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("Plots")

using Distributed
addprocs(20)

@everywhere using LinearAlgebra
@everywhere sigma = [[1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0], [0.0 -1.0im; 1.0im 0.0], [1.0 0.0; 0.0 -1.0]]

@everywhere struct Parm
    m::Float64
    lamda::Float64
    #=
    t::Float64
    lamda_u::Float64
    lamda_d::Float64
    Pr::Float64
    =#
    hx::Float64
    hy::Float64
    hz::Float64
    mu::Float64
    eta::Float64
    T::Float64
    K_MAX::Float64
    K_SIZE::Int
    W_in::Float64
    W_MAX::Float64
    W_SIZE::Int
end


@everywhere mutable struct Hamiltonian
    Hk::Array{ComplexF64,2}
    Vx::Array{ComplexF64,2}
    Vy::Array{ComplexF64,2}
    Vxx::Array{ComplexF64,2}
    Vyx::Array{ComplexF64,2}
    Vyy::Array{ComplexF64,2}
    #後々非エルミートに拡張できるようにComplexF64にしているが、別にFloat64でも良いはず
    E::Array{ComplexF64,1}
end

#Photo-voltaicまでしか計算しないと思ってGRp,GAmまでしか用意してません。（SHGを計算したいならGRpp,GAmmを用意する必要あり）
@everywhere mutable struct Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    dGR::Array{ComplexF64,2}
    ddGR::Array{ComplexF64,2}
    GRmA::Array{ComplexF64,2}
    GRp::Array{ComplexF64,2}
    GAp::Array{ComplexF64,2}
    GRm::Array{ComplexF64,2}
    GAm::Array{ComplexF64,2}
end

@everywhere mutable struct Hamiltonian_3D
    Hk::Array{ComplexF64,2}
    Vx::Array{ComplexF64,2}
    Vy::Array{ComplexF64,2}
    Vz::Array{ComplexF64,2}
    Vxx::Array{ComplexF64,2}
    Vyx::Array{ComplexF64,2}
    Vyy::Array{ComplexF64,2}
    Vxz::Array{ComplexF64,2}
    Vyz::Array{ComplexF64,2}
    Vzz::Array{ComplexF64,2}
    #後々非エルミートに拡張できるようにComplexF64にしているが、別にFloat64でも良いはず
    E::Array{ComplexF64,1}
end

@everywhere function HandV_weyl(k::NTuple{3, Float64},p::Parm)

    #eps::Float64 = (k[1]*k[1] + k[2]*k[2] + k[3]*k[3])/(2.0*p.m) + p.mu
    #eps::Float64 = (k[1]*k[1])/(2.0*p.m) + p.mu
    eps::Float64 = p.m*k[1] + p.mu
    g_x::Float64 = p.lamda*k[1] + p.hx
    g_y::Float64 = p.lamda*k[2] + p.hy
    g_z::Float64 = p.lamda*k[3] + p.hz
    gg = [eps, g_x, g_y, g_z]
    H::Array{ComplexF64,2} =  gg' * sigma

    eps_vx::Float64 = p.m
    gx_vx::Float64 = p.lamda
    gy_vx::Float64 = 0.0
    gz_vx::Float64 = 0.0
    gg_x = [eps_vx, gx_vx, gy_vx, gz_vx]
    Vx::Array{ComplexF64,2} = gg_x' * sigma


    eps_vy::Float64 = 0.0
    gx_vy::Float64 = 0.0
    gy_vy::Float64 = p.lamda
    gz_vy::Float64 = 0.0
    gg_y = [eps_vy, gx_vy, gy_vy, gz_vy]
    Vy::Array{ComplexF64,2} = gg_y' * sigma

    eps_vz::Float64 = 0.0
    gx_vz::Float64 = 0.0
    gy_vz::Float64 = 0.0
    gz_vz::Float64 = p.lamda
    gg_z = [eps_vz, gx_vz, gy_vz, gz_vz]
    Vz::Array{ComplexF64,2} = gg_z' * sigma

    Vxx::Array{ComplexF64,2} = zeros(2,2)
    #sigma[1]./p.m
    
    Vyy::Array{ComplexF64,2} = zeros(2,2)
    #sigma[1]./p.m

    Vyx::Array{ComplexF64,2} = zeros(2,2)

    Vxz::Array{ComplexF64,2} = zeros(2,2)
    Vyz::Array{ComplexF64,2} = zeros(2,2)

    Vzz::Array{ComplexF64,2} = zeros(2,2)
    #[0.0 0.0; 0.0 0.0]

    E::Array{ComplexF64,1} = zeros(2)

    return H, Vx, Vy, Vz, Vxx, Vyx, Vyy, Vxz, Vyz, Vzz, E 
end


@everywhere function Gk(w::Float64, p::Parm, Ham::Hamiltonian_3D)
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    GRmA::Array{ComplexF64,2} = GR - GA
    dGR::Array{ComplexF64,2} = - GR * GR
    ddGR::Array{ComplexF64,2} = 2.0 * GR * GR * GR
    GRp0::Array{ComplexF64,2} = -Ham.Hk + (w+p.W_in)*Matrix{Complex{Float64}}(I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)
    GRp = inv(GRp0)
    GAp = GRp'

    GRm0::Array{ComplexF64,2} = -Ham.Hk + (w-p.W_in)*Matrix{Complex{Float64}}(I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)
    GRm = inv(GRm0)
    GAm = GRm'
    return GR, GA, dGR, ddGR, GRmA, GRp, GAp, GRm, GAm
end

@everywhere function HV_BI_3D(H::Hamiltonian_3D)

    H.E, BI::Array{ComplexF64,2} = eigen(H.Hk)
    Vx_BI::Array{ComplexF64,2} = BI' * H.Vx * BI
    Vy_BI::Array{ComplexF64,2} = BI' * H.Vy * BI
    Vz_BI::Array{ComplexF64,2} = BI' * H.Vz * BI
    Vxx_BI::Array{ComplexF64,2} = BI' * H.Vxx * BI
    Vyx_BI::Array{ComplexF64,2} = BI' * H.Vyx * BI
    Vyy_BI::Array{ComplexF64,2} = BI' * H.Vyy * BI
    Vxz_BI::Array{ComplexF64,2} = BI' * H.Vxz * BI
    Vyz_BI::Array{ComplexF64,2} = BI' * H.Vyz * BI
    Vzz_BI::Array{ComplexF64,2} = BI' * H.Vzz * BI

    H.Vx = Vx_BI
    H.Vy = Vy_BI
    H.Vz = Vz_BI
    H.Vxx = Vxx_BI
    H.Vyx = Vyx_BI
    H.Vyy = Vyy_BI
    H.Vxz = Vxz_BI
    H.Vyz = Vyz_BI
    H.Vzz = Vzz_BI
end

@everywhere f(e::Float64,T::Float64) = 1.0/(1.0+exp(e/T))
@everywhere df(e::Float64,T::Float64) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T

@everywhere f(e::ComplexF64,T::Float64) = 1.0/(1.0+exp(e/T))
@everywhere df(e::ComplexF64,T::Float64) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T

@everywhere function SC_PV_NLH(p::Parm, H::Hamiltonian_3D)
    ZXY_YX::Float64 = 0.0
    #XYZ_ZY::Float64 = 0.0
    HV_BI_3D(H)
    for i = 1:2
        ZXY_YX += imag((H.Vz[i,3-i]*H.Vx[3-i,i]*H.Vy[i,i] + H.Vz[i,3-i]*H.Vy[3-i,i]*H.Vx[i,i])/((H.E[i]-H.E[3-i])^2+p.eta^2))*real(df(H.E[i],p.T)) 
        #XYZ_ZY += imag((H.Vx[i,3-i]*H.Vy[3-i,i]*H.Vz[i,i] + H.Vx[i,3-i]*H.Vz[3-i,i]*H.Vy[i,i])/((H.E[i]-H.E[3-i])^2+p.eta^2))*real(df(H.E[i],p.T))
    end
    return ZXY_YX
    #, XYZ_ZY
end
#=
@everywhere function Green_PV_NLH(p::Parm, H::Hamiltonian_3D)
    ZXY_YX::Float64 = 0.0
    XYZ_ZY::Float64 = 0.0
    dw::Float64 = p.W_MAX/p.W_SIZE/pi
    for w in collect(-p.W_MAX:2.0p.W_MAX/p.W_SIZE:p.W_MAX)
        #range(-p.W_MAX, p.W_MAX, length=p.W_SIZE)
        G = Green(Gk(w,p,H)...)
        ZXY_YX += imag(tr(H.Vz*G.GR*H.Vyx*G.GRmA) + tr(H.Vz*G.GRmA*H.Vyx*G.GA) + tr(H.Vxz*G.GRp*H.Vy*G.GRmA) + tr(H.Vxz*G.GRmA*H.Vy*G.GAm) + tr(H.Vz*G.GR*H.Vx*G.GRp*H.Vy*G.GRmA) + tr(H.Vz*G.GRm*H.Vx*G.GRmA*H.Vy*G.GAm) + tr(H.Vz*G.GRmA*H.Vx*G.GAp*H.Vy*G.GA))*f(w,p.T)/(p.W_in^2)
        XYZ_ZY += imag(tr(H.Vx*G.GR*H.Vyz*G.GRmA) + tr(H.Vx*G.GRmA*H.Vyz*G.GA) + tr(H.Vyx*G.GRp*H.Vz*G.GRmA) + tr(H.Vyx*G.GRmA*H.Vz*G.GAm) + tr(H.Vx*G.GR*H.Vy*G.GRp*H.Vz*G.GRmA) + tr(H.Vx*G.GRm*H.Vy*G.GRmA*H.Vz*G.GAm) + tr(H.Vx*G.GRmA*H.Vy*G.GAp*H.Vz*G.GA))*f(w,p.T)/(p.W_in^2)
    end
    return dw*ZXY_YX, dw*XYZ_ZY
end
=#

@everywhere function Green_PV_NLH(p::Parm, H::Hamiltonian_3D)
    ZXY_YX::Float64 = 0.0
    #XYZ_ZY::Float64 = 0.0
    dw::Float64 = p.W_MAX/p.W_SIZE/pi
    for w in collect(-p.W_MAX:2.0p.W_MAX/p.W_SIZE:p.W_MAX)
        #range(-p.W_MAX, p.W_MAX, length=p.W_SIZE)
        G = Green(Gk(w,p,H)...)
        ZXY_YX += imag(tr(H.Vz*G.GR*H.Vyx*G.GRmA) + tr(H.Vz*G.GRmA*H.Vyx*G.GA) + tr(H.Vxz*G.GRp*H.Vy*G.GRmA) + tr(H.Vxz*G.GRmA*H.Vy*G.GAm) + tr(H.Vz*G.GR*H.Vx*G.GRp*H.Vy*G.GRmA) + tr(H.Vz*G.GRm*H.Vx*G.GRmA*H.Vy*G.GAm) + tr(H.Vz*G.GRmA*H.Vx*G.GAp*H.Vy*G.GA))*f(w,p.T)/(p.W_in^2)
        ZXY_YX += imag(tr(H.Vz*G.GR*H.Vyx*G.GRmA) + tr(H.Vz*G.GRmA*H.Vyx*G.GA) + tr(H.Vyz*G.GRm*H.Vx*G.GRmA) + tr(H.Vyz*G.GRmA*H.Vx*G.GAp) + tr(H.Vz*G.GR*H.Vy*G.GRm*H.Vx*G.GRmA) + tr(H.Vz*G.GRp*H.Vy*G.GRmA*H.Vx*G.GAp) + tr(H.Vz*G.GRmA*H.Vy*G.GAm*H.Vx*G.GA))*f(w,p.T)/(p.W_in^2)

        #XYZ_ZY += imag(tr(H.Vx*G.GR*H.Vyz*G.GRmA) + tr(H.Vx*G.GRmA*H.Vyz*G.GA) + tr(H.Vyx*G.GRp*H.Vz*G.GRmA) + tr(H.Vyx*G.GRmA*H.Vz*G.GAm) + tr(H.Vx*G.GR*H.Vy*G.GRp*H.Vz*G.GRmA) + tr(H.Vx*G.GRm*H.Vy*G.GRmA*H.Vz*G.GAm) + tr(H.Vx*G.GRmA*H.Vy*G.GAp*H.Vz*G.GA))*f(w,p.T)/(p.W_in^2)
        #XYZ_ZY += imag(tr(H.Vx*G.GR*H.Vyz*G.GRmA) + tr(H.Vx*G.GRmA*H.Vyz*G.GA) + tr(H.Vxz*G.GRm*H.Vy*G.GRmA) + tr(H.Vxz*G.GRmA*H.Vy*G.GAp) + tr(H.Vx*G.GR*H.Vz*G.GRm*H.Vy*G.GRmA) + tr(H.Vx*G.GRp*H.Vz*G.GRmA*H.Vy*G.GAp) + tr(H.Vx*G.GRmA*H.Vz*G.GAm*H.Vy*G.GA))*f(w,p.T)/(p.W_in^2)
    end
    return dw*ZXY_YX/(2*p.eta)
    #, dw*XYZ_ZY
end

@everywhere function Green_DC_NRC(p::Parm, H::Hamiltonian_3D)
    XXX::Float64 = 0.0
    #XYZ_ZY::Float64 = 0.0
    dw::Float64 = p.W_MAX/p.W_SIZE/pi
    for w in collect(-p.W_MAX:2.0p.W_MAX/p.W_SIZE:p.W_MAX)
        #range(-p.W_MAX, p.W_MAX, length=p.W_SIZE)
        G = Green(Gk(w,p,H)...)
        XXX += 2.0*imag(tr(H.Vx*G.dGR*H.Vxx*G.GRmA) + 2.0*tr(H.Vx*G.dGR*H.Vx*G.GR*H.Vx*G.GRmA))*df(w,p.T)/(p.W_in^2)

        #XYZ_ZY += imag(tr(H.Vx*G.GR*H.Vyz*G.GRmA) + tr(H.Vx*G.GRmA*H.Vyz*G.GA) + tr(H.Vyx*G.GRp*H.Vz*G.GRmA) + tr(H.Vyx*G.GRmA*H.Vz*G.GAm) + tr(H.Vx*G.GR*H.Vy*G.GRp*H.Vz*G.GRmA) + tr(H.Vx*G.GRm*H.Vy*G.GRmA*H.Vz*G.GAm) + tr(H.Vx*G.GRmA*H.Vy*G.GAp*H.Vz*G.GA))*f(w,p.T)/(p.W_in^2)
        #XYZ_ZY += imag(tr(H.Vx*G.GR*H.Vyz*G.GRmA) + tr(H.Vx*G.GRmA*H.Vyz*G.GA) + tr(H.Vxz*G.GRm*H.Vy*G.GRmA) + tr(H.Vxz*G.GRmA*H.Vy*G.GAp) + tr(H.Vx*G.GR*H.Vz*G.GRm*H.Vy*G.GRmA) + tr(H.Vx*G.GRp*H.Vz*G.GRmA*H.Vy*G.GAp) + tr(H.Vx*G.GRmA*H.Vz*G.GAm*H.Vy*G.GA))*f(w,p.T)/(p.W_in^2)
    end
    return dw*XXX
    #, dw*XYZ_ZY
end

@everywhere function Green_ZXY_BI(p::Parm, H::Hamiltonian_3D)
    
    Drude::Float64 = 0.0
    BCD::Float64 = 0.0
    sQMD::Float64 = 0.0
    dQMD::Float64 = 0.0
    Inter::Float64 = 0.0
    dInter::Float64 = 0.0

    HV_BI_3D(H)

    for i = 1:2
        Drude += 2.0*real(2.0*H.Vz[i,i]*(H.Vx[i,i]*H.Vy[i,i]*imag(df(H.E[i]+1.0im*p.eta, p.T))/(2.0p.eta) + H.Vx[i,3-i]*H.Vy[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta)*real(df(H.E[i]+1.0im*p.eta, p.T)) + H.Vyx[i,i]*real(df(H.E[i]+1.0im*p.eta, p.T))))/((2.0p.eta)^2)

        BCD += -2.0*imag(H.Vz[i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))real(H.Vy[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        sQMD += -2.0*imag(2.0*H.Vz[i,i]*H.Vx[i,3-i]*H.Vy[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta))*imag(df(H.E[i]+1.0im*p.eta, p.T))/((2.0p.eta)^2)
        dQMD += -2.0*real(H.Vz[i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))imag(H.Vy[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        Inter += -2.0*real(H.Vz[i,3-i]*H.Vx[3-i,3-i]*H.Vy[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3))*real(df(H.E[i]+1.0im*p.eta, p.T))
        dInter += 2.0*imag(H.Vz[i,3-i]*H.Vx[3-i,3-i]*H.Vy[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3))*imag(df(H.E[i]+1.0im*p.eta, p.T))
    end

    return Drude, BCD, sQMD, dQMD, Inter, dInter
end

@everywhere function Green_XXX_BI(p::Parm, H::Hamiltonian_3D)
    
    Drude::Float64 = 0.0
    BCD::Float64 = 0.0
    sQMD::Float64 = 0.0
    dQMD::Float64 = 0.0
    Inter::Float64 = 0.0
    dInter::Float64 = 0.0

    HV_BI_3D(H)

    for i = 1:2
        Drude += 2.0*real(2.0*H.Vz[i,i]*(H.Vx[i,i]*H.Vx[i,i]*imag(df(H.E[i]+1.0im*p.eta, p.T))/(2.0p.eta) + H.Vx[i,3-i]*H.Vx[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta)*real(df(H.E[i]+1.0im*p.eta, p.T)) + H.Vxx[i,i]*real(df(H.E[i]+1.0im*p.eta, p.T))))/((2.0p.eta)^2)

        BCD += -2.0*imag(H.Vx[i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))real(H.Vx[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        sQMD += -2.0*imag(2.0*H.Vx[i,i]*H.Vx[i,3-i]*H.Vx[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta))*imag(df(H.E[i]+1.0im*p.eta, p.T))/((2.0p.eta)^2)
        dQMD += -2.0*real(H.Vx[i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))imag(H.Vx[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        Inter += -2.0*real(H.Vx[i,3-i]*H.Vx[3-i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3))*real(df(H.E[i]+1.0im*p.eta, p.T))
        dInter += 2.0*imag(H.Vx[i,3-i]*H.Vx[3-i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3))*imag(df(H.E[i]+1.0im*p.eta, p.T))
    endd

    return Drude, BCD, sQMD, dQMD, Inter, dInter
end

using DataFrames
using CSV
using Plots
#gr()

function main(arg::Array{String,1})
    #println(arg)
    println("m, lamda, hx, hy, hz, mu, eta, T, K_MAX, K_SIZE, W_in, W_MAX, W_SIZE")
    T0 = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
    #collect(0.001:0.005:0.04)
    XXXG_mu = zeros(Float64,length(T0))
    XXXSC_mu = zeros(Float64,length(T0))
    #A::String = []
    for j in 1:length(T0)
        #Parm: m, lamda, hx, hy, hz, mu, eta, T, K_MAX, K_SIZE, W_in, W_MAX, W_SIZE
        p = Parm(parse(Float64,arg[1]), parse(Float64,arg[2]), 0.0, 0.0, 0.0, parse(Float64,arg[3]), parse(Float64,arg[4]), T0[j], 1.0, parse(Int,arg[5]), parse(Float64,arg[6]), parse(Float64,arg[7]), parse(Int,arg[8]))
        if j == 1
            println(p)
        end

        k2 = collect(Iterators.product((-p.K_MAX:2*p.K_MAX/p.K_SIZE:p.K_MAX)[1:end-1], (-p.K_MAX:2*p.K_MAX/p.K_SIZE:p.K_MAX)[1:end-1]))
        for kz in collect(-p.K_MAX:2*p.K_MAX/p.K_SIZE:p.K_MAX)[1:end-1]
            XXXG::Float64 = 0.0
            XXXSC::Float64 = 0.0
            XXXG, XXXSC = @distributed (+) for i in 1:length(k2)
                k = (k2[i][1], k2[i][2], kz)
                Hamk = Hamiltonian_3D(HandV_weyl(k,p)...)
                XXXG0 = Green_PV_NLH(p,Hamk)
                XXXSC0 = SC_PV_NLH(p, Hamk)
                [XXXG0/(p.K_SIZE^3), XXXSC0/(p.K_SIZE^3)]
            end
            XXXG_mu[j] += XXXG
            XXXSC_mu[j] += XXXSC
        end
        #push!(A,"#")
        #println(A)
    end
    println("finish the calculation!")
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data = DataFrame(T=T0,XXX_Green=XXXG_mu,XXX_SC=XXXSC_mu)
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./T_dep_NRC.csv", save_data)

    #gr()
    ENV["GKSwstype"]="nul"
    p = plot(T0, XXXG_mu, label="Green",xlabel="T",ylabel="σ",title="nonreciprocal conductivity", width=2.0, marker=:circle)
    p = plot!(T0, XXXSC_mu, label="SC", width=2.0, marker=:circle)
    savefig(p,"./T_dep_NLH.png")
end

main(ARGS)