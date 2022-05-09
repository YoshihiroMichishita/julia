#Parm: m, lamda, hx, hy, hz, mu, eta, T, K_MAX, K_SIZE, W_in, W_MAX, W_SIZE
#p = Parm(ARGS[1], ARGS[2], 0.0, 0.0, 0.0, mu0[j], ARGS[3], ARGS[4], 1.0, ARGS[5], ARGS[6] , ARGS[7], ARGS[8])
# ARGS =(m(tilting), lamda, eta, T, K_SIZE, W_in, W_MAX, W_SIZE)

#import Pkg
#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("Plots")

using Distributed
addprocs(30)

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

#Photo-voltaicまでしか計算しないと思ってGRp,GAmまでしか用意してません。（SHGを計算したいならGRpp,GAmmを用意する必要あり）
@everywhere mutable struct Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    dGR::Array{ComplexF64,2}
    dGA::Array{ComplexF64,2}
    ddGR::Array{ComplexF64,2}
    GRmA::Array{ComplexF64,2}
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
    dGA::Array{ComplexF64,2} = - GA * GA
    ddGR::Array{ComplexF64,2} = 2.0 * GR * GR * GR
    
    return GR, GA, dGR, dGA, ddGR, GRmA
end

@everywhere function G_M(m::Int, p::Parm, Ham::Hamiltonian_3D)
    #Green関数のinverse
    wm = pi*(2m+1)*p.T
    GR0::Array{ComplexF64,2} = -Ham.Hk + wm*Matrix{Complex{Float64}}(1.0im*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA0::Array{ComplexF64,2} = -Ham.Hk + wm*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)
    GA::Array{ComplexF64,2} = inv(GA0)
    GRmA::Array{ComplexF64,2} = GR - GA
    dGR::Array{ComplexF64,2} = - GR * GR
    dGA::Array{ComplexF64,2} = - GA * GA
    ddGR::Array{ComplexF64,2} = 2.0* GR * GR * GR
    
    return GR, GA, GRmA, dGR, dGA, ddGR
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

    H.Hk = [H.E[1] 0; 0 H.E[2]]
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

@everywhere function SC_BI_NLH(p::Parm, H::Hamiltonian_3D)
    BCD::Float64 = 0.0
    Drude::Float64 = 0.0
    ChS::Float64 = 0.0
    gBC::Float64 = 0.0
    HV_BI_3D(H)
    for i = 1:2
        BCD += imag(H.Vz[i,3-i]*H.Vx[3-i,i]*H.Vy[i,i] + H.Vz[i,3-i]*H.Vy[3-i,i]*H.Vx[i,i])*real(1.0/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)*df(H.E[i]+1.0im*p.eta,p.T))/(2.0p.eta)
        Drude+= real(H.Vz[i,i]*(2.0*H.Vx[i,i]*H.Vy[i,i]/(2.0im*p.eta) + (H.Vx[i,3-i]*H.Vy[3-i,i]+H.Vy[i,3-i]*H.Vx[3-i,i])*real(1.0/(H.E[i]-H.E[3-i]+2.0im*p.eta)))/(-4.0*p.eta^2)*df(H.E[i]+1.0im*p.eta,p.T)) 
        ChS += real(H.Vz[i,3-i]*H.Vx[3-i,i]*H.Vy[i,i] + H.Vz[i,3-i]*H.Vy[3-i,i]*H.Vx[i,i])*imag(1.0/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)*df(H.E[i]+1.0im*p.eta,p.T))/(2.0p.eta)
        ChS += real(H.Vz[i,i]*(H.Vx[i,3-i]*H.Vy[3-i,i]+H.Vy[i,3-i]*H.Vx[3-i,i])*2.0im/((H.E[i]-H.E[3-i])^2+4.0*p.eta^2)/(-4.0*p.eta)*df(H.E[i]+1.0im*p.eta,p.T)) 
        ChS += real(H.Vz[i,3-i]*(H.Vx[3-i,3-i]*H.Vy[3-i,i]+H.Vy[3-i,3-i]*H.Vx[3-i,i]))*real(1.0/(H.E[i]-H.E[3-i]+2.0im*p.eta)^3*df(H.E[i]+1.0im*p.eta,p.T))
        gBC += -imag(H.Vz[i,3-i]*(H.Vx[3-i,3-i]*H.Vy[3-i,i]+H.Vy[3-i,3-i]*H.Vx[3-i,i]))*imag(1.0/(H.E[i]-H.E[3-i]+2.0im*p.eta)^3*df(H.E[i]+1.0im*p.eta,p.T))
    end
    return Drude, BCD, ChS, gBC
end

@everywhere function Green_NLH(p::Parm, H::Hamiltonian_3D)
    gNLH::Float64 = 0.0
    #HV_BI_3D(H)
    dw = 2.0p.W_MAX/p.W_SIZE
    for w = 1:p.W_SIZE
        w0 = dw*(w-p.W_SIZE/2)
        G = Green(Gk(w0,p,H)...)
        gNLH += 2.0*dw*imag(tr(H.Vz*G.dGR*((H.Vx*G.GR*H.Vy)+(H.Vy*G.GR*H.Vx))*G.GRmA))*df(w0,p.T)
    end
    return gNLH
end

@everywhere function Green_ZXY_M(p::Parm, H::Hamiltonian_3D)
    G0::Float64 = 0.0
    #mi = minimum([p.W_MAX,12p.T])
    #dw::Float64 = mi/p.W_SIZE/pi
    for w in 0:p.W_SIZE
        G = Green(G_M(w,p,H)...)
        G0 += 2.0imag(1.0im*tr(H.Vz*G.ddGR*(H.Vx*G.GR*H.Vy + H.Vy*G.GR*H.Vx)*G.GRmA))
        G0 += 2.0imag(1.0im*tr(H.Vz*G.dGR*(H.Vx*G.dGR*H.Vy+H.Vy*G.dGR*H.Vx)*G.GRmA))
        G0 += 2.0imag(1.0im*tr(H.Vz*G.dGR*(H.Vx*G.GR*H.Vy + H.Vy*G.GR*H.Vx)*(G.dGR-G.dGA)))
    end
    return p.T*G0
end

using DataFrames
using CSV
using Plots
#gr()

function main(arg::Array{String,1})
    println(arg)
    mu0 = collect(-0.2:0.01:0.2)
    Drude_mu = zeros(Float64,length(mu0))
    BCD_mu = zeros(Float64,length(mu0))
    ChS_mu = zeros(Float64,length(mu0))
    gBC_mu = zeros(Float64,length(mu0))
    gr_mu = zeros(Float64,length(mu0))

    for j in 1:length(mu0)
        #Parm: m, lamda, hx, hy, hz, mu, eta, T, K_MAX, K_SIZE, W_in, W_MAX, W_SIZE
        p = Parm(parse(Float64,arg[1]), parse(Float64,arg[2]), 0.0, 0.0, 0.0, mu0[j], parse(Float64,arg[3]), parse(Float64,arg[4]), 1.0, parse(Int,arg[5]), 0.0, parse(Float64,arg[6]), parse(Float64,arg[7]))

        k2 = collect(Iterators.product((-p.K_MAX:2*p.K_MAX/p.K_SIZE:p.K_MAX)[1:end-1], (-p.K_MAX:2*p.K_MAX/p.K_SIZE:p.K_MAX)[1:end-1]))
        for kz in collect(-p.K_MAX:2*p.K_MAX/p.K_SIZE:p.K_MAX)[1:end-1]
            Drude::Float64 = 0.0
            BCD::Float64 = 0.0
            ChS::Float64 = 0.0
            gBC::Float64 = 0.0
            #gr::Float64 = 0.0
            Drude, BCD, ChS, gBC, gr = @distributed (+) for i in 1:length(k2)
                k = (k2[i][1], k2[i][2], kz)
                Hamk = Hamiltonian_3D(HandV_weyl(k,p)...)
                gr0 = 0.0
                #Green_ZXY_M(p,Hamk)
                #HV_BI_3D(Hamk)
                Dr0, BC0, Ch0, gB0 = SC_BI_NLH(p, Hamk)
                [Dr0/(p.K_SIZE^3), BC0/(p.K_SIZE^3), Ch0/(p.K_SIZE^3), gB0/(p.K_SIZE^3), gr0/(p.K_SIZE^3)]
            end
            Drude_mu[j] += Drude
            BCD_mu[j] += BCD
            ChS_mu[j] += ChS
            gBC_mu[j] += gBC
            gr_mu[j] += gr/(2pi)
        end
    end

    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data = DataFrame(mu=mu0,Drude=Drude_mu,BCD=BCD_mu, ChS=ChS_mu, gBC=gBC_mu, Green=gr_mu)
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./mu_dep_NLH_singleWeyl.csv", save_data)

    ENV["GKSwstype"]="nul"
    plot(mu0, Drude_mu, label="Drude",xlabel="μ",ylabel="σ",title="nonlinear conductivity", width=2.0, marker=:circle)
    plot!(mu0, BCD_mu, label="BCD", width=2.0, marker=:circle)
    plot!(mu0, ChS_mu, label="ChS", width=2.0, marker=:circle)
    plot!(mu0, gBC_mu, label="gBC", width=2.0, marker=:circle)
    plot!(mu0, gr_mu, label="Green", width=2.0, marker=:circle)
    plot!(mu0, 2.0*(BCD_mu+gBC_mu)+gr_mu, label="sum", width=2.0, marker=:circle)
    savefig("./mu_dep_NLH_singleWeyl.png")
    
    #=
    for j in 1:length(mu0)
        #Parm: m, lamda, hx, hy, hz, mu, eta, T, K_MAX, K_SIZE, W_in, W_MAX, W_SIZE
        p = Parm(-parse(Float64,arg[1]), parse(Float64,arg[2]), 0.0, 0.0, 0.0, mu0[j], parse(Float64,arg[3]), parse(Float64,arg[4]), 1.0, parse(Int,arg[5]), 0.0, parse(Float64,arg[6]), parse(Float64,arg[7]))

        k2 = collect(Iterators.product((-p.K_MAX:2*p.K_MAX/p.K_SIZE:p.K_MAX)[1:end-1], (-p.K_MAX:2*p.K_MAX/p.K_SIZE:p.K_MAX)[1:end-1]))
        for kz in collect(-p.K_MAX:2*p.K_MAX/p.K_SIZE:p.K_MAX)[1:end-1]
            Drude::Float64 = 0.0
            BCD::Float64 = 0.0
            ChS::Float64 = 0.0
            gBC::Float64 = 0.0
            gr::Float64 = 0.0
            Drude, BCD, ChS, gBC, gr = @distributed (+) for i in 1:length(k2)
                k = (k2[i][1], k2[i][2], kz)
                Hamk = Hamiltonian_3D(HandV_weyl(k,p)...)
                gr0 = 0.0
                #Green_NLH(p,Hamk)
                #HV_BI_3D(Hamk)
                Dr0, BC0, Ch0, gB0 = SC_BI_NLH(p, Hamk)
                [Dr0/(p.K_SIZE^3), BC0/(p.K_SIZE^3), Ch0/(p.K_SIZE^3), gB0/(p.K_SIZE^3), gr0/(p.K_SIZE^3)]
            end
            Drude_mu[j] += Drude
            BCD_mu[j] += BCD
            ChS_mu[j] += ChS
            gBC_mu[j] += gBC
            gr_mu[j] += gr/(2pi)
        end
    end

    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data = DataFrame(mu=mu0,Drude=Drude_mu,BCD=BCD_mu, ChS=ChS_mu, gBC=gBC_mu, Green=gr_mu)
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./mu_dep_NLH.csv", save_data)

    ENV["GKSwstype"]="nul"
    plot(mu0, Drude_mu, label="Drude",xlabel="μ",ylabel="σ",title="nonlinear conductivity", width=2.0, marker=:circle)
    plot!(mu0, BCD_mu, label="BCD", width=2.0, marker=:circle)
    plot!(mu0, ChS_mu, label="ChS", width=2.0, marker=:circle)
    plot!(mu0, gBC_mu, label="gBC", width=2.0, marker=:circle)
    plot!(mu0, gr_mu, label="Green", width=2.0, marker=:circle)
    savefig("./mu_dep_NLH.png")=#
end

@time main(ARGS)

