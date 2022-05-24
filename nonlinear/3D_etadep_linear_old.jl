#Parm: m, lamda, hx, hy, hz, mu, eta, T, K_MAX, K_SIZE, W_in, W_MAX, W_SIZE
#p = Parm(ARGS[1], ARGS[2], 0.0, 0.0, 0.0, ARGS[3], T0, ARGS[4], 1.0, ARGS[5], ARGS[6] , ARGS[7], ARGS[8])
# ARGS =(m(tilting), lamda, eta, mu, K_SIZE, W_in, W_MAX, W_SIZE)

#import Pkg
#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("Plots")

using Distributed
addprocs(32)

@everywhere using LinearAlgebra
@everywhere sigma = [[1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0], [0.0 -1.0im; 1.0im 0.0], [1.0 0.0; 0.0 -1.0]]

@everywhere struct Parm
    t::Float64
    p0::Float64
    v::Float64
    mu::Float64
    Delta::Float64
    eta::Float64
    T::Float64
    K_SIZE::Int
    W_MAX::Float64
    W_SIZE::Int
end


@everywhere f(e::Float64,T::Float64) = 1.0/(1.0+exp(e/T))
@everywhere df(e::Float64,T::Float64) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T

@everywhere f(e::ComplexF64,T::Float64) = 1.0/(1.0+exp(e/T))
@everywhere df(e::ComplexF64,T::Float64) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T

@everywhere mutable struct Hamiltonian
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
    E::Array{Float64,1}
end

@everywhere mutable struct Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    GRmA::Array{ComplexF64,2}
    dGR::Array{ComplexF64,2}
end

@everywhere function Gk(w::Float64, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    GRmA::Array{ComplexF64,2} = GR - GA
    dGR::Array{ComplexF64,2} = - GR * GR
    
    return GR, GA, GRmA, dGR
end
@everywhere function Gk_M(M::Int, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + pi*p.T*(2*M+1)*Matrix{Complex{Float64}}(I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA0::Array{ComplexF64,2} = -Ham.Hk + 1.0im*pi*p.T*(2*M+1)*Matrix{Complex{Float64}}(I,2,2) - p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)    
    GA::Array{ComplexF64,2} = inv(GA0)
    GRmA::Array{ComplexF64,2} = GR - GA
    dGR::Array{ComplexF64,2} = - GR * GR
    
    return GR, GA, GRmA, dGR
end
@everywhere function HandV(k::NTuple{3, Float64},p::Parm)

    eps::Float64 = p.mu
    g_x::Float64 = p.Delta
    g_y::Float64 = p.v*sin(k[3])
    g_z::Float64 = p.t*(2.0 + cos(p.p0) -cos(k[1]) - cos(k[2]) -cos(k[3]))
    gg = [eps, g_x, g_y, g_z]
    H::Array{ComplexF64,2} =  gg' * sigma

    eps_vx::Float64 = 0.0
    gx_vx::Float64 = 0.0
    gy_vx::Float64 = 0.0
    gz_vx::Float64 = p.t*sin(k[1])
    gg_x = [eps_vx, gx_vx, gy_vx, gz_vx]
    Vx::Array{ComplexF64,2} = gg_x' * sigma


    eps_vy::Float64 = 0.0
    gx_vy::Float64 = 0.0
    gy_vy::Float64 = 0.0
    gz_vy::Float64 = p.t*sin(k[2])
    gg_y = [eps_vy, gx_vy, gy_vy, gz_vy]
    Vy::Array{ComplexF64,2} = gg_y' * sigma

    eps_vz::Float64 = 0.0
    gx_vz::Float64 = 0.0
    gy_vz::Float64 = p.v*cos(k[3])
    gz_vz::Float64 = p.t*sin(k[3])
    gg_z = [eps_vz, gx_vz, gy_vz, gz_vz]
    Vz::Array{ComplexF64,2} = gg_z' * sigma

    Vxx::Array{ComplexF64,2} = sigma[4].*p.t*cos(k[1])
    
    Vyy::Array{ComplexF64,2} = sigma[3].*p.t*cos(k[2])

    Vyx::Array{ComplexF64,2} = [0.0 0.0
    0.0 0.0]

    Vxz::Array{ComplexF64,2} = [0.0 0.0
    0.0 0.0]
    
    Vyz::Array{ComplexF64,2} = [0.0 0.0
    0.0 0.0]

    Vzz::Array{ComplexF64,2} = sigma[4].*p.t*cos(k[3]) - sigma[3].*p.v*sin(k[3])

    E::Array{ComplexF64,1} = zeros(2)

    return H, Vx, Vy, Vz, Vxx, Vyx, Vyy, Vxz, Vyz, Vzz, E 
end

@everywhere function HV_BI(H::Hamiltonian)

    H.E, BI::Array{ComplexF64,2} = eigen(H.Hk)
    H.Hk = [H.E[1] 0.0; 0.0 H.E[2]]
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


@everywhere function Green_ZZ_BI(p::Parm, H::Hamiltonian)
    
    Drude::Float64 = 0.0
    Drude0::Float64 = 0.0
    BC::Float64 = 0.0
    dQM::Float64 = 0.0
    app_QM::Float64 = 0.0
    
    HV_BI(H)

    for i = 1:2
        Drude += -real(H.Vz[i,i]*H.Vz[i,i])*real(df(H.E[i]+1.0im*p.eta, p.T))/(2.0p.eta)
        Drude0 += -real(H.Vz[i,i]*H.Vz[i,i])*real(df(H.E[i], p.T))/(2.0p.eta)
        BC += 2.0*imag(H.Vz[i,3-i]*H.Vz[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta)/(H.E[i]-H.E[3-i]+2.0im*p.eta))*real(f(H.E[i]+1.0im*p.eta, p.T))
        #BC += 2.0p.eta*real(H.Vz[i,3-i]*H.Vz[3-i,i]/((H.E[i]-H.E[3-i])^2+4.0*p.eta^2))*real(-df(H.E[i]+1.0im*p.eta, p.T))
        dQM += 2.0*real(H.Vz[i,3-i]*H.Vz[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))*imag(f(H.E[i]+1.0im*p.eta, p.T))
        #dQM += (H.E[i]-H.E[3-i])*real(H.Vz[i,3-i]*H.Vz[3-i,i]/((H.E[i]-H.E[3-i])^2+4.0*p.eta^2))*imag(df(H.E[i]+1.0im*p.eta, p.T))
        app_QM += p.eta*real(H.Vz[i,3-i]*H.Vz[3-i,i])/((H.E[i]-H.E[3-i])^2+4.0*p.eta^2)*(-df(H.E[i], p.T))
    end
    return Drude, Drude0, BC, dQM, app_QM
end
@everywhere function Green_DC_2D(p::Parm, H::Hamiltonian)
    ZZ::Float64 = 0.0
    #dw::Float64 = p.W_MAX/p.W_SIZE/pi
    mi = minimum([p.W_MAX,12p.T])
    dw::Float64 = mi/p.W_SIZE/pi
    for w in collect(-mi:2.0mi/p.W_SIZE:mi)
        #range(-p.W_MAX, p.W_MAX, length=p.W_SIZE)
        G = Green(Gk(w,p,H)...)
        ZZ += real(tr(H.Vz*G.GR*H.Vz*G.GRmA))*df(w,p.T)
        #ZZ += -2.0real(tr(H.Vz*G.dGR*H.Vz*G.GR))*f(w,p.T)
    end
    return dw*ZZ
end


@everywhere function Green_DC_BI(p::Parm, H::Hamiltonian)
    DrG::Float64 = 0.0
    G_tot::Float64 = 0.0
    #dw::Float64 = p.W_MAX/p.W_SIZE/pi
    mi = minimum([p.W_MAX,12p.T])
    dw::Float64 = mi/p.W_SIZE/pi
    for w in collect(-mi:2.0mi/p.W_SIZE:mi)
        #range(-p.W_MAX, p.W_MAX, length=p.W_SIZE)
        G = Green(Gk(w,p,H)...)
        for i in 1:2
            DrG += real(H.Vz[i,i]*G.GR[i,i]*H.Vz[i,i]*G.GRmA[i,i])*df(w,p.T)
        end
        G_tot += real(tr(H.Vz*G.GR*H.Vz*G.GRmA))*df(w,p.T)
    end
    return dw*DrG, dw*G_tot
end


using DataFrames
using CSV
using Plots
#gr()

function main(arg::Array{String,1})
    #println(arg)
    println("t, p0, v, mu, Delta, eta, T, K_SIZE, W_MAX, W_SIZE")
    mu0 = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.14, 0.16]
    #[0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04]
    #[0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.1]
    #collect(-0.2:0.02:0.2)
    #collect(-0.2:0.01:0.01)
    DrudeX_mu = zeros(Float64,length(mu0))
    Drude0_mu = zeros(Float64,length(mu0))
    BCX_mu = zeros(Float64,length(mu0))
    dQMX_mu = zeros(Float64,length(mu0))
    app_QM_mu = zeros(Float64,length(mu0))
    Green_mu = zeros(Float64,length(mu0))

    for j in 1:length(mu0)
        #Parm: t, p0, v, mu, Delta, eta, T, K_SIZE, W_MAX, W_SIZE
        p = Parm(parse(Float64,arg[1]), pi/2, parse(Float64,arg[2]), parse(Float64,arg[3]), parse(Float64,arg[4]), mu0[j], parse(Float64,arg[5]), parse(Int,arg[6]), parse(Float64,arg[7]), parse(Int,arg[8]))

        k2 = collect(Iterators.product((-pi:2pi/p.K_SIZE:pi)[1:end-1], (-pi:2pi/p.K_SIZE:pi)[1:end-1]))
        for kz in collect(-pi:2pi/p.K_SIZE:pi)[1:end-1]
            Dr0::Float64 = 0.0
            Dr00::Float64 = 0.0
            BC0::Float64 = 0.0
            dQM0::Float64 = 0.0
            app_QM0::Float64 = 0.0
            Green0::Float64 = 0.0
            Dr0, Dr00, BC0, dQM0, app_QM0, Green0 = @distributed (+) for i in 1:length(k2)
                k = (k2[i][1], k2[i][2], kz)
                Hamk = Hamiltonian(HandV(k,p)...)
                Dr, Dr_0, BC, dQM, app_QM = Green_ZZ_BI(p,Hamk)
                Green_Dr, Green_tot = Green_DC_BI(p, Hamk)
                [Dr/(p.K_SIZE^3), Green_Dr/(p.K_SIZE^3), BC/(p.K_SIZE^3), dQM/(p.K_SIZE^3), app_QM/(p.K_SIZE^3), Green_tot/(p.K_SIZE^3)]
            end
            DrudeX_mu[j] += Dr0
            Drude0_mu[j] += Dr00
            BCX_mu[j] += BC0
            dQMX_mu[j] += dQM0
            app_QM_mu[j] += app_QM0
            Green_mu[j] += Green0
        end
        if j == 1
            println(p)
        end
        println("eta, Drude, QM_re, QM_im, Green_sum, TF_sum")
        println(mu0[j], ", ", DrudeX_mu[j], ", ", BCX_mu[j], ", ", dQMX_mu[j], ", ", Green_mu[j], ", ", Green_mu[j]-DrudeX_mu[j]-BCX_mu[j]-dQMX_mu[j])
    end
    println("finish the calculation!")
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data = DataFrame(eta=mu0, Drude=DrudeX_mu, BCD=BCX_mu, dQM=dQMX_mu, Green_Dr=Drude0_mu, app_QM=app_QM_mu, Green_tot=Green_mu)
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./eta_dep_ZZ.csv", save_data)

    #gr()
    ENV["GKSwstype"]="nul"
    p1=plot(mu0, DrudeX_mu, label="Drude",xlabel="μ",ylabel="σ",title="linear conductivity", width=2.0, marker=:circle)
    p1=plot!(mu0, BCX_mu, label="sQM_re", width=2.0, marker=:circle)
    p1=plot!(mu0, dQMX_mu, label="sQM_im", width=2.0, marker=:circle)
    p1=plot!(mu0, Green_mu-DrudeX_mu-BCX_mu-dQMX_mu, label="TF", width=2.0, marker=:circle)
    p1=plot!(mu0, Green_mu, label="Green", width=2.0, marker=:circle)
    savefig(p1,"./eta_dep_ZZ.png")
end

@time main(ARGS)
