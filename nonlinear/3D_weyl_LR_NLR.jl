
using Distributed
addprocs(26)

@everywhere using LinearAlgebra
@everywhere sigma = [[1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0], [0.0 -1.0im; 1.0im 0.0], [1.0 0.0; 0.0 -1.0]]

#Parm(t1, t2, delta, mu, kw, eta, T, K_SIZE, W_MAX, W_SIZE)
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

    e0 = p.t2*cos(k[1]+k[2]) + p.delta*cos(k[1]-k[2]) + p.mu
    ex = p.t1*(cos(p.kw)-cos(k[2])) + p.delta*(1.0-cos(k[3]))
    ey = p.t1*sin(k[3])
    ez = p.t1*(cos(p.kw)-cos(k[1])) + p.delta*(1.0-cos(k[3]))
    ee = [e0, ex, ey, ez]
    H::Array{ComplexF64,2} = ee' * sigma
    Hermitian(H)

    eps_vx::Float64 = -p.t2*sin(k[1]+k[2]) - p.delta*sin(k[1]-k[2])
    gx_vx::Float64 = 0.0
    gy_vx::Float64 = 0.0
    gz_vx::Float64 = p.t1*sin(k[1])
    gg_x = [eps_vx, gx_vx, gy_vx, gz_vx]
    Vx::Array{ComplexF64,2} = gg_x' * sigma


    eps_vy::Float64 = -p.t2*sin(k[1]+k[2]) + p.delta*sin(k[1]-k[2])
    gx_vy::Float64 = p.t1*sin(k[2])
    gy_vy::Float64 = 0.0
    gz_vy::Float64 = 0.0
    gg_y = [eps_vy, gx_vy, gy_vy, gz_vy]
    Vy::Array{ComplexF64,2} = gg_y' * sigma

    eps_vz::Float64 = 0.0
    gx_vz::Float64 = p.delta*sin(k[3])
    gy_vz::Float64 = p.t1*cos(k[3])
    gz_vz::Float64 = p.delta*sin(k[3])
    gg_z = [eps_vz, gx_vz, gy_vz, gz_vz]
    Vz::Array{ComplexF64,2} = gg_z' * sigma

    Vxx::Array{ComplexF64,2} = (-p.t2*cos(k[1]+k[2]) - p.delta*cos(k[1]-k[2]))*sigma[1] + (p.t1*cos(k[1]))*sigma[4]
    
    Vyy::Array{ComplexF64,2} = (-p.t2*cos(k[1]+k[2]) - p.delta*cos(k[1]-k[2]))*sigma[1] + (p.t1*cos(k[2]))*sigma[2]

    Vyx::Array{ComplexF64,2} = (-p.t2*cos(k[1]+k[2]) + p.delta*cos(k[1]-k[2]))*sigma[1]

    Vxz::Array{ComplexF64,2} = [0.0 0.0; 0.0 0.0]
    
    Vyz::Array{ComplexF64,2} = [0.0 0.0; 0.0 0.0]

    Vzz::Array{ComplexF64,2} = p.delta*cos(k[3])*(sigma[2]+sigma[4]) - p.t1*sin(k[3])*sigma[3]

    E::Array{ComplexF64,1} = [0.0, 0.0]

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
        BC += 2.0*imag(H.Vz[i,3-i]*H.Vz[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))*real(f(H.E[i]+1.0im*p.eta, p.T))
        dQM += 2.0*real(H.Vz[i,3-i]*H.Vz[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))*imag(f(H.E[i]+1.0im*p.eta, p.T))
        app_QM += 2.0*p.eta*real(H.Vz[i,3-i]*H.Vz[3-i,i])/((H.E[i]-H.E[3-i])^2+4.0*p.eta^2)*(-df(H.E[i], p.T))
    end
    return Drude, Drude0, BC, dQM, app_QM
end
@everywhere function Green_DC_2D(p::Parm, H::Hamiltonian)
    ZZ::Float64 = 0.0
    dw::Float64 = p.W_MAX/p.W_SIZE/pi
    mi = minimum([p.W_MAX,12p.T])
    for w in collect(-mi:2.0p.W_MAX/p.W_SIZE:mi)
        #range(-p.W_MAX, p.W_MAX, length=p.W_SIZE)
        G = Green(Gk(w,p,H)...)
        ZZ += -real(tr(H.Vz*G.GR*H.Vz*G.GA))*df(w,p.T)
        #ZZ += -2.0real(tr(H.Vz*G.dGR*H.Vz*G.GR))*f(w,p.T)
    end
    return dw*ZZ
end

@everywhere function Green_XXX_BI(p::Parm, H::Hamiltonian)
    
    Drude::Float64 = 0.0
    BCD::Float64 = 0.0
    sQMD::Float64 = 0.0
    dQMD::Float64 = 0.0
    Inter::Float64 = 0.0
    dInter::Float64 = 0.0

    #HV_BI(H)

    for i = 1:2
        Drude += 2.0*real(2.0*H.Vx[i,i]*(H.Vx[i,i]*H.Vx[i,i]*imag(df(H.E[i]+1.0im*p.eta, p.T))/(2.0p.eta) + H.Vx[i,3-i]*H.Vx[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta)*real(df(H.E[i]+1.0im*p.eta, p.T)) + H.Vxx[i,i]*real(df(H.E[i]+1.0im*p.eta, p.T))))/((2.0p.eta)^2)

        BCD += -2.0*imag(H.Vx[i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))real(H.Vx[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        sQMD += -2.0*imag(2.0*H.Vx[i,i]*H.Vx[i,3-i]*H.Vx[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta))*imag(df(H.E[i]+1.0im*p.eta, p.T))/((2.0p.eta)^2)
        dQMD += -2.0*real(H.Vx[i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))imag(H.Vx[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        Inter += -real(H.Vx[i,3-i]*(2.0*H.Vx[3-i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3) + H.Vxx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)))*real(df(H.E[i]+1.0im*p.eta, p.T))
        dInter += imag(H.Vx[i,3-i]*(2.0*H.Vx[3-i,3-i]*H.Vx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3) + H.Vxx[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)))*imag(df(H.E[i]+1.0im*p.eta, p.T))
    end
    #=
    for w = collect(Float64,-p.W_MAX:2*p.W_MAX/p.W_SIZE:p.W_MAX)
        G = Green(Green_BI(w,p,H)...)
    end=#
    return Drude, BCD, sQMD, dQMD, Inter, dInter
end

@everywhere function Green_ZZZ_BI(p::Parm, H::Hamiltonian)
    
    Drude::Float64 = 0.0
    BCD::Float64 = 0.0
    sQMD::Float64 = 0.0
    dQMD::Float64 = 0.0
    Inter::Float64 = 0.0
    dInter::Float64 = 0.0

    #HV_BI(H)

    for i = 1:2
        Drude += 2.0*real(2.0*H.Vz[i,i]*(H.Vz[i,i]*H.Vz[i,i]*imag(df(H.E[i]+1.0im*p.eta, p.T))/(2.0p.eta) + H.Vz[i,3-i]*H.Vz[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta)*real(df(H.E[i]+1.0im*p.eta, p.T)) + H.Vxx[i,i]*real(df(H.E[i]+1.0im*p.eta, p.T))))/((2.0p.eta)^2)

        BCD += -4.0*imag(H.Vz[i,3-i]*H.Vz[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))real(H.Vz[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        sQMD += -2.0*imag(2.0*H.Vz[i,i]*H.Vz[i,3-i]*H.Vz[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta))*imag(df(H.E[i]+1.0im*p.eta, p.T))/((2.0p.eta)^2)
        dQMD += -4.0*real(H.Vz[i,3-i]*H.Vz[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))imag(H.Vz[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        Inter += -2.0real(H.Vz[i,3-i]*(2.0*H.Vz[3-i,3-i]*H.Vz[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3) + H.Vzz[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)))*real(df(H.E[i]+1.0im*p.eta, p.T))
        dInter += 2.0imag(H.Vz[i,3-i]*(2.0*H.Vz[3-i,3-i]*H.Vz[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3) + H.Vzz[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)))*imag(df(H.E[i]+1.0im*p.eta, p.T))
    end
    #=
    for w = collect(Float64,-p.W_MAX:2*p.W_MAX/p.W_SIZE:p.W_MAX)
        G = Green(Green_BI(w,p,H)...)
    end=#
    return Drude, BCD, sQMD, dQMD, Inter, dInter
end

@everywhere function Green_XXX(p::Parm, H::Hamiltonian)
    G0::Float64 = 0.0
    dw::Float64 = 2.0*p.W_MAX/p.W_SIZE
    for w in collect(-p.W_MAX:dw:p.W_MAX) 
        G = Green(Gk(w,p,H)...)
        G0 += imag(tr(H.Vx*G.dGR*(2.0*H.Vx*G.GR*H.Vx + H.Vxx)*G.GRmA)*df(w,p.T))
    end
    return dw*G0/(2pi)
end
@everywhere function Green_ZZZ(p::Parm, H::Hamiltonian)
    G0::Float64 = 0.0
    dw::Float64 = 2.0*p.W_MAX/p.W_SIZE
    for w in collect(-p.W_MAX:dw:p.W_MAX) 
        G = Green(Gk(w,p,H)...)
        G0 += imag(tr(H.Vz*G.dGR*(2.0*H.Vz*G.GR*H.Vz + H.Vzz)*G.GRmA)*df(w,p.T))
    end
    return dw*G0/(2pi)
end

using DataFrames
using CSV
using Plots
#gr()

function main(arg::Array{String,1})
    #println(arg)
    println("t1, t2, delta, mu, eta, T, K_SIZE, W_MAX, W_SIZE")
    mu0 = collect(-1.0:0.2:1.2)
    #[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
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

    Green_XXX_mu = zeros(Float64,length(mu0))
    Drude_XXX_mu = zeros(Float64,length(mu0))
    BCD_XXX_mu = zeros(Float64,length(mu0))
    sQMD_XXX_mu = zeros(Float64,length(mu0))
    dQMD_XXX_mu = zeros(Float64,length(mu0))
    Inter_XXX_mu = zeros(Float64,length(mu0))
    dInter_XXX_mu = zeros(Float64,length(mu0))

    for j in 1:length(mu0)
        #Parm(t1, t2, delta, mu, kw, eta, T, K_SIZE, W_MAX, W_SIZE)
        p = Parm(parse(Float64,arg[1]), parse(Float64,arg[2]), parse(Float64,arg[3]), mu0[j], pi/4, parse(Float64,arg[4]), parse(Float64,arg[5]), parse(Int,arg[6]), parse(Float64,arg[7]), parse(Int,arg[8]))

        k2 = collect(Iterators.product((-pi:2pi/p.K_SIZE:pi)[1:end-1], (-pi:2pi/p.K_SIZE:pi)[1:end-1]))
        for kz in collect(-pi:2pi/p.K_SIZE:pi)[1:end-1]
            Dr0::Float64 = 0.0
            Dr00::Float64 = 0.0
            BC0::Float64 = 0.0
            dQM0::Float64 = 0.0
            app_QM0::Float64 = 0.0
            Green0::Float64 = 0.0

            DrudeN0::Float64 = 0.0
            BCDN0::Float64 = 0.0
            sQMDN0::Float64 = 0.0
            dQMDN0::Float64 = 0.0
            InterN0::Float64 = 0.0
            dInterN0::Float64 = 0.0
            GreenN0::Float64 = 0.0


            Dr0, Dr00, BC0, dQM0, app_QM0, Green0, DrudeN0, BCDN0, sQMDN0, dQMDN0, InterN0, dInterN0, GreenN0 = @distributed (+) for i in 1:length(k2)
                k = (k2[i][1], k2[i][2], kz)
                Hamk = Hamiltonian(HandV(k,p)...)
                Green = Green_DC_2D(p, Hamk)
                Green_NL = Green_ZZZ(p,Hamk)
                Dr, Dr_0, BC, dQM, app_QM = Green_ZZ_BI(p,Hamk)
                Drude0, BCD0, sQMD0, dQMD0, Inter0, dInter0 = Green_ZZZ_BI(p,Hamk)
                [Dr/(p.K_SIZE^3), Dr_0/(p.K_SIZE^3), BC/(p.K_SIZE^3), dQM/(p.K_SIZE^3), app_QM/(p.K_SIZE^3), Green/(p.K_SIZE^3), Drude0/(p.K_SIZE^3), BCD0/(p.K_SIZE^3), sQMD0/(p.K_SIZE^3), dQMD0/(p.K_SIZE^3), Inter0/(p.K_SIZE^3), dInter0/(p.K_SIZE^3), Green_NL/(p.K_SIZE^3)]
            end
            DrudeX_mu[j] += Dr0
            Drude0_mu[j] += Dr00
            BCX_mu[j] += BC0
            dQMX_mu[j] += dQM0
            app_QM_mu[j] += app_QM0
            Green_mu[j] += Green0

            Green_XXX_mu[j] += GreenN0
            Drude_XXX_mu[j] += DrudeN0
            BCD_XXX_mu[j] += BCDN0
            sQMD_XXX_mu[j] += sQMDN0
            dQMD_XXX_mu[j] += dQMDN0
            Inter_XXX_mu[j] += InterN0
            dInter_XXX_mu[j] += dInterN0
        end
    end
    println("finish the calculation!")
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data = DataFrame(mu=mu0, Drude=DrudeX_mu, BC=BCX_mu, dQM=dQMX_mu, Drude0=Drude0_mu, app_QM=app_QM_mu, Green=Green_mu)
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./T_dep_ZZ.csv", save_data)
    save_data2 = DataFrame(mu=mu0, Drude=Drude_XXX_mu, BCD=BCD_XXX_mu, sQMD=sQMD_XXX_mu, dQMD=dQMD_XXX_mu, Inter=Inter_XXX_mu, dInter=dInter_XXX_mu, Green=Green_XXX_mu)
    CSV.write("./T_dep_XXX.csv", save_data2)

    #gr()
    ENV["GKSwstype"]="nul"
    p1=plot(mu0, DrudeX_mu, label="Drude",xlabel="T",ylabel="σ",title="linear conductivity", width=2.0, marker=:circle)
    p1=plot!(mu0, BCX_mu+dQMX_mu, label="sQM", width=2.0, marker=:circle)
    p1=plot!(mu0, Green_mu-DrudeX_mu-BCX_mu-dQMX_mu, label="TF", width=2.0, marker=:circle)
    p1=plot!(mu0, Green_mu, label="Green", width=2.0, marker=:circle)
    savefig(p1,"./T_dep_ZZ.png")

    p2=plot(mu0, Drude_XXX_mu, label="Drude",xlabel="T",ylabel="σ",title="nonlinear conductivity", width=2.0, marker=:circle)
    p2=plot!(mu0, sQMD_XXX_mu, label="sQMD", width=2.0, marker=:circle)
    p2=plot!(mu0, BCD_XXX_mu, label="BCD", width=2.0, marker=:circle)
    p2=plot!(mu0, dQMD_XXX_mu, label="dQMD", width=2.0, marker=:circle)
    p2=plot!(mu0, Green_XXX_mu, label="Green", width=2.0, marker=:circle)
    savefig(p2,"./T_dep_XXX.png")
    #plot(mu0, BCD_XXX_mu, label="BCD", width=2.0, marker=:circle)
    #plot!(mu0, dQMD_XXX_mu+BCD_XXX_mu, label="sum_TR", width=2.0, marker=:circle)
    p2=plot(mu0, Drude_XXX_mu+sQMD_XXX_mu+BCD_XXX_mu+dQMD_XXX_mu+Inter_XXX_mu+dInter_XXX_mu, label="sum",xlabel="T",ylabel="σ",title="nonlinear conductivity", width=2.0, marker=:circle)
    p2=plot!(mu0, Green_XXX_mu-(Drude_XXX_mu+sQMD_XXX_mu+BCD_XXX_mu+dQMD_XXX_mu+Inter_XXX_mu+dInter_XXX_mu), label="TF", width=2.0, marker=:circle)
    p2=plot!(mu0, Green_XXX_mu, label="Green", width=2.0, marker=:circle)
    #plot!(mu0, total_XXX_mu, label="total", width=2.0, marker=:circle)
    savefig(p2,"./T_dep_XXX2.png")
end

@time main(ARGS)