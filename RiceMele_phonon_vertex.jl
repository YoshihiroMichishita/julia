using Distributed
addprocs(32)

#Parm(t_i, t_e, t_aa, t_bb, Delta, mu, eta, hx, hy, hz, Cp, m, M, U, T, K_SIZE, W_MAX, W_SIZE, W_in)
@everywhere struct Parm
    #Parm of electrons
    t_i::Float64
    t_e::Float64
    t_aa::Float64
    t_bb::Float64
    Delta::Float64
    mu::Float64
    eta::Float64
    hx::Float64
    hy::Float64
    hz::Float64

    #Parm of phonon
    Cp::Float64
    m::Float64
    M::Float64
    U::Float64
    T::Float64
    K_SIZE::Int
    W_MAX::Float64
    W_SIZE::Int
    W_in::Float64
end

@everywhere using SharedArrays

@everywhere mutable struct Hamiltonian
    Hk::SharedArray{ComplexF64,3}
    Vx::SharedArray{ComplexF64,3}
    Vxx::SharedArray{ComplexF64,3}
    E::SharedArray{ComplexF64,2}
    #=
    Hk::Array{ComplexF64,2}
    Vx::Array{ComplexF64,2}
    Vxx::Array{ComplexF64,2}
    E::Array{ComplexF64,1}=#
end

@everywhere mutable struct Green
    GR::SharedArray{ComplexF64,4}
    GA::SharedArray{ComplexF64,4}
    GRmA::SharedArray{ComplexF64,4}
    #=
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    GRmA::Array{ComplexF64,2}
    GRp::Array{ComplexF64,2}
    GAp::Array{ComplexF64,2}
    GRm::Array{ComplexF64,2}
    GAm::Array{ComplexF64,2}=#
end

function HandV(p::Parm)
    Hk = SharedArray{ComplexF64,3}(p.K_SIZE,2,2)
    Vx = SharedArray{ComplexF64,3}(p.K_SIZE,2,2)
    Vxx = SharedArray{ComplexF64,3}(p.K_SIZE,2,2)
    E = SharedArray{ComplexF64,2}(p.K_SIZE,2)

    @distributed for k in 1:p.K_SIZE
        #k0::Float64 = 2.0*(k - p.K_SIZE/2)*pi/p.K_SIZE
        k0::Float64 = 2.0k*pi/p.K_SIZE
        eps::Float64 = -0.5*(p.t_aa + p.t_bb)*cos(k0)-p.mu
        g_x::Float64 = (p.t_e*cos(k0) -p.t_i*sin(k0)) - p.hx
        g_y::Float64 = (p.t_e*sin(k0) + p.t_i*cos(k0)) - p.hy
        g_z::Float64 = p.Delta - p.hz -0.5*(p.t_aa - p.t_bb)*cos(k0)
        gg = [eps, g_x, g_y, g_z]
        Hk[k,:,:] =  gg' * sigma

        eps_vx::Float64 = 0.5*(p.t_aa + p.t_bb)*sin(k0)
        gx_vx::Float64 = (-p.t_e*sin(k0) -p.t_i*cos(k0))
        gy_vx::Float64 = (p.t_e*cos(k0) - p.t_i*sin(k0))
        gz_vx::Float64 = 0.5*(p.t_aa - p.t_bb)*sin(k0)
        gg_x = [eps_vx, gx_vx, gy_vx, gz_vx]
        Vx[k,:,:] = gg_x' * sigma

        eps_vxx::Float64 = 0.5*(p.t_aa + p.t_bb)*cos(k0)
        gx_vxx::Float64 = (-p.t_e*cos(k0) +p.t_i*sin(k0))
        gy_vxx::Float64 = (-p.t_e*sin(k0) - p.t_i*cos(k0))
        gz_vxx::Float64 = 0.5*(p.t_aa - p.t_bb)*cos(k0)
        gg_xx = [eps_vxx, gx_vxx, gy_vxx, gz_vxx]
        Vxx[k,:,:] = gg_xx' * sigma

        E[k,:] = zeros(2)    
    end

    return Hk, Vx, Vxx, E 
end

@everywhere using LinearAlgebra

function Gk(Ham::Hamiltonian, p::Parm)

    GR = SharedArray{ComplexF64,4}(p.K_SIZE,p.W_SIZE,2,2)
    GA = SharedArray{ComplexF64,4}(p.K_SIZE,p.W_SIZE,2,2)
    GRmA = SharedArray{ComplexF64,4}(p.K_SIZE,p.W_SIZE,2,2)

    @distributed for k in 1:p.K_SIZE
        for w in 1:p.W_SIZE
            w0 = 2.0*(w - p.W_SIZE/2)*p.W_MAX/p.W_SIZE
            GR0::Array{ComplexF64,2} = -Ham.Hk[k,:,:] + Matrix{Complex{Float64}}(w0*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)
            GR[k,w,:,:] = inv(GR0)
            GA[k,w,:,:] = GR[k,w,:,:]'
            GRmA[k,w,:,:] = GR[k,w,:,:] - GA[k,w,:,:]
        end
    end
    return GR, GA, GRmA
end
#=
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
=#

@everywhere function Gp(w::Float64, q::Float64, p::Parm)
    #Green関数のinverse
    wq_p = p.Cp*((1.0/p.m + 1.0/p.M) + sqrt(-4.0p.m*p.M*sin(q/2.0)*sin(q/2.0) + (p.m+p.M)^2)/(p.m*p.M))
    wq_m = p.Cp*((1.0/p.m + 1.0/p.M) - sqrt(-4.0p.m*p.M*sin(q/2.0)*sin(q/2.0) + (p.m+p.M)^2)/(p.m*p.M))

    DRq = 1.0/(w-wq_p+1.0im*p.eta) + 1.0/(w-wq_m+1.0im*p.eta)
    DAq = DRq'
    
    return DRq, DAq
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

#=
@everywhere function HandV(k0,p::Parm)
    
    eps::Float64 = -0.5*(p.t_aa + p.t_bb)*cos(k0)-p.mu
    g_x::Float64 = (p.t_e*cos(k0) -p.t_i*sin(k0)) - p.hx
    g_y::Float64 = (p.t_e*sin(k0) + p.t_i*cos(k0)) - p.hy
    g_z::Float64 = p.Delta - p.hz -0.5*(p.t_aa - p.t_bb)*cos(k0)
    gg = [eps, g_x, g_y, g_z]
    H::Array{ComplexF64,2} =  gg' * sigma

    eps_vx::Float64 = 0.5*(p.t_aa + p.t_bb)*sin(k0)
    gx_vx::Float64 = (-p.t_e*sin(k0) -p.t_i*cos(k0))
    gy_vx::Float64 = (p.t_e*cos(k0) - p.t_i*sin(k0))
    gz_vx::Float64 = 0.5*(p.t_aa - p.t_bb)*sin(k0)
    gg_x = [eps_vx, gx_vx, gy_vx, gz_vx]
    Vx::Array{ComplexF64,2} = gg_x' * sigma

    eps_vxx::Float64 = 0.5*(p.t_aa + p.t_bb)*cos(k0)
    gx_vxx::Float64 = (-p.t_e*cos(k0) +p.t_i*sin(k0))
    gy_vxx::Float64 = (-p.t_e*sin(k0) - p.t_i*cos(k0))
    gz_vxx::Float64 = 0.5*(p.t_aa - p.t_bb)*cos(k0)
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
end=#

#@everywhere 
function HV_BI(H::Hamiltonian)

    #@distributed 
    for k in 1:size(H.Hk,1)
        H.E[k,:], BI = eigen(H.Hk[k,:,:])
        H.Hk[k,:,:] = [H.E[k,1] 0.0; 0.0 H.E[k,2]]
        Vx_BI = BI' * H.Vx[k,:,:] * BI
        Vxx_BI = BI' * H.Vxx[k,:,:] * BI
        H.Vx[k,:,:] = Vx_BI
        H.Vxx[k,:,:] = Vxx_BI
    end
end

@everywhere f(e::Float64,T::Float64) = 1.0/(1.0+exp(e/T))
@everywhere df(e::Float64,T::Float64) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T
@everywhere b(e::Float64,T::Float64) = 1.0/(1.0-exp(e/T))

#自己エネルギーを入れてGRを更新
function calcu_phonon_scattering(p::Parm,H::Hamiltonian, G::Green)
    Σw = SharedArray{ComplexF64,4}(p.K_SIZE,p.W_SIZE,2,2)
    @distributed for k in 1:p.K_SIZE
        dk::Float64 = 2pi/p.K_SIZE
        dw::Float64 = 2p.W_MAX/p.W_SIZE
        for w in 1:p.W_SIZE
            for q in 1:p.K_SIZE
                q0 = 2pi*q/p.K_SIZE
                kk = (k+q)%p.K_SIZE
                for wp in 0:p.W_SIZE-1
                    if(wp+w<=p.W_SIZE)
                        wp0 = p.W_MAX*wp/p.W_SIZE
                        Σw[k,w,:,:] += dk * dw * G.GR[kk,w+wp,:,:] * Gp(wp0,q0,p) * b(wp0, p.T)
                    end
                end
            end
        end
    end
    for k in 1:p.K_SIZE
        for w in 1:p.W_SIZE
            w0::Float64 = 2.0p.W_MAX*(w-p.W_SIZE/2)/p.W_SIZE
            GR0 = -H.Hk[k,:,:] + Matrix{Complex{Float64}}(w0*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.U*p.U*Σw[k,w,:,:]
            G.GR[k,w,:,:] = inv(GR0)
            G.GA[k,w,:,:] = G.GR[k,w,:,:]'
            G.GRmA[k,w,:,:] = G.GR[k,w,:,:]-G.GA[k,w,:,:]
        end
    end
end


function PV_calcu_ver(p::Parm, H::Hamiltonian, G::Green)

    Jxxx = SharedArray{Float64,1}(p.K_SIZE)
    @distributed for k in 1:p.K_SIZE
        dk::Float64 = 2pi/p.K_SIZE
        dw::Float64 = 2*p.W_MAX/p.W_SIZE
        wi::Int = convert(Int,round(p.W_in/dw))
        for w in 1:p.W_SIZE
            ww = 2.0p.W_MAX*(w-p.W_SIZE/2)/p.W_SIZE
            for q in 1:p.K_SIZE
                q0 = 2pi*q/p.K_SIZE
                kk = (k+q)%p.K_SIZE
                for wp in 0:p.W_SIZE-1
                    w0 = wp + w
                    if(w0<=p.W_SIZE)
                        wp0 = p.W_MAX*wp/p.W_SIZE
                        DRq, DAq = Gp(wp0,q0,p)
                        Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vxx[kk,:,:] *G.GR[kk,w0,:,:] * DRq * G.GRmA[k,w,:,:])) * f(ww,p.T) *b(wp0,p.T)
                        Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GA[kk,w0,:,:] * H.Vxx[kk,:,:] *G.GA[kk,w0,:,:] * DAq * G.GRmA[k,w,:,:])) * f(ww,p.T) *b(wp0,p.T)

                        Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vxx[kk,:,:] *G.GRmA[kk,w0,:,:] * DRq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                        Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GA[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vxx[kk,:,:] *G.GRmA[kk,w0,:,:] * DAq * G.GR[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                        #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GR * Gqw.GR * Hq.Vxx *Gqw.GRmA * DRq * Gkw.GA)) * f(w+wp,p.T) *b(wp,p.T)
                        #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GA * Gqw.GR * Hq.Vxx *Gqw.GRmA * DAq * Gkw.GR)) * f(w+wp,p.T) *b(wp,p.T)

                        Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GRmA[kk,w0,:,:] * H.Vxx[kk,:,:] *G.GA[kk,w0,:,:] * DRq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                        Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GA[k,w,:,:] * G.GRmA[kk,w0,:,:] * H.Vxx[kk,:,:] *G.GA[kk,w0,:,:] * DAq * G.GR[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                        #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GR * Gqw.GRmA * Hq.Vxx *Gqw.GA * DRq * Gkw.GA)) * f(w+wp,p.T) *b(wp,p.T)
                        #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GA * Gqw.GRmA * Hq.Vxx *Gqw.GA * DAq * Gkw.GR)) * f(w+wp,p.T) *b(wp,p.T)

                        Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GRmA[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vxx[kk,:,:] *G.GR[kk,w0,:,:] * DRq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                        Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GRmA[k,w,:,:] * G.GA[kk,w0,:,:] * H.Vxx[kk,:,:] *G.GA[kk,w0,:,:] * DAq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                        #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRmA * Gqw.GR * Hq.Vxx *Gqw.GR * DRq * Gkw.GA)) * f(w,p.T) *b(wp,p.T)
                        #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRmA * Gqw.GA * Hq.Vxx *Gqw.GA * DAq * Gkw.GA)) * f(w,p.T) *b(wp,p.T)

                        if(w0+wi<=p.W_SIZE)
                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0+wi,:,:]) * H.Vx[kk,:,:] *G.GR[kk,w0,:,:] * DRq * G.GRmA[k,w,:,:])) * f(ww,p.T) *b(wp0,p.T)
                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GA[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0+wi,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0,:,:] * DAq * G.GRmA[k,w,:,:])) * f(ww,p.T) *b(wp0,p.T)

                        

                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0+wi,:,:]) * H.Vx[kk,:,:] *G.GRmA[kk,w0,:,:] * DRq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GA[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0+wi,:,:]) * H.Vx[kk,:,:] *G.GRmA[kk,w0,:,:] * DAq * G.GR[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GR * Gqw.GR * Hq.Vx * (Gqw.GRp + Gqw.GRm) * Hq.Vx *Gqw.GRmA * DRq * Gkw.GA)) * f(w+wp,p.T) *b(wp,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GA * Gqw.GR * Hq.Vx * (Gqw.GRp + Gqw.GRm) * Hq.Vx *Gqw.GRmA * DAq * Gkw.GR)) * f(w+wp,p.T) *b(wp,p.T)

                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w+wi,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GRmA[kk,w0,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0+wi,:,:] * DRq * G.GA[k,w+wi,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRm * Gqw.GRm * Hq.Vx * Gqw.GRmA * Hq.Vx *Gqw.GAm * DRq * Gkw.GAm)) * f(w+wp,p.T) *b(wp,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRp * Gqw.GRp * Hq.Vx * Gqw.GRmA * Hq.Vx *Gqw.GAp * DRq * Gkw.GAp)) * f(w+wp,p.T) *b(wp,p.T)

                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GA[k,w+wi,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GRmA[kk,w0,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0+wi,:,:] * DAq * G.GR[k,w+wi,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GAm * Gqw.GRm * Hq.Vx * Gqw.GRmA * Hq.Vx *Gqw.GAm * DAq * Gkw.GRm)) * f(w+wp,p.T) *b(wp,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GAp * Gqw.GRp * Hq.Vx * Gqw.GRmA * Hq.Vx *Gqw.GAp * DAq * Gkw.GRp)) * f(w+wp,p.T) *b(wp,p.T)

                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GRmA[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GA[kk,w0+wi,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0,:,:] * DRq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GA[k,w,:,:] * G.GRmA[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GA[kk,w0+wi,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0,:,:] * DAq * G.GR[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GR * Gqw.GRmA * Hq.Vx * (Gqw.GAp + Gqw.GAm) * Hq.Vx *Gqw.GA * DRq * Gkw.GA)) * f(w+wp,p.T) *b(wp,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GA * Gqw.GRmA * Hq.Vx * (Gqw.GAp + Gqw.GAm) * Hq.Vx *Gqw.GA * DAq * Gkw.GR)) * f(w+wp,p.T) *b(wp,p.T)

                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GRmA[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0+wi,:,:]) * H.Vx[kk,:,:] *G.GR[kk,w0,:,:] * DRq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            Jxxx += dk * dw * imag(tr(H.Vx[k,:,:] * G.GRmA[k,w,:,:] * G.GA[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GA[kk,w0+wi,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0,:,:] * DAq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRmA * Gqw.GR * Hq.Vx * (Gqw.GRp + Gqw.GRm) * Hq.Vx *Gqw.GR * DRq * Gkw.GA)) * f(w,p.T) *b(wp,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRmA * Gqw.GA * Hq.Vx * (Gqw.GAp + Gqw.GAm) * Hq.Vx *Gqw.GA * DAq * Gkw.GA)) * f(w,p.T) *b(wp,p.T)
                        end

                        if (w0 -wi >= 1)
                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0-wi,:,:]) * H.Vx[kk,:,:] *G.GR[kk,w0,:,:] * DRq * G.GRmA[k,w,:,:])) * f(ww,p.T) *b(wp0,p.T)
                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GA[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0-wi,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0,:,:] * DAq * G.GRmA[k,w,:,:])) * f(ww,p.T) *b(wp0,p.T)

                        

                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0-wi,:,:]) * H.Vx[kk,:,:] *G.GRmA[kk,w0,:,:] * DRq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GA[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0-wi,:,:]) * H.Vx[kk,:,:] *G.GRmA[kk,w0,:,:] * DAq * G.GR[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GR * Gqw.GR * Hq.Vx * (Gqw.GRp + Gqw.GRm) * Hq.Vx *Gqw.GRmA * DRq * Gkw.GA)) * f(w+wp,p.T) *b(wp,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GA * Gqw.GR * Hq.Vx * (Gqw.GRp + Gqw.GRm) * Hq.Vx *Gqw.GRmA * DAq * Gkw.GR)) * f(w+wp,p.T) *b(wp,p.T)

                            

                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * G.GRmA[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GA[kk,w0-wi,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0,:,:] * DRq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GA[k,w,:,:] * G.GRmA[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GA[kk,w0-wi,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0,:,:] * DAq * G.GR[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GR * Gqw.GRmA * Hq.Vx * (Gqw.GAp + Gqw.GAm) * Hq.Vx *Gqw.GA * DRq * Gkw.GA)) * f(w+wp,p.T) *b(wp,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GA * Gqw.GRmA * Hq.Vx * (Gqw.GAp + Gqw.GAm) * Hq.Vx *Gqw.GA * DAq * Gkw.GR)) * f(w+wp,p.T) *b(wp,p.T)

                            if(w-wi>=1)
                                Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w-wi,:,:] * G.GR[kk,w0-wi,:,:] * H.Vx[kk,:,:] * (G.GRmA[kk,w0,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0-wi,:,:] * DRq * G.GA[k,w-wi,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                                #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRm * Gqw.GRm * Hq.Vx * Gqw.GRmA * Hq.Vx *Gqw.GAm * DRq * Gkw.GAm)) * f(w+wp,p.T) *b(wp,p.T)
                                #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRp * Gqw.GRp * Hq.Vx * Gqw.GRmA * Hq.Vx *Gqw.GAp * DRq * Gkw.GAp)) * f(w+wp,p.T) *b(wp,p.T)

                                Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GA[k,w-wi,:,:] * G.GR[kk,w0-wi,:,:] * H.Vx[kk,:,:] * (G.GRmA[kk,w0,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0-wi,:,:] * DAq * G.GR[k,w-wi,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                                #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GAm * Gqw.GRm * Hq.Vx * Gqw.GRmA * Hq.Vx *Gqw.GAm * DAq * Gkw.GRm)) * f(w+wp,p.T) *b(wp,p.T)
                                #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GAp * Gqw.GRp * Hq.Vx * Gqw.GRmA * Hq.Vx *Gqw.GAp * DAq * Gkw.GRp)) * f(w+wp,p.T) *b(wp,p.T)
                            end

                            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GRmA[k,w,:,:] * G.GR[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GR[kk,w0-wi,:,:]) * H.Vx[kk,:,:] *G.GR[kk,w0,:,:] * DRq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            Jxxx += dk * dw * imag(tr(H.Vx[k,:,:] * G.GRmA[k,w,:,:] * G.GA[kk,w0,:,:] * H.Vx[kk,:,:] * (G.GA[kk,w0-wi,:,:]) * H.Vx[kk,:,:] *G.GA[kk,w0,:,:] * DAq * G.GA[k,w,:,:])) * f(ww+wp0,p.T) *b(wp0,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRmA * Gqw.GR * Hq.Vx * (Gqw.GRp + Gqw.GRm) * Hq.Vx *Gqw.GR * DRq * Gkw.GA)) * f(w,p.T) *b(wp,p.T)
                            #Jxxx += dk * dw * imag(tr(Hk.Vx * Gkw.GRmA * Gqw.GA * Hq.Vx * (Gqw.GAp + Gqw.GAm) * Hq.Vx *Gqw.GA * DAq * Gkw.GA)) * f(w,p.T) *b(wp,p.T)
                        end
                        
                    end
                end
            end
        end
        Jxxx[k] *= dk*dw*p.U*p.U/(8*pi^3)/(p.W_in^2)
    end
    return Jxxx
end

function PV_calcu_simple(p::Parm, H::Hamiltonian, G::Green)

    Jxxx = SharedArray{Float64,1}(p.K_SIZE)

    @distributed for k in 1:p.K_SIZE
        dk::Float64 = 2pi/p.K_SIZE
        dw::Float64 = 2*p.W_MAX/p.W_SIZE
        wi::Int = convert(Int,round(p.W_in/dw))
        for w in 1:p.W_SIZE
            ww = 2.0p.W_MAX*(w-p.W_SIZE/2)/p.W_SIZE
            if(w+wi<=p.W_SIZE)
                Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * H.Vx[k,:,:] * (G.GR[k,w+wi,:,:]) * H.Vx[k,:,:] * G.GRmA[k,w,:,:])) * f(ww,p.T)
                Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w+wi,:,:] * H.Vx[k,:,:] * (G.GRmA[k,w,:,:]) * H.Vx[k,:,:] * G.GA[k,w+wi,:,:])) * f(ww,p.T)
                Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GRmA[k,w,:,:] * H.Vx[k,:,:] * (G.GR[k,w+wi,:,:]) * H.Vx[k,:,:] * G.GA[k,w,:,:])) * f(ww,p.T)
            end
            if(w-wi>=1)
                Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * H.Vx[k,:,:] * (G.GR[k,w-wi,:,:]) * H.Vx[k,:,:] * G.GRmA[k,w,:,:])) * f(ww,p.T)
                Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w-wi,:,:] * H.Vx[k,:,:] * (G.GRmA[k,w,:,:]) * H.Vx[k,:,:] * G.GA[k,w-wi,:,:])) * f(ww,p.T)
                Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GRmA[k,w,:,:] * H.Vx[k,:,:] * (G.GR[k,w-wi,:,:]) * H.Vx[k,:,:] * G.GA[k,w,:,:])) * f(ww,p.T)
            end
            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GR[k,w,:,:] * H.Vxx[k,:,:] * G.GRmA[k,w,:,:])) * f(ww,p.T)
            Jxxx[k] += dk * dw * imag(tr(H.Vx[k,:,:] * G.GRmA[k,w,:,:] * H.Vxx[k,:,:] * G.GA[k,w,:,:])) * f(ww,p.T)
        end
        Jxxx[k] /= ((2*pi)/(p.W_in^2))
    end
    return Jxxx
end


using DataFrames
using CSV
using Plots

function main(arg::Array{String,1})

    #Parm(t_i, t_e, t_aa, t_bb, Delta, mu, eta, hx, hy, hz, Cp, m, M, U, T, K_SIZE, W_MAX, W_SIZE, W_in)
    p = Parm(parse(Float64,arg[1]), parse(Float64,arg[2]), 0.1, 0.2, parse(Float64,arg[3]), parse(Float64,arg[4]), 0.02, 0.6, 0.0, 0.0, 0.5, parse(Float64,arg[5]),parse(Float64,arg[6]), parse(Float64,arg[7]),parse(Float64,arg[8]), parse(Int,arg[9]), 1.5, parse(Int,arg[10]),parse(Float64,arg[11]))
    kk = range(0.0,2pi,length=p.K_SIZE)
    
    H = Hamiltonian(HandV(p)...)
    G = Green(Gk(H,p)...)

    calcu_phonon_scattering(p,H,G)

    #PV_XXX_ver_mu = zeros(Float64,length(kk))
    PV_XXX_ver_mu = PV_calcu_ver(p, H, G)
    PV_XXX_mu = PV_calcu_simple(p, H, G)


    
    e1_k = SharedArray{Float64,1}(p.K_SIZE)
    #zeros(Float64,length(kk))
    e2_k = SharedArray{Float64,1}(p.K_SIZE)
    wq_p = zeros(Float64,length(kk))
    wq_m = zeros(Float64,length(kk))
    
    HV_BI(H)

    for i in 1:length(kk)
        if(i==1)
            println(H.Hk[i,:,:])
            println(H.E[i,:])
        end
        e1_k[i] = H.E[i,1]
        e2_k[i] = H.E[i,2]
        wq_p[i] = p.Cp*((1.0/p.m + 1.0/p.M) + sqrt(-4.0p.m*p.M*sin(kk[i]/2.0)*sin(kk[i]/2.0) + (p.m+p.M)^2)/(p.m*p.M))
        wq_m[i] = p.Cp*((1.0/p.m + 1.0/p.M) - sqrt(-4.0p.m*p.M*sin(kk[i]/2.0)*sin(kk[i]/2.0) + (p.m+p.M)^2)/(p.m*p.M))
    end
    

    save_data1 = DataFrame(k=kk, e1=e1_k, e2=e2_k, Wq_m=wq_m, Wq_p=wq_p)
    CSV.write("./disp_T002.csv", save_data1)
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data2 = DataFrame(k=kk, PV_simple=PV_XXX_mu, PV_ver=PV_XXX_ver_mu)
    CSV.write("./PV_kdep_XXX_T002.csv", save_data2)

    
    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)
    
    p1 = plot(kk, e1_k, label="e1",xlabel="kx",ylabel="e",title="dispersion", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(kk, e2_k, label="e2", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(kk, wq_m, label="wq_m", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(kk, wq_p, label="wq_p", width=4.0, marker=:circle, markersize = 4.8)
    savefig(p1,"./disp.png")

    p2 = plot(kk, PV_XXX_mu, label="w/o vertex",xlabel="kx",ylabel="PV",title="Ω-dependence", width=4.0, marker=:circle, markersize = 4.8)
    p2 = plot!(kk, PV_XXX_ver_mu, label="vertex", width=4.0, marker=:circle, markersize = 4.8)
    savefig(p2,"./kdep_PV_XXX.png")
end

@time main(ARGS)