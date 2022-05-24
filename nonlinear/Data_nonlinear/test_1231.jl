using LinearAlgebra

N = 2

module Parm
    export parm1

    t = 1.0
    V = 0.5
    mu = 0
    T = 0.01
    K_SIZE = 100
    W_SIZE = 2000
    W_MAX = 0.2
    dw = 2 * W_MAX / W_SIZE
    function parm1(filename)
        include(pwd()*"/"*filename)
        return t,V,mu,T,K_SIZE,W_SIZE,W_MAX,dw
    end
end



module Hamiltonian
    using ..Parm
    export Ham,setV

    mutable struct HandV
        Hk::Array{ComplexF64,2}
        Vx::Array{ComplexF64,2}
        Vy::Array{ComplexF64,2}
        Vxx::Array{ComplexF64,2}
        Vyx::Array{ComplexF64,2}
    end

    function Ham(k)
        Hk = [Parm.t*(cos(k[1])+cos(k[2])) Parm.V
        Parm.V -Parm.t*(cos(k[1])+cos(k[2]))]
        return Hk
    end

    function setV(k)
        Vx = [-Parm.t*(sin(k[1])) 0.0
        0.0 Parm.t*(sin(k[1]))]

        Vy = [-Parm.t*(sin(k[2])) 0.0
        0.0 Parm.t*(sin(k[2]))]

        Vxx = [-Parm.t*(cos(k[1])) 0.0
        0.0 Parm.t*(cos(k[1]))]

        Vyx = [0.0 0.0
        0.0 0.0]
        return Vx,Vy,Vxx,Vyx
    end

end


module Green
    using ..Parm
    using ..Hamiltonian
    using LinearAlgebra

    export green,dgreen

    mutable struct GG
        GR::Array{ComplexF64,2}
        GA::Array{ComplexF64,2}
        dGR::Array{ComplexF64,2}
        dGA::Array{ComplexF64,2}
    end

    function green(w,HV::HandV)
        GR0 = -HV.Hk + Matrix{Complex{Float64}}(w*I,2,2) + Matrix{Complex{Float64}}(1.0im*I,2,2)
        GA0 = GR0'
        GR = inv(GR0)
        GA = inv(GA0)
        return GR,GA
    end

    function dgreen()
        dGR = - GR * GR
        dGA = - GA * GA
        return dGR,dGA
    end

    #=
    export rho_xx

    function fermiD(w,T)
        f = 1.0/(1.0 + exp(w/T))
        return f
    end

    function dfermiD(w,T)
        f = -1.0/(1.0 + exp(w/T))//(1.0 + exp(-w/T))/T
        return f
    end

    function rho_xx(w)
        GR,GA = green(w)
        XGR = Hamiltonian.Vx * GR
        XGA = Hamiltonian.Vx * GA
        rx = real(tr(XGR *  XGA) * dfermiD(w,Parm.T))
        return rx
    end
    =#

end


module transport
    using LinearAlgebra
    using ..Parm
    using ..Hamiltonian
    using ..Green

    export rho_xx,rho_yx,rho_yxx,rho_xxx

    #XGR = Hamiltonian.Vx * Green.GR
    #XGA = Hamiltonian.Vx * Green.GA
    #YdG = Hamiltonian.Vy * Green.dGR

    function fermiD(w,T)
        f = 1.0/(1.0 + exp(w/T))
        return f
    end

    function dfermiD(w,T)
        f = -1.0/(1.0 + exp(w/T))/(1.0 + exp(-w/T))/T
        return f
    end

    function rho_xx(w,HV::HandV,G::GG)
        rx = real(Parm.dw * tr(HV.Vx * G.GR * HV.Vx * G.GA) * dfermiD(w,Parm.T))
        #rx = Parm.dw
        return rx
    end

    #=
    function rho_yx(w)
        ryx = Parm.dw * tr(Hamiltonian.Vy * Green.dGR *  (XGR - XGA )) * fermiD(w,Parm.T)
        return ryx
    end

    function rho_yxx(w)
        ryxx = Parm.dw * tr(Hamiltonian.Vy * Green.dGR *  (XGR * XGA + Hamiltonian.Vxx * Green.GA) ) * dfermiD(w,Parm.T)
        return ryxx
    end

    function rho_xxx(w)
        rxxx = Parm.dw * tr(Hamiltonian.Vx * Green.dGR *  (XGR * XGA + Hamiltonian.Vxx * Green.GA) ) * dfermiD(w,Parm.T)
        return rxxx
    end
    =#
end


#=
module transport2
    using LinearAlgebra
    using ..Parm
    using ..Hamiltonian
    using ..Green

    export rho_xx,rho_yx,rho_yxx,rho_xxx

    function fermiD(w,T)
        f = 1.0/(1.0 + exp(w/T))
        return f
    end

    function dfermiD(w,T)
        f = -1.0/(1.0 + exp(w/T))//(1.0 + exp(-w/T))/T
        return f
    end

    function rho_xx(w)
        Green.GR,Green.GA = green(w)
        XGR = Hamiltonian.Vx * Green.GR
        XGA = Hamiltonian.Vx * Green.GA
        rx = tr(XGR *  XGA) * dfermiD(w,Parm.T)
        return rx
    end


end
=#


using .Parm
Parm.parm1("./Documents/Codes/julia/parm.jl")
#Parm.parm1(ARGS[1])



q = collect(Float64,0:pi/Parm.K_SIZE:4pi)
s = length(q)
ev1 = zeros(s)
ev2 = zeros(s)

#=
for i = 1:s
    if i<(s/4)
        k = [q[i],0]
        #H = Ham(k)
        #e,d = eigen(H)
        Hamiltonian.Ham(k)
        e,d = eigen(Hamiltonian.Hk)
        ev1[i] = e[1]
        ev2[i] = e[2]
    
    elseif i<(s/2) && i>=(s/4)
        k = [2pi-q[i],q[i]-pi]
        #H = Ham(k)
        #e,d = eigen(H)
        Hamiltonian.Ham(k)
        e,d = eigen(Hamiltonian.Hk)
        ev1[i] = e[1]
        ev2[i] = e[2]
    elseif i<(3*s/4) && i>=(s/2)
        k = [0,3pi-q[i]]
        #H = Ham(k)
        #e,d = eigen(H)
        Hamiltonian.Ham(k)
        e,d = eigen(Hamiltonian.Hk)
        ev1[i] = e[1]
        ev2[i] = e[2]
    else
        k = [q[i]-3pi,q[i]-3pi]
        #H = Ham(k)
        #e,d = eigen(H)
        Hamiltonian.Ham(k)
        e,d = eigen(Hamiltonian.Hk)
        ev1[i] = e[1]
        ev2[i] = e[2]
    end
end
=#


using Plots

plot(q, ev1)
plot!(q, ev2)
#savefig("plot.png")

#using QuadGK

k0 = collect(Float64,-pi:2*pi/Parm.K_SIZE:pi)

dk = 2.0*pi/Parm.K_SIZE
w_sum =  zeros(Float64, Parm.K_SIZE, Parm.K_SIZE)
total = 0.0

using .Hamiltonian
using .Green


for i = 1:Parm.K_SIZE
    for j = 1:Parm.K_SIZE
        kk = [k0[i],k0[j]]
        
        HamV = Hamiltonian.HandV(Ham(kk),setV(kk))
        #Hamiltonian.Hk = Hamiltonian.Ham(kk)
        #Hamiltonian.Vx,Hamiltonian.Vy,Hamiltonian.Vxx,Hamiltonian.Vyx = Hamiltonian.setV(kk)
        println(HamV.Vx)
        #=
        for ww = 1:Parm.W_SIZE 
            w = 2.0 * (ww-Parm.W_SIZE/2) * Parm.W_MAX / Parm.W_SIZE
            #println(w)
            Green.GR,Green.GA = green(w)
            #Green.dgreen()
            w_sum[i,j] += transport.rho_xx(w)/(2.0*pi)
        end
        println(Green.GR)
        println(w_sum[i,j])
        #w_sum[i][j] = quadgk(Green.rho_xx,-Parm.W_MAX,Parm.W_MAX, rtol=1e-6 )[1]/2pi
        total += dk* dk * w_sum[i,j]/(4*pi*pi)
        =#
    end
end

#println(total)

#using PyPlot: PyPlot, plt

#plt.figure(figsize=(5,5))
#plt.pcolormesh(k0,k0,w_sum, cmap="CMRmap")

