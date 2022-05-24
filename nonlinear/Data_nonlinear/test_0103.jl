using LinearAlgebra

struct Parm
    t::Float64
    V::Float64
    mu::Float64
    T::Float64
    W_MAX::Float64
    K_SIZE::Int
    W_SIZE::Int
end

mutable struct Hamiltonian
    Hk::Array{ComplexF64,2}
    Vx::Array{ComplexF64,2}
    Vy::Array{ComplexF64,2}
    Vxx::Array{ComplexF64,2}
    Vyx::Array{ComplexF64,2}
end

mutable struct Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    dGR::Array{ComplexF64,2}
    dGA::Array{ComplexF64,2}
end

module transport
    using LinearAlgebra
    export rho_xx

    
    function fermiD(w::Float64, T::Float64)
        f = 1.0/(1.0 + exp(w/T))
        return f
    end

    function dfermiD(w::Float64, T::Float64)
        f = -1.0/(1.0 + exp(w/T))/(1.0 + exp(-w/T))/T
        return f
    end

    function rho_xx(w::Float64, p::Main.Parm, Ham::Main.Hamiltonian, G::Main.Green)
        rx = real(2.0 * p.W_MAX / p.W_SIZE * tr(Ham.Vx * G.GR * Ham.Vx * G.GA) * dfermiD(w, p.T))
        return rx
    end
end

function HandV(k::Array{Float64},p::Parm, Ham::Hamiltonian)
    Ham.Hk = [p.t*(cos(k[1])+cos(k[2])) p.V
    p.V -p.t*(cos(k[1])+cos(k[2]))]

    Ham.Vx = [-p.t*(sin(k[1])) 0.0
    0.0 p.t*(sin(k[1]))]

    Ham.Vy = [-p.t*(sin(k[2])) 0.0
    0.0 p.t*(sin(k[2]))]

    Ham.Vxx = [-p.t*(cos(k[1])) 0.0
    0.0 p.t*(cos(k[1]))]

    Ham.Vyx = [0.0 0.0
    0.0 0.0]
end

function Gk(w::Float64, p::Parm, Ham::Hamiltonian, G::Green)
    GR0 = -Ham.Hk + Matrix{Complex{Float64}}(w*I,2,2) + 0.005*Matrix{Complex{Float64}}(1.0im*I,2,2)
    GA0 = GR0'
    G.GR = inv(GR0)
    G.GA = inv(GA0)
    G.dGR = - G.GR * G.GR
    G.dGA = - G.GA * G.GA
end


p = Parm(1.0,0.2,0.4,0.01,0.2,45,400)

kx = collect(Float64,-pi:2*pi/p.K_SIZE:pi)
ky = collect(Float64,-pi:2*pi/p.K_SIZE:pi)

using .transport

total = 0.0
w_sum =  zeros(Float64, p.K_SIZE+1, p.K_SIZE+1)
dk = 2.0*pi/p.K_SIZE
dw = 2.0 * p.W_MAX /p.W_SIZE

for i = 1:p.K_SIZE+1
    for j = 1:p.K_SIZE+1
        kk = [kx[i],ky[j]]
        HVk = Hamiltonian(zeros(ComplexF64,2,2),zeros(ComplexF64,2,2),zeros(ComplexF64,2,2),zeros(ComplexF64,2,2),zeros(ComplexF64,2,2))
        HandV(kk,p,HVk)
        for ww = 1:p.W_SIZE
            w = dw * (ww-p.W_SIZE/2)
            G = Green(zeros(ComplexF64,2,2),zeros(ComplexF64,2,2),zeros(ComplexF64,2,2),zeros(ComplexF64,2,2))
            Gk(w,p,HVk,G)
            w_sum[i,j] += dw * transport.rho_xx(w, p, HVk, G) /(2.0*pi)
        end
        total += dk * dk * w_sum[i,j]/(4.0 * pi^2)
    end
end

println(total)

using Plots

heatmap(kx,ky,w_sum)

#=
using PyPlot: PyPlot, plt

plt.figure(figsize=(5,5))
plt.pcolormesh(kx,ky,w_sum, cmap="CMRmap")
plt.show()
=#