#include("2D_TMD_parm.jl")

#@everywhere 
using LinearAlgebra

#@everywhere 
mutable struct Hamiltonian
    Hk::Array{ComplexF64,2}
    Va::Array{ComplexF64,2}
    Vb::Array{ComplexF64,2}
    Vc::Array{ComplexF64,2}
    Vab::Array{ComplexF64,2}
    Vbc::Array{ComplexF64,2}
    Vca::Array{ComplexF64,2}
    Vabc::Array{ComplexF64,2}
    E::Array{ComplexF64,1}
end

using ForwardDiff

#=
function set_vx_fd(k,p::Parm)
    m(k) = set_H_v(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    return gg
end

function set_vy_fd(k,p::Parm)
    m(k) = set_H_v(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,2]
    return gg
end

function set_vxx_fd(k,p::Parm)
    m(k) = set_vx_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    return gg
end

function set_vxy_fd(k,p::Parm)
    m(k) = set_vy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    return gg
end

function set_vyy_fd(k,p::Parm)
    m(k) = set_vy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,2]
    return gg
end

function set_vxxx_fd(k,p::Parm)
    m(k) = set_vxx_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    return gg
end

function set_vxxy_fd(k,p::Parm)
    m(k) = set_vxy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    return gg
end

function set_vxyy_fd(k,p::Parm)
    m(k) = set_vyy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,1]
    return gg
end

function set_vyyy_fd(k,p::Parm)
    m(k) = set_vyy_fd(k,p)
    gg = (ForwardDiff.jacobian(m, k))[:,2]
    return gg
end=#
function VtoM(v)
    M::Array{ComplexF64,2} = sigma' * v 
    return M
end
#set Va~Vc
function set_v1_fd(k,p::Parm)
    m(x) = set_H_v(x,p)
    V = (ForwardDiff.jacobian(m, k))
    Va = V[:, p.abc[1]]
    Vb = V[:, p.abc[2]]
    Vc = V[:, p.abc[3]]
    return Va, Vb, Vc
end

#set Va~Vca
function set_v2_fd(k,p::Parm)
    Va(x)= set_v1_fd(x,p)[1]
    Vb(x)= set_v1_fd(x,p)[2]
    Vab = (ForwardDiff.jacobian(Va, k))[:,p.abc[2]]
    Vbc = (ForwardDiff.jacobian(Vb, k))[:,p.abc[3]]
    Vca = (ForwardDiff.jacobian(Va, k))[:,p.abc[3]]
    return Vab, Vbc, Vca
end

#set Va~Vabc
function set_v3_fd(k,p::Parm)
    Vab(x) = set_v2_fd(x,p)[1]
    Vabc = (ForwardDiff.jacobian(Vab, k))[:,p.abc[3]]
    return Vabc
end

#init_Hamiltonian
#function HandV_fd(k0::NTuple{2, Float64},p::Parm)
function HandV_fd(k::Vector{Float64},p::Parm)
    #k = [k0[1], k0[2]]

    H = set_H(k,p)

    #=
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
    =#
    Va_v, Vb_v, Vc_v = set_v1_fd(k,p)
    Vab_v, Vbc_v, Vca_v = set_v2_fd(k,p)
    Vabc_v = set_v3_fd(k,p)
    Va = VtoM(Va_v)
    Vb = VtoM(Vb_v)
    Vc = VtoM(Vc_v)
    Vab = VtoM(Vab_v)
    Vbc = VtoM(Vbc_v)
    Vca = VtoM(Vca_v)
    Vabc = VtoM(Vabc_v)
    
    E::Array{ComplexF64,1} = zeros(p.H_size)

    return H, Va, Vb, Vc, Vab, Vbc, Vca, Vabc, E 
end

# when you focus on the DC (input is also DC) conductivity
#@everywhere 
mutable struct DC_Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    GRmA::Array{ComplexF64,2}
    dGR::Array{ComplexF64,2}
    dGA::Array{ComplexF64,2}
    ddGR::Array{ComplexF64,2}
end

#@everywhere 
function set_DC_Gk(w::Float64, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,p.H_size,p.H_size) + p.eta*Matrix{Complex{Float64}}(1.0im*I,p.H_size,p.H_size)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    GRmA::Array{ComplexF64,2} = GR - GA

    #If you use the self-energy, you should change this part.
    dGR::Array{ComplexF64,2} = - GR * GR
    dGA::Array{ComplexF64,2} = - GA * GA
    ddGR::Array{ComplexF64,2} = 2.0 * GR * GR * GR
    
    return GR, GA, GRmA, dGR, dGA, ddGR
end

# when you focus on the photo-voltaic effect
#@everywhere 
mutable struct PV_Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    GRmA::Array{ComplexF64,2}
    GRp::Array{ComplexF64,2}
    GAp::Array{ComplexF64,2}
    GRm::Array{ComplexF64,2}
    GAm::Array{ComplexF64,2}
end

#@everywhere 
function set_PV_Gk(w::Float64, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,p.H_size,p.H_size) + p.eta*Matrix{Complex{Float64}}(1.0im*I,p.H_size,p.H_size)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    GRmA::Array{ComplexF64,2} = GR - GA

    #If you use the self-energy, you should change this part.
    GRp0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w+p.W_in)*I,p.H_size,p.H_size) + p.eta*Matrix{Complex{Float64}}(1.0im*I,p.H_size,p.H_size)

    GRp::Array{ComplexF64,2} = inv(GRp0)
    GAp::Array{ComplexF64,2} = GRp'

    GRm0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w-p.W_in)*I,p.H_size,p.H_size) + p.eta*Matrix{Complex{Float64}}(1.0im*I,p.H_size,p.H_size)

    GRm::Array{ComplexF64,2} = inv(GRm0)
    GAm::Array{ComplexF64,2} = GRm'
    
    return GR, GA, GRmA, GRp, GAp, GRm, GAm
end

# when you focus on the second harmonic generation
#@everywhere 
mutable struct SHG_Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    GRmA::Array{ComplexF64,2}
    GRp::Array{ComplexF64,2}
    GAp::Array{ComplexF64,2}
    GRm::Array{ComplexF64,2}
    GAm::Array{ComplexF64,2}
    GRpp::Array{ComplexF64,2}
    GApp::Array{ComplexF64,2}
    GRmm::Array{ComplexF64,2}
    GAmm::Array{ComplexF64,2}
end

#@everywhere 
function set_SHG_Gk(w::Float64, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,p.H_size,p.H_size) + p.eta*Matrix{Complex{Float64}}(1.0im*I,p.H_size,p.H_size)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    GRmA::Array{ComplexF64,2} = GR - GA

    #If you use the self-energy, you should change this part.
    GRp0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w+p.W_in)*I,p.H_size,p.H_size) + p.eta*Matrix{Complex{Float64}}(1.0im*I,p.H_size,p.H_size)

    GRp::Array{ComplexF64,2} = inv(GRp0)
    GAp::Array{ComplexF64,2} = GRp'

    GRm0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w-p.W_in)*I,p.H_size,p.H_size) + p.eta*Matrix{Complex{Float64}}(1.0im*I,p.H_size,p.H_size)

    GRm::Array{ComplexF64,2} = inv(GRm0)
    GAm::Array{ComplexF64,2} = GRm'

    GRpp0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w+2*p.W_in)*I,p.H_size,p.H_size) + p.eta*Matrix{Complex{Float64}}(1.0im*I,p.H_size,p.H_size)

    GRpp::Array{ComplexF64,2} = inv(GRpp0)
    GApp::Array{ComplexF64,2} = GRpp'

    GRmm0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w-2*p.W_in)*I,p.H_size,p.H_size) + p.eta*Matrix{Complex{Float64}}(1.0im*I,p.H_size,p.H_size)

    GRmm::Array{ComplexF64,2} = inv(GRmm0)
    GAmm::Array{ComplexF64,2} = GRmm'
    
    return GR, GA, GRmA, GRp, GAp, GRm, GAm, GRpp, GApp, GRmm, GAmm
end

#change the basis to the band-index representation
#@everywhere 
function HV_BI!(H::Hamiltonian)

    H.E, BI::Array{ComplexF64,2} = eigen(H.Hk)
    H.Hk = Diagonal(H.E)
    #H.Hk = [H.E[1] 0.0; 0.0 H.E[2]]
    Va_BI::Array{ComplexF64,2} = BI' * H.Va * BI
    Vb_BI::Array{ComplexF64,2} = BI' * H.Vb * BI
    Vc_BI::Array{ComplexF64,2} = BI' * H.Vc * BI
    Vab_BI::Array{ComplexF64,2} = BI' * H.Vab * BI
    Vbc_BI::Array{ComplexF64,2} = BI' * H.Vbc * BI
    Vca_BI::Array{ComplexF64,2} = BI' * H.Vca * BI
    Vabc_BI::Array{ComplexF64,2} = BI' * H.Vabc * BI
    

    H.Va = Va_BI
    H.Vb = Vb_BI
    H.Vc = Vc_BI
    H.Vab = Vab_BI
    H.Vbc = Vbc_BI
    H.Vca = Vca_BI
    H.Vabc = Vabc_BI
end

#@everywhere 
f(e::Float64,T::Float64) = 1.0/(1.0+exp(e/T))
#@everywhere 
df(e::Float64,T::Float64) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T
ddf(e::Float64,T::Float64) = df(e,T)*(f(e,T)-f(-e,T))/T

#@everywhere 
f(e::ComplexF64,T::Float64) = 1.0/(1.0+exp(e/T))
#@everywhere 
df(e::ComplexF64,T::Float64) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T
ddf(e::ComplexF64,T::Float64) = df(e,T)*(f(e,T)-f(-e,T))/T

#calculate the DC (input is also DC) linear conductivity in band-index representation with teh imaginary part of the Fermi distribution function
#@everywhere 
function Green_DC_BI_linear_full(p::Parm, H::Hamiltonian)
    
    Drude::Float64 = 0.0
    Drude0::Float64 = 0.0
    BC::Float64 = 0.0
    dQM::Float64 = 0.0
    app_QM::Float64 = 0.0
    
    
    HV_BI!(H)

    for i = 1:2
        Drude += -real(H.Va[i,i]*H.Vb[i,i])*real(df(H.E[i]+1.0im*p.eta, p.T))/(2.0p.eta)
        Drude0 += -real(H.Va[i,i]*H.Vb[i,i])*real(df(H.E[i], p.T))/(2.0p.eta)
        BC += imag(H.Va[i,3-i]*H.Vb[3-i,i])*real(f(H.E[i]+1.0im*p.eta, p.T)/(H.E[i]-H.E[3-i]+2.0im*p.eta)^2)
        dQM += real(H.Va[i,3-i]*H.Vb[3-i,i])*imag(df(H.E[i]+1.0im*p.eta, p.T)/(H.E[i]-H.E[3-i]+2.0im*p.eta))
        app_QM += 2.0*p.eta*real(H.Va[i,3-i]*H.Vb[3-i,i])/((H.E[i]-H.E[3-i])^2+4.0*p.eta^2)*(-df(H.E[i], p.T))
    end
    return Drude, Drude0, BC, dQM, app_QM
end

#calculate DC conductivity with the Green function method
#@everywhere 
function Green_DC(p::Parm, H::Hamiltonian)
    sym::Float64 = 0.0
    asym::Float64 =0.0
    #dw::Float64 = p.W_MAX/p.W_SIZE/pi


    mi = minimum([p.W_MAX,10p.T])
    dw::Float64 = 2mi/p.W_SIZE
    for w in collect(-mi:dw:mi)
        G = DC_Green(set_DC_Gk(w,p,H)...)
        sym += real(tr(H.Va*G.GR*H.Vb*G.GRmA))*df(w,p.T)
        asym += -real(tr(H.Va*G.dGR*H.Vb*G.GRmA))*f(w,p.T)
    end
    for w in collect(-p.W_MAX:dw:-mi)
        G = DC_Green(set_DC_Gk(w,p,H)...)
        asym += -real(tr(H.Va*G.dGR*H.Vb*G.GRmA))*f(w,p.T)
    end
    return dw*sym/(2pi), dw*asym/(2pi)
end

#calculate the DC (input is also DC) nonlinear conductivity in band-index representation with teh imaginary part of the Fermi distribution function
#@everywhere 
#=
function Green_DC_BI_nonlinear_full(p::Parm, H::Hamiltonian)
    
    Drude::Float64 = 0.0
    BCD::Float64 = 0.0
    sQMD::Float64 = 0.0
    dQMD::Float64 = 0.0
    Inter::Float64 = 0.0
    dInter::Float64 = 0.0

    HV_BI!(H)

    for i = 1:2
        Drude += 2.0*real(2.0*H.Va[i,i]*(H.Vb[i,i]*H.Vc[i,i]*imag(df(H.E[i]+1.0im*p.eta, p.T))/(2.0p.eta) + H.Vb[i,3-i]*H.Vc[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta)*real(df(H.E[i]+1.0im*p.eta, p.T)) + H.Vbc[i,i]*real(df(H.E[i]+1.0im*p.eta, p.T))))/((2.0p.eta)^2)

        BCD += -4.0*imag(H.Va[i,3-i]*H.Vb[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))real(H.Vc[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        sQMD += -4.0*imag(2.0*H.Va[i,i]*H.Vb[i,3-i]*H.Vc[3-i,i]/(H.E[i]-H.E[3-i]+2.0im*p.eta))*imag(df(H.E[i]+1.0im*p.eta, p.T))/((2.0p.eta)^2)
        dQMD += -4.0*real(H.Va[i,3-i]*H.Vb[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2))*imag(H.Vc[i,i]*df(H.E[i]+1.0im*p.eta, p.T))/p.eta
        Inter += -2.0real(H.Va[i,3-i]*(2.0*H.Vb[3-i,3-i]*H.Vc[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3) + H.Vbc[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)))*real(df(H.E[i]+1.0im*p.eta, p.T))
        dInter += 2.0imag(H.Va[i,3-i]*(2.0*H.Vb[3-i,3-i]*H.Vc[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^3) + H.Vbc[3-i,i]/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)))*imag(df(H.E[i]+1.0im*p.eta, p.T))
    end
    #=
    for w = collect(Float64,-p.W_MAX:2*p.W_MAX/p.W_SIZE:p.W_MAX)
        G = Green(Green_BI(w,p,H)...)
    end=#
    return Drude, BCD, sQMD, dQMD, Inter, dInter
end=#

function Green_DC_BI_nonlinear_full(p::Parm, H::Hamiltonian)
    BCD::Float64 = 0.0
    Drude::Float64 = 0.0
    ChS::Float64 = 0.0
    gBC::Float64 = 0.0
    HV_BI!(H)
    for i = 1:2
        BCD += imag(H.Va[i,3-i]*H.Vb[3-i,i]*H.Vc[i,i] + H.Va[i,3-i]*H.Vc[3-i,i]*H.Vb[i,i])*real(1.0/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)*df(H.E[i]+1.0im*p.eta,p.T))/(2.0p.eta)
        Drude+= real(H.Va[i,i]*(2.0*H.Vb[i,i]*H.Vc[i,i]/(2.0im*p.eta) + (H.Vb[i,3-i]*H.Vc[3-i,i]+H.Vc[i,3-i]*H.Vb[3-i,i] + H.Vbc[i,i])*real(1.0/(H.E[i]-H.E[3-i]+2.0im*p.eta)))/(-4.0*p.eta^2)*df(H.E[i]+1.0im*p.eta,p.T)) 
        ChS += real(H.Va[i,3-i]*H.Vb[3-i,i]*H.Vc[i,i] + H.Va[i,3-i]*H.Vc[3-i,i]*H.Vb[i,i] + H.Va[i,3-i]*H.Vbc[3-i,i])*imag(1.0/((H.E[i]-H.E[3-i]+2.0im*p.eta)^2)*df(H.E[i]+1.0im*p.eta,p.T))/(2.0p.eta)
        ChS += real(H.Va[i,i]*(H.Vb[i,3-i]*H.Vc[3-i,i]+H.Vc[i,3-i]*H.Vb[3-i,i])*2.0im/((H.E[i]-H.E[3-i])^2+4.0*p.eta^2)/(-4.0*p.eta)*df(H.E[i]+1.0im*p.eta,p.T)) 
        ChS += real(H.Va[i,3-i]*(H.Vb[3-i,3-i]*H.Vc[3-i,i]+H.Vc[3-i,3-i]*H.Va[3-i,i]))*real(1.0/(H.E[i]-H.E[3-i]+2.0im*p.eta)^3*df(H.E[i]+1.0im*p.eta,p.T))
        
        gBC += -imag(H.Va[i,3-i]*(H.Vb[3-i,3-i]*H.Vc[3-i,i]+H.Vc[3-i,3-i]*H.Vb[3-i,i]))*imag(1.0/(H.E[i]-H.E[3-i]+2.0im*p.eta)^3*df(H.E[i]+1.0im*p.eta,p.T))
        gBC += -imag(H.Va[i,3-i]*H.Vbc[3-i,i])*imag(1.0/(H.E[i]-H.E[3-i]+2.0im*p.eta)^2*df(H.E[i]+1.0im*p.eta,p.T))
    end
    return Drude, BCD, ChS, gBC
end

#calculate the Fermi surface term the DC (input is also DC) nonlinear conductivity with the Green function method
#@everywhere 
function Green_DC_nonlinear(p::Parm, H::Hamiltonian)
    G_sur::Float64 = 0.0
    G_sea::Float64 = 0.0
    mi = minimum([p.W_MAX,10p.T])
    dw::Float64 = mi/p.W_SIZE/pi
    for w in collect(-mi:2.0mi/p.W_SIZE:mi)
        G = DC_Green(set_DC_Gk(w,p,H)...)
        G_sur += 2.0imag(tr(H.Va*G.dGR*(2.0*H.Vb*G.GR*H.Vc + H.Vbc)*G.GRmA)*df(w,p.T))
        G_sea += -2.0imag(tr(H.Va*G.dGR*(H.Vbc +2.0*H.Vb*G.GR*H.Vc)*G.dGR))*f(w,p.T)
    end
    for w in collect(-p.W_MAX:2.0mi/p.W_SIZE:-mi)
        G = DC_Green(set_DC_Gk(w,p,H)...)
        G_sea += -2.0imag(tr(H.Va*G.dGR*(H.Vbc +2.0*H.Vb*G.GR*H.Vc)*G.dGR))*f(w,p.T)
    end
    return dw*G_sur, dw*G_sea
end

#calculate the Fermi sea term the DC (input is also DC) nonlinear conductivity with the Green function
#@everywhere
#= 
function Green_DC_sea_nonlinear(p::Parm, H::Hamiltonian)
    G0::Float64 = 0.0

    mi = minimum([p.W_MAX,10p.T])
    dw::Float64 = (p.W_MAX+mi)/p.W_SIZE/(2.0pi)
    for w in collect(-p.W_MAX:2pi*dw:10.0p.T)
        G = DC_Green(set_DC_Gk(w,p,H)...)
        G0 += -2.0imag(tr(H.Va*G.dGR*(H.Vbc +2.0*H.Vb*G.GR*H.Vc)*G.dGR))*f(w,p.T)
    end
    return dw*G0
end=#

#Calculate photo-voltaic effect with Parker's BI methods (under the velocity gauge)

function Length_PV_BI(p::Parm, H::Hamiltonian)
    #(iii, eei, eie, iee, eee)*(LP, CP)
    PV = zeros(Float64, 12)

    #(iii, eei, eie, iee, eee)*(bc, cb)
    PV_bc = zeros(ComplexF64, 10)
    for i in 1:p.H_size
        PV_bc[1] += H.Va[i,i]*(H.Vbc[i,i]*df(H.E[i], p.T)+H.Vb[i,i]*H.Vc[i,i]*ddf(H.E[i], p.T))/p.W_in^2
        PV_bc[2] += H.Va[i,i]*(H.Vbc[i,i]*df(H.E[i], p.T)+H.Vb[i,i]*H.Vc[i,i]*ddf(H.E[i], p.T))/p.W_in^2
        for j in 1:p.H_size
            if(j!=i)
                #Drude
                PV_bc[1] += (H.E[i]-H.E[j])*H.Va[i,i]*(H.Vb[i,j]*H.Vc[j,i]+H.Vb[i,j]*H.Vc[j,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2)*df(H.E[i], p.T)/p.W_in^2
                PV_bc[2] += (H.E[i]-H.E[j])*H.Va[i,i]*(H.Vb[i,j]*H.Vc[j,i]+H.Vb[i,j]*H.Vc[j,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2)*df(H.E[i], p.T)/p.W_in^2
                #BCD
                PV_bc[3] += ((H.Va[i,j]*H.Vb[j,i]-H.Vb[i,j]*H.Va[j,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2)*H.Vc[i,i]-(H.Va[i,j]*H.Vc[j,i]-H.Vc[i,j]*H.Va[j,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2)*H.Vb[i,i])*df(H.E[i],p.T)/p.W_in
                PV_bc[4] += -((H.Va[i,j]*H.Vb[j,i]-H.Vb[i,j]*H.Va[j,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2)*H.Vc[i,i]-(H.Va[i,j]*H.Vc[j,i]-H.Vc[i,j]*H.Va[j,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2)*H.Vb[i,i])*df(H.E[i],p.T)/p.W_in
                
                #Injection
                PV_bc[7] += 2*2p.eta*(H.Va[i,i]-H.Va[j,j])*H.Vb[i,j]*H.Vc[j,i]*(f(H.E[i],p.T)-f(H.E[j],p.T))/((p.W_in+H.E[i]-H.E[j])^2+(2p.eta)^2)
                PV_bc[8] += 2*2p.eta*(H.Va[i,i]-H.Va[j,j])*H.Vb[i,j]*H.Vc[j,i]*(f(H.E[i],p.T)-f(H.E[j],p.T))/((-p.W_in+H.E[i]-H.E[j])^2+(2p.eta)^2)

                #eie
                PV_bc[5] += (H.Va[i,j]*(H.Vb[j,j]-H.Va[i,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2) + H.Vab[i,j]*(H.E[i]-H.E[j])/((H.E[i]-H.E[j])^2+(2p.eta)^2))*H.Vc[j,i]*(f(H.E[j],p.T)-f(H.E[i],p.T))/((H.E[i]-H.E[j])^2+(2p.eta)^2)/(p.W_in+H.E[i]-H.E[j]+2.0im*p.eta) + (H.Va[i,j]*(H.Vc[j,j]-H.Va[i,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2) + H.Vca[i,j]*(H.E[i]-H.E[j])/((H.E[i]-H.E[j])^2+(2p.eta)^2))*H.Vb[j,i]*(f(H.E[j],p.T)-f(H.E[i],p.T))/((H.E[i]-H.E[j])^2+(2p.eta)^2)/(-p.W_in+H.E[i]-H.E[j]+2.0im*p.eta)
                PV_bc[6] += (H.Va[i,j]*(H.Vb[j,j]-H.Va[i,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2) + H.Vab[i,j]*(H.E[i]-H.E[j])/((H.E[i]-H.E[j])^2+(2p.eta)^2))*H.Vc[j,i]*(f(H.E[j],p.T)-f(H.E[i],p.T))/((H.E[i]-H.E[j])^2+(2p.eta)^2)/(-p.W_in+H.E[i]-H.E[j]+2.0im*p.eta) + (H.Va[i,j]*(H.Vc[j,j]-H.Va[i,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2) + H.Vca[i,j]*(H.E[i]-H.E[j])/((H.E[i]-H.E[j])^2+(2p.eta)^2))*H.Vb[j,i]*(f(H.E[j],p.T)-f(H.E[i],p.T))/((H.E[i]-H.E[j])^2+(2p.eta)^2)/(p.W_in+H.E[i]-H.E[j]+2.0im*p.eta)

                #IFS
                PV_bc[9] += 2H.Va[i,i]*(H.Vb[i,j]*H.Vc[j,i])*df(H.E[i], p.T)*p.W_in*(p.W_in -(H.E[i]-H.E[j]))/((p.W_in -(H.E[i]-H.E[j]))^2+4.0*p.eta^2)
                PV_bc[10] += 2H.Va[i,i]*(H.Vb[i,j]*H.Vc[j,i])*df(H.E[i], p.T)*p.W_in*(-p.W_in -(H.E[i]-H.E[j]))/((-p.W_in -(H.E[i]-H.E[j]))^2+4.0*p.eta^2)

                #eee
                for k in 1:p.H_size
                    PV_bc[9] += -(H.E[i]-H.E[j])*(H.E[j]-H.E[k])*(H.E[k]-H.E[i])*(H.Va[i,j]*H.Vb[j,k]-H.Vb[i,j]*H.Va[j,k])*H.Vc[k,i]/((H.E[i]-H.E[j])^2+(2p.eta)^2)/((H.E[j]-H.E[k])^2+(2p.eta)^2)/((H.E[k]-H.E[i])^2+(2p.eta)^2) * (f(H.E[i],p.T)-f(H.E[k],p.T))/(p.W_in + H.E[i]-H.E[k]+ 2.0im*p.eta)-(H.E[i]-H.E[j])*(H.E[j]-H.E[k])*(H.E[k]-H.E[i])*(H.Va[i,j]*H.Vc[j,k]-H.Vc[i,j]*H.Va[j,k])*H.Vb[k,i]/((H.E[i]-H.E[j])^2+(2p.eta)^2)/((H.E[j]-H.E[k])^2+(2p.eta)^2)/((H.E[k]-H.E[i])^2+(2p.eta)^2) * (f(H.E[i],p.T)-f(H.E[k],p.T))/(-p.W_in + H.E[i]-H.E[k]+ 2.0im*p.eta)

                    PV_bc[10] += -(H.E[i]-H.E[j])*(H.E[j]-H.E[k])*(H.E[k]-H.E[i])*(H.Va[i,j]*H.Vb[j,k]-H.Vb[i,j]*H.Va[j,k])*H.Vc[k,i]/((H.E[i]-H.E[j])^2+(2p.eta)^2)/((H.E[j]-H.E[k])^2+(2p.eta)^2)/((H.E[k]-H.E[i])^2+(2p.eta)^2) * (f(H.E[i],p.T)-f(H.E[k],p.T))/(-p.W_in + H.E[i]-H.E[k]+ 2.0im*p.eta)-(H.E[i]-H.E[j])*(H.E[j]-H.E[k])*(H.E[k]-H.E[i])*(H.Va[i,j]*H.Vc[j,k]-H.Vc[i,j]*H.Va[j,k])*H.Vb[k,i]/((H.E[i]-H.E[j])^2+(2p.eta)^2)/((H.E[j]-H.E[k])^2+(2p.eta)^2)/((H.E[k]-H.E[i])^2+(2p.eta)^2) * (f(H.E[i],p.T)-f(H.E[k],p.T))/(p.W_in + H.E[i]-H.E[k]+ 2.0im*p.eta)
                end
            end
        end
    end
    for i in 1:5
        PV[2i-1] = real(PV_bc[2i-1] + PV_bc[2i])/2
        PV[2i] = imag(PV_bc[2i-1] - PV_bc[2i])
        PV[11] += PV[2i-1]
        PV[12] += PV[2i]
    end 
    return PV
end


function Length_PV_BI2(p::Parm, H::Hamiltonian)
    #(iii, eei, eie, iee, eee)*(LP, CP)
    PV = zeros(Float64, 12)

    #(iii, eei, eie, iee, eee)*(bc, cb)
    PV_bc = zeros(ComplexF64, 10)
    for i in 1:p.H_size
        PV_bc[1] += H.Va[i,i]*(H.Vbc[i,i]*df(H.E[i], p.T)+H.Vb[i,i]*H.Vc[i,i]*ddf(H.E[i], p.T))/p.W_in^2
        PV_bc[2] += H.Va[i,i]*(H.Vbc[i,i]*df(H.E[i], p.T)+H.Vb[i,i]*H.Vc[i,i]*ddf(H.E[i], p.T))/p.W_in^2
        for j in 1:p.H_size
            if(j!=i)
                #Drude
                PV_bc[1] += (H.E[i]-H.E[j])*H.Va[i,i]*(H.Vb[i,j]*H.Vc[j,i]+H.Vb[i,j]*H.Vc[j,i])/((H.E[i]-H.E[j])^2+(p.eta)^2)*df(H.E[i], p.T)/p.W_in^2
                PV_bc[2] += (H.E[i]-H.E[j])*H.Va[i,i]*(H.Vb[i,j]*H.Vc[j,i]+H.Vb[i,j]*H.Vc[j,i])/((H.E[i]-H.E[j])^2+(p.eta)^2)*df(H.E[i], p.T)/p.W_in^2
                #BCD
                PV_bc[3] += ((H.Va[i,j]*H.Vb[j,i]-H.Vb[i,j]*H.Va[j,i])/((H.E[i]-H.E[j])^2+(p.eta)^2)*H.Vc[i,i]-(H.Va[i,j]*H.Vc[j,i]-H.Vc[i,j]*H.Va[j,i])/((H.E[i]-H.E[j])^2+(p.eta)^2)*H.Vb[i,i])*df(H.E[i],p.T)/p.W_in
                PV_bc[4] += -((H.Va[i,j]*H.Vb[j,i]-H.Vb[i,j]*H.Va[j,i])/((H.E[i]-H.E[j])^2+(p.eta)^2)*H.Vc[i,i]-(H.Va[i,j]*H.Vc[j,i]-H.Vc[i,j]*H.Va[j,i])/((H.E[i]-H.E[j])^2+(p.eta)^2)*H.Vb[i,i])*df(H.E[i],p.T)/p.W_in
                
                #Injection
                PV_bc[7] += 2p.eta*(H.Va[i,i]-H.Va[j,j])*H.Vb[i,j]*H.Vc[j,i]*(f(H.E[i],p.T)-f(H.E[j],p.T))/((p.W_in+H.E[i]-H.E[j])^2+(p.eta)^2)
                PV_bc[8] += 2p.eta*(H.Va[i,i]-H.Va[j,j])*H.Vb[i,j]*H.Vc[j,i]*(f(H.E[i],p.T)-f(H.E[j],p.T))/((-p.W_in+H.E[i]-H.E[j])^2+(p.eta)^2)

                #eie
                PV_bc[5] += (H.Va[i,j]*(H.Vb[j,j]-H.Va[i,i])/((H.E[i]-H.E[j])^2+(2p.eta)^2) + H.Vab[i,j]*(H.E[i]-H.E[j])/((H.E[i]-H.E[j])^2+(p.eta)^2))*H.Vc[j,i]*(f(H.E[j],p.T)-f(H.E[i],p.T))/((H.E[i]-H.E[j])^2+(p.eta)^2)/(p.W_in+H.E[i]-H.E[j]+1.0im*p.eta) + (H.Va[i,j]*(H.Vc[j,j]-H.Va[i,i])/((H.E[i]-H.E[j])^2+(p.eta)^2) + H.Vca[i,j]*(H.E[i]-H.E[j])/((H.E[i]-H.E[j])^2+(p.eta)^2))*H.Vb[j,i]*(f(H.E[j],p.T)-f(H.E[i],p.T))/((H.E[i]-H.E[j])^2+(p.eta)^2)/(-p.W_in+H.E[i]-H.E[j]+1.0im*p.eta)
                PV_bc[6] += (H.Va[i,j]*(H.Vb[j,j]-H.Va[i,i])/((H.E[i]-H.E[j])^2+(p.eta)^2) + H.Vab[i,j]*(H.E[i]-H.E[j])/((H.E[i]-H.E[j])^2+(p.eta)^2))*H.Vc[j,i]*(f(H.E[j],p.T)-f(H.E[i],p.T))/((H.E[i]-H.E[j])^2+(p.eta)^2)/(-p.W_in+H.E[i]-H.E[j]+1.0im*p.eta) + (H.Va[i,j]*(H.Vc[j,j]-H.Va[i,i])/((H.E[i]-H.E[j])^2+(p.eta)^2) + H.Vca[i,j]*(H.E[i]-H.E[j])/((H.E[i]-H.E[j])^2+(p.eta)^2))*H.Vb[j,i]*(f(H.E[j],p.T)-f(H.E[i],p.T))/((H.E[i]-H.E[j])^2+(p.eta)^2)/(p.W_in+H.E[i]-H.E[j]+1.0im*p.eta)

                #IFS
                PV_bc[9] += 2H.Va[i,i]*(H.Vb[i,j]*H.Vc[j,i])*df(H.E[i], p.T)*p.W_in*(p.W_in -(H.E[i]-H.E[j]))/((p.W_in -(H.E[i]-H.E[j]))^2+p.eta^2)
                PV_bc[10] += 2H.Va[i,i]*(H.Vb[i,j]*H.Vc[j,i])*df(H.E[i], p.T)*p.W_in*(-p.W_in -(H.E[i]-H.E[j]))/((-p.W_in -(H.E[i]-H.E[j]))^2+p.eta^2)

                #eee
                for k in 1:p.H_size
                    PV_bc[9] += -(H.E[i]-H.E[j])*(H.E[j]-H.E[k])*(H.E[k]-H.E[i])*(H.Va[i,j]*H.Vb[j,k]-H.Vb[i,j]*H.Va[j,k])*H.Vc[k,i]/((H.E[i]-H.E[j])^2+(p.eta)^2)/((H.E[j]-H.E[k])^2+(p.eta)^2)/((H.E[k]-H.E[i])^2+(p.eta)^2) * (f(H.E[i],p.T)-f(H.E[k],p.T))/(p.W_in + H.E[i]-H.E[k]+ 1.0im*p.eta)
                        -(H.E[i]-H.E[j])*(H.E[j]-H.E[k])*(H.E[k]-H.E[i])*(H.Va[i,j]*H.Vc[j,k]-H.Vc[i,j]*H.Va[j,k])*H.Vb[k,i]/((H.E[i]-H.E[j])^2+(p.eta)^2)/((H.E[j]-H.E[k])^2+(p.eta)^2)/((H.E[k]-H.E[i])^2+(p.eta)^2) * (f(H.E[i],p.T)-f(H.E[k],p.T))/(-p.W_in + H.E[i]-H.E[k]+ 1.0im*p.eta)

                    PV_bc[10] += -(H.E[i]-H.E[j])*(H.E[j]-H.E[k])*(H.E[k]-H.E[i])*(H.Va[i,j]*H.Vb[j,k]-H.Vb[i,j]*H.Va[j,k])*H.Vc[k,i]/((H.E[i]-H.E[j])^2+(p.eta)^2)/((H.E[j]-H.E[k])^2+(p.eta)^2)/((H.E[k]-H.E[i])^2+(p.eta)^2) * (f(H.E[i],p.T)-f(H.E[k],p.T))/(-p.W_in + H.E[i]-H.E[k]+ 1.0im*p.eta)-(H.E[i]-H.E[j])*(H.E[j]-H.E[k])*(H.E[k]-H.E[i])*(H.Va[i,j]*H.Vc[j,k]-H.Vc[i,j]*H.Va[j,k])*H.Vb[k,i]/((H.E[i]-H.E[j])^2+(p.eta)^2)/((H.E[j]-H.E[k])^2+(p.eta)^2)/((H.E[k]-H.E[i])^2+(p.eta)^2) * (f(H.E[i],p.T)-f(H.E[k],p.T))/(p.W_in + H.E[i]-H.E[k]+ 1.0im*p.eta)
                end
            end
        end
    end
    for i in 1:5
        PV[2i-1] = real(PV_bc[2i-1] + PV_bc[2i])/2
        PV[2i] = imag(PV_bc[2i-1] - PV_bc[2i])
        PV[11] += PV[2i-1]
        PV[12] += PV[2i]
    end 
    return PV
end


#Calculate photo-voltaic effect with Parker's BI methods (under the velocity gauge)
function Velocity_PV_BI(p::Parm, H::Hamiltonian)
    #(iii, eei, eie, iee, eee)*(LP, CP)
    #(Drude, BCD, Shift1, Injection, Shift2)
    PV = zeros(Float64, 12)

    #(iii, eei, eie, iee, eee)*(bc, cb)
    PV_bc = zeros(ComplexF64, 10)
    for i in 1:p.H_size
        PV_bc[1] += H.Vabc[i,i]*f(H.E[i], p.T)/2
        PV_bc[2] += H.Vabc[i,i]*f(H.E[i], p.T)/2
        #PV_bc[1] += H.Va[i,i]*H.Vbc[i,i]*f(H.E[i], p.T)/(2.0im*p.eta)/(p.W_in)^2/2
        #PV_bc[2] += H.Va[i,i]*H.Vbc[i,i]*f(H.E[i], p.T)/(2.0im*p.eta)/(p.W_in)^2/2
        #PV_bc[3] += H.Vab[i,i]*H.Vc[i,i]*f(H.E[i], p.T)/(p.W_in+1.0im*p.eta)/(p.W_in)^2
        #PV_bc[4] += H.Vac[i,i]*H.Vb[i,i]*f(H.E[i], p.T)/(-p.W_in+1.0im*p.eta)/(p.W_in)^2
        for j in 1:p.H_size
            PV1 = H.Va[i,j]*H.Vbc[j,i]*(f(H.E[i], p.T)-f(H.E[j], p.T))/(H.E[i]-H.E[j]+1.0im*p.eta)/2
            PV_bc1 = (H.Vab[i,j]*H.Vc[j,i]/(p.W_in +H.E[i]-H.E[j]+1.0im*p.eta))*(f(H.E[i], p.T)-f(H.E[j], p.T))
            PV_cb1 = (H.Vab[i,j]*H.Vc[j,i]/(-p.W_in +H.E[i]-H.E[j]+1.0im*p.eta))*(f(H.E[i], p.T)-f(H.E[j], p.T))
            PV_bc2 = (H.Vca[i,j]*H.Vb[j,i]/(-p.W_in +H.E[i]-H.E[j]+1.0im*p.eta))*(f(H.E[i], p.T)-f(H.E[j], p.T))
            PV_cb2 = (H.Vca[i,j]*H.Vb[j,i]/(p.W_in +H.E[i]-H.E[j]+1.0im*p.eta))*(f(H.E[i], p.T)-f(H.E[j], p.T))
            if(i==j)
                PV_bc[7] += PV1
                PV_bc[8] += PV1
                PV_bc[3] += PV_bc1
                PV_bc[4] += PV_cb1
                PV_bc[5] += PV_bc2
                PV_bc[6] += PV_cb2
            else
                PV_bc[9] += PV1 + PV_bc1 + PV_bc2
                PV_bc[10] += PV1 + PV_cb1 + PV_cb2
            end
            for k in 1:p.H_size
                #PVbc1 = H.Va[i,j] * H.Vb[j,k] * H.Vc[k,i]* (f(H.E[i], p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(p.W_in+H.E[i]-H.E[k]+1.0im*p.eta)) + f(H.E[k],p.T)*(1.0/((-p.W_in-H.E[j]+H.E[k]+1.0im*p.eta)*(-p.W_in-H.E[i]+H.E[k]-1.0im*p.eta))+1.0/((p.W_in-H.E[j]+H.E[k]+1.0im*p.eta)*(p.W_in-H.E[i]+H.E[k]-1.0im*p.eta))) + f(H.E[j],p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(-p.W_in+H.E[k]-H.E[j]+1.0im*p.eta)))
                PVbc1 = H.Va[i,j] * H.Vb[j,k] * H.Vc[k,i]* (f(H.E[i], p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(p.W_in+H.E[i]-H.E[k]+1.0im*p.eta)) + f(H.E[k],p.T)*(1.0/((-p.W_in-H.E[j]+H.E[k]+1.0im*p.eta)*(-p.W_in-H.E[i]+H.E[k]-1.0im*p.eta))) + f(H.E[j],p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(-p.W_in+H.E[k]-H.E[j]+1.0im*p.eta)))

                PVbc2 = H.Va[i,j] * H.Vc[j,k] * H.Vb[k,i]* (f(H.E[i], p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(-p.W_in+H.E[i]-H.E[k]+1.0im*p.eta)) + f(H.E[k],p.T)*(1.0/((p.W_in-H.E[j]+H.E[k]+1.0im*p.eta)*(p.W_in-H.E[i]+H.E[k]-1.0im*p.eta))) + f(H.E[j],p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(p.W_in+H.E[k]-H.E[j]+1.0im*p.eta)) )

                PVcb1 = H.Va[i,j] * H.Vb[j,k] * H.Vc[k,i]* (f(H.E[i], p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(-p.W_in+H.E[i]-H.E[k]+1.0im*p.eta)) + f(H.E[k],p.T)*(1.0/((p.W_in-H.E[j]+H.E[k]+1.0im*p.eta)*(p.W_in-H.E[i]+H.E[k]-1.0im*p.eta))) + f(H.E[j],p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(p.W_in+H.E[k]-H.E[j]+1.0im*p.eta)) )

                PVcb2 = H.Va[i,j] * H.Vc[j,k] * H.Vb[k,i]* (f(H.E[i], p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(p.W_in+H.E[i]-H.E[k]+1.0im*p.eta)) + f(H.E[k],p.T)*(1.0/((-p.W_in-H.E[j]+H.E[k]+1.0im*p.eta)*(-p.W_in-H.E[i]+H.E[k]-1.0im*p.eta))) + f(H.E[j],p.T)/(H.E[i]-H.E[j]+2.0im*p.eta)*(1.0/(-p.W_in+H.E[k]-H.E[j]+1.0im*p.eta)) )

                if(i==j)
                    if(j==k)
                        PV_bc[1] += PVbc1+ PVbc2
                        PV_bc[2] += PVcb1 + PVcb2
                    else
                        PV_bc[7] += PVbc1+ PVbc2
                        PV_bc[8] += PVcb1 + PVcb2
                    end
                elseif(i==k)
                    PV_bc[3] += PVbc1+ PVbc2
                    PV_bc[4] += PVcb1 + PVcb2
                elseif(j==k)
                    PV_bc[5] += PVbc1+ PVbc2
                    PV_bc[6] += PVcb1 + PVcb2
                else
                    PV_bc[9] += PVbc1+ PVbc2
                    PV_bc[10] += PVcb1 + PVcb2
                end
            end
        end
    end
    
    for i in 1:5
        PV[2i-1] = -real(PV_bc[2i-1] + PV_bc[2i])/(2*p.W_in^2)
        PV[2i] = imag(PV_bc[2i-1] - PV_bc[2i])/p.W_in^2
        PV[11] += PV[2i-1]
        PV[12] += PV[2i]
    end
    return PV
end

#before using this function, you must use the band-index representation(use HV_BI!())
function Green_PV_BI(p::Parm, H::Hamiltonian)
    G_PV = zeros(Float64, 12)
    #mi = minimum([p.W_MAX,10p.T])
    dw::Float64 = (2p.W_MAX)/p.W_SIZE/(2.0pi)
    
    for w in collect(-p.W_MAX:2pi*dw:10p.T)
        G = PV_Green(set_PV_Gk(w,p,H)...)
        PV_bc = zeros(ComplexF64, 10)
        for i in 1:p.H_size
            PV_bc[1] += H.Vabc[i,i]*G.GRmA[i,i]/2
            PV_bc[2] += H.Vabc[i,i]*G.GRmA[i,i]/2
            for j in 1:p.H_size
                pv1 = H.Va[i,j]*H.Vbc[j,i]*(G.GR[j,j]*G.GRmA[i,i]+ G.GRmA[j,j]*G.GA[i,i])/2
                PV_bc1 = (H.Vab[i,j]*H.Vc[j,i])*(G.GRp[j,j]*G.GRmA[i,i]+G.GRmA[j,j]*G.GAm[i,i])
                PV_cb1 = (H.Vab[i,j]*H.Vc[j,i])*(G.GRm[j,j]*G.GRmA[i,i]+G.GRmA[j,j]*G.GAp[i,i])
                PV_bc2 = (H.Vca[i,j]*H.Vb[j,i])*(G.GRm[j,j]*G.GRmA[i,i]+G.GRmA[j,j]*G.GAp[i,i])
                PV_cb2 = (H.Vca[i,j]*H.Vb[j,i])*(G.GRp[j,j]*G.GRmA[i,i]+G.GRmA[j,j]*G.GAm[i,i])
                if(i==j)
                    PV_bc[7] += pv1
                    PV_bc[8] += pv1
                    PV_bc[3] += PV_bc1
                    PV_bc[4] += PV_cb1
                    PV_bc[5] += PV_bc2
                    PV_bc[6] += PV_cb2
                else
                    PV_bc[9] += pv1 + PV_bc1 + PV_bc2
                    PV_bc[10] += pv1 + PV_cb1 + PV_cb2
                end
                for k in 1:p.H_size
                    PVbc1 = H.Va[i,j]*H.Vb[j,k]*H.Vc[k,i]*(G.GR[j,j]*G.GRp[k,k]*G.GRmA[i,i] + G.GRm[j,j]*G.GRmA[k,k]*G.GAm[i,i] + G.GRmA[j,j]*G.GAp[k,k]*G.GA[i,i])
                    PVcb1 = H.Va[i,j]*H.Vb[j,k]*H.Vc[k,i]*(G.GR[j,j]*G.GRm[k,k]*G.GRmA[i,i] + G.GRp[j,j]*G.GRmA[k,k]*G.GAp[i,i] + G.GRmA[j,j]*G.GAm[k,k]*G.GA[i,i])
                    PVbc2 = H.Va[i,j]*H.Vc[j,k]*H.Vb[k,i]*(G.GR[j,j]*G.GRm[k,k]*G.GRmA[i,i] + G.GRp[j,j]*G.GRmA[k,k]*G.GAp[i,i] + G.GRmA[j,j]*G.GAm[k,k]*G.GA[i,i])
                    PVcb2 = H.Va[i,j]*H.Vc[j,k]*H.Vb[k,i]*(G.GR[j,j]*G.GRp[k,k]*G.GRmA[i,i] + G.GRm[j,j]*G.GRmA[k,k]*G.GAm[i,i] + G.GRmA[j,j]*G.GAp[k,k]*G.GA[i,i])
                    if(i==j)
                        if(j==k)
                            PV_bc[1] += PVbc1+ PVbc2
                            PV_bc[2] += PVcb1 + PVcb2
                        else
                            PV_bc[7] += PVbc1+ PVbc2
                            PV_bc[8] += PVcb1 + PVcb2
                        end
                    elseif(i==k)
                        PV_bc[3] += PVbc1+ PVbc2
                        PV_bc[4] += PVcb1 + PVcb2
                    elseif(j==k)
                        PV_bc[5] += PVbc1+ PVbc2
                        PV_bc[6] += PVcb1 + PVcb2
                    else
                        PV_bc[9] += PVbc1+ PVbc2
                        PV_bc[10] += PVcb1 + PVcb2
                    end
                end
            end
        end
        for i in 1:5
            G_PV[2i-1] += dw*imag(PV_bc[2i-1] + PV_bc[2i])*f(w,p.T)/(2*p.W_in^2)
            G_PV[2i] += dw*real(PV_bc[2i-1] - PV_bc[2i])*f(w,p.T)/p.W_in^2
        end
    end
    for i in 1:5
        G_PV[11] += G_PV[2i-1]
        G_PV[12] += G_PV[2i]
    end
    return G_PV
end

#Calculate photo voltaic effect with the Green function method
#@everywhere 
function Green_PV_nonlinear(p::Parm, H::Hamiltonian)
    G0::Float64 = 0.0
    mi = minimum([p.W_MAX,10p.T])
    dw::Float64 = (p.W_MAX+mi)/p.W_SIZE/(2.0pi)
    for w in collect(-p.W_MAX:2pi*dw:10.0p.T)
        G = PV_Green(set_PV_Gk(w,p,H)...)
        G0 += imag(tr(H.Va*G.GR*(H.Vb*(G.GRp + G.GRm)*H.Vc + H.Vbc)*G.GRmA)*f(w,p.T))
        G0 += imag(tr(H.Va * G.GRm * H.Vb * G.GRmA * H.Vc*G.GAm)*f(w,p.T))
        G0 += imag(tr(H.Va * G.GRp * H.Vb * G.GRmA * H.Vc*G.GAp)*f(w,p.T))
        G0 += imag(tr(H.Va*G.GRmA*(H.Vb*(G.GAp + G.GAm)*H.Vc + H.Vbc)*G.GA)*f(w,p.T))
        G0 += imag(tr(H.Vab*(G.GRp + G.GRm)*H.Vc*G.GRmA)*f(w,p.T))
        G0 += imag(tr(H.Vab*G.GRmA*H.Vc*(G.GAm + G.GAp))*f(w,p.T))
        G0 += imag(tr(H.Vca*(G.GRp + G.GRm)*H.Vb*G.GRmA)*f(w,p.T))
        G0 += imag(tr(H.Vca*G.GRmA*H.Vb*(G.GAm + G.GAp))*f(w,p.T))
        G0 += imag(tr(H.Vabc*G.GRmA)*f(w,p.T))
    end
    return dw*G0/p.W_in^2
end

#linear polarized
#@everywhere 
function Green_SHG_nonlinear(p::Parm, H::Hamiltonian)
    G0::Float64 = 0.0
    mi = minimum([p.W_MAX,10p.T])
    dw::Float64 = (p.W_MAX+mi)/p.W_SIZE/(2.0pi)
    for w in collect(-p.W_MAX:2pi*dw:10.0p.T)
        G = PV_Green(set_PV_Gk(w,p,H)...)
        G0 += imag(tr(H.Va*G.GRpp*(H.Vb*G.GRp*H.Vc + H.Vc*G.GRp*H.Vb  + H.Vbc)*G.GRmA)*f(w,p.T))
        G0 += imag(tr(H.Va * G.GRp * (H.Vb * G.GRmA * H.Vc + H.Vc * G.GRmA * H.Vb)*G.GAm)*f(w,p.T))
        G0 += imag(tr(H.Va*G.GRmA*(H.Vb*G.GAm*H.Vc + H.Vc*G.GAm*H.Vb + H.Vbc)*G.GAmm)*f(w,p.T))
        G0 += imag(tr(H.Va*G.GRmm*(H.Vb*G.GRm*H.Vc + H.Vc*G.GRm*H.Vb  + H.Vbc)*G.GRmA)*f(w,p.T))
        G0 += imag(tr(H.Va * G.GRm * (H.Vb * G.GRmA * H.Vc + H.Vc * G.GRmA * H.Vb)*G.GAp)*f(w,p.T))
        G0 += imag(tr(H.Va*G.GRmA*(H.Vb*G.GAp*H.Vc + H.Vc*G.GAp*H.Vb + H.Vbc)*G.GApp)*f(w,p.T))

        G0 += imag(tr(H.Vab*(G.GRp + G.GRm)*H.Vc*G.GRmA)*f(w,p.T))
        G0 += imag(tr(H.Vab*G.GRmA*H.Vc*(G.GAm + G.GAp))*f(w,p.T))
        G0 += imag(tr(H.Vca*(G.GRp + G.GRm)*H.Vb*G.GRmA)*f(w,p.T))
        G0 += imag(tr(H.Vca*G.GRmA*H.Vb*(G.GAm + G.GAp))*f(w,p.T))

        G0 += imag(tr(H.Vabc*G.GRmA)*f(w,p.T))
    end
    return dw*G0/p.W_in^2
end

#circular polarized
#@everywhere 
function Green_SHG_nonlinear_CP(p::Parm, H::Hamiltonian)
    G0::Float64 = 0.0
    mi = minimum([p.W_MAX,10p.T])
    dw::Float64 = (p.W_MAX+mi)/p.W_SIZE/(2.0pi)
    for w in collect(-p.W_MAX:2pi*dw:10.0p.T)
        G = PV_Green(set_PV_Gk(w,p,H)...)
        G0 += real(tr(H.Va*G.GRpp*(H.Vb*G.GRp*H.Vc - H.Vc*G.GRp*H.Vb)*G.GRmA)*f(w,p.T))
        G0 += real(tr(H.Va * G.GRp * (H.Vb * G.GRmA * H.Vc - H.Vc * G.GRmA * H.Vb)*G.GAm)*f(w,p.T))
        G0 += real(tr(H.Va*G.GRmA*(H.Vb*G.GAm*H.Vc - H.Vc*G.GAm*H.Vb)*G.GAmm)*f(w,p.T))

        G0 += real(tr(H.Va*G.GRmm*(-H.Vb*G.GRm*H.Vc + H.Vc*G.GRm*H.Vb)*G.GRmA)*f(w,p.T))
        G0 += real(tr(H.Va * G.GRm * (-H.Vb * G.GRmA * H.Vc + H.Vc * G.GRmA * H.Vb)*G.GAp)*f(w,p.T))
        G0 += real(tr(H.Va*G.GRmA*(-H.Vb*G.GAp*H.Vc + H.Vc*G.GAp*H.Vb)*G.GApp)*f(w,p.T))

        G0 += real(tr(H.Vab*(G.GRp - G.GRm)*H.Vc*G.GRmA)*f(w,p.T))
        G0 += real(tr(H.Vab*G.GRmA*H.Vc*(G.GAm - G.GAp))*f(w,p.T))
        G0 += real(tr(H.Vca*(G.GRp - G.GRm)*H.Vb*G.GRmA)*f(w,p.T))
        G0 += real(tr(H.Vca*G.GRmA*H.Vb*(G.GAm - G.GAp))*f(w,p.T))
    end
    return dw*G0/p.W_in^2
end