#include("2D_TMD_parm.jl")
#include("model_2D_IB.jl")
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
function set_DC_Gk(id_w::Int, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    w = p.w_mesh[id_w]
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.Σw[id_w]

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    GRmA::Array{ComplexF64,2} = GR - GA

    dGR::Array{ComplexF64,2} = - GR * p.γ1[id_w] * GR
    dGA::Array{ComplexF64,2} = dGR'
    ddGR::Array{ComplexF64,2} = -2.0 * dGR * p.γ1[id_w] * GR - GR * p.γ2[id_w] * GR
    
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
function set_PV_Gk(id_w::Int, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    w = p.w_mesh[id_w]
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{ComplexF64}(w*I,2,2) + p.eta*Matrix{ComplexF64}(1.0im*I,2,2) - p.Σw[id_w]


    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    GRmA::Array{ComplexF64,2} = GR - GA

    id_Ω::Int = Int(floor(p.Ω/p.dw+0.001))

    if(id_w+id_Ω > p.w_size)
        GRp = zeros(ComplexF64, 2, 2)
        GAp = zeros(ComplexF64, 2, 2)
    else
        GRp0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w+p.Ω)*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.Σw[id_w+id_Ω]

        GRp::Array{ComplexF64,2} = inv(GRp0)
        GAp::Array{ComplexF64,2} = GRp'
    end

    if(id_w-id_Ω < 1)
        GRm = zeros(ComplexF64, 2, 2)
        GAm = zeros(ComplexF64, 2, 2)
    else
        GRm0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w-p.Ω)*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.Σw[id_w-id_Ω]

        GRm::Array{ComplexF64,2} = inv(GRm0)
        GAm::Array{ComplexF64,2} = GRm'
    end
    
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
function set_SHG_Gk(id_w::Int, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    w = p.w_mesh[id_w]
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.Σw[id_w]

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    GRmA::Array{ComplexF64,2} = GR - GA

    #If you use the self-energy, you should change this part.
    id_Ω::Int = Int(floor(p.Ω/p.dw+0.001))

    if(id_w+id_Ω > p.w_size)
        GRp = zeros(ComplexF64, 2, 2)
        GAp = zeros(ComplexF64, 2, 2)
    else
        GRp0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w+p.Ω)*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.Σw[id_w+id_Ω]

        GRp::Array{ComplexF64,2} = inv(GRp0)
        GAp::Array{ComplexF64,2} = GRp'
    end

    if(id_w-id_Ω < 1)
        GRm = zeros(ComplexF64, 2, 2)
        GAm = zeros(ComplexF64, 2, 2)
    else
        GRm0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w-p.Ω)*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.Σw[id_w-id_Ω]

        GRm::Array{ComplexF64,2} = inv(GRm0)
        GAm::Array{ComplexF64,2} = GRm'
    end

    if(id_w+2id_Ω > p.w_size)
        GRpp = zeros(ComplexF64, 2, 2)
        GApp = zeros(ComplexF64, 2, 2)
    else
        GRpp0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w+2p.Ω)*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.Σw[id_w+2id_Ω]

        GRpp::Array{ComplexF64,2} = inv(GRpp0)
        GApp::Array{ComplexF64,2} = GRpp'
    end

    if(id_w-id_Ω < 1)
        GRmm = zeros(ComplexF64, 2, 2)
        GAmm = zeros(ComplexF64, 2, 2)
    else
        GRmm0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}((w-2p.Ω)*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.Σw[id_w-2id_Ω]

        GRmm::Array{ComplexF64,2} = inv(GRmm0)
        GAmm::Array{ComplexF64,2} = GRmm'
    end
    
    return GR, GA, GRmA, GRp, GAp, GRm, GAm, GRpp, GApp, GRmm, GAmm
end

#change the basis to the band-index representation
#@everywhere 
function HV_BI!(H::Hamiltonian)

    H.E, BI::Array{ComplexF64,2} = eigen(H.Hk)
    H.Hk = [H.E[1] 0.0; 0.0 H.E[2]]
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

#@everywhere 
f(e::ComplexF64,T::Float64) = 1.0/(1.0+exp(e/T))
#@everywhere 
df(e::ComplexF64,T::Float64) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T

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


    #mi = minimum([p.W_MAX,10p.T])
    #dw::Float64 = 2mi/p.W_SIZE
    for id_w in 1:p.w_size#collect(-mi:dw:mi)
        G = DC_Green(set_DC_Gk(id_w,p,H)...)
        w = p.w_mesh[id_w]
        if(w < 10p.T)
            asym += -real(tr(H.Va*G.dGR*H.Vb*G.GRmA))*f(w,p.T)
            if(w > -10p.T)
                sym += real(tr(H.Va*G.GR*H.Vb*G.GRmA))*df(w,p.T)
            end
        end
    end
    return p.dw*sym/(2pi), p.dw*asym/(2pi)
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
    #mi = minimum([p.W_MAX,10p.T])
    #dw::Float64 = mi/p.W_SIZE/pi
    for id_w in 1:p.w_size#collect(-mi:2.0mi/p.W_SIZE:mi)
        G = DC_Green(set_DC_Gk(id_w,p,H)...)
        w = p.w_mesh[id_w]
        if(w<10p.T)
            if(w > -10p.T)
                G_sur += 2.0imag(tr(H.Va*G.dGR*(2.0*H.Vb*G.GR*H.Vc + H.Vbc)*G.GRmA)*df(w,p.T))
                G_sea += -2.0imag(tr(H.Va*G.dGR*(H.Vbc +2.0*H.Vb*G.GR*H.Vc)*G.dGR))*f(w,p.T)
            end
            G_sea += -2.0imag(tr(H.Va*G.dGR*(H.Vbc +2.0*H.Vb*G.GR*H.Vc)*G.dGR))*f(w,p.T)
        end
    end
    return p.dw*G_sur, p.dw*G_sea
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

#Calculate photo voltaic effect with the Green function method
#@everywhere 
function Green_PV_nonlinear(p::Parm, H::Hamiltonian)
    G0::Float64 = 0.0
    #mi = minimum([p.W_MAX,10p.T])
    #dw::Float64 = (p.W_MAX+mi)/p.W_SIZE/(2.0pi)
    for id_w in 1:p.w_size#collect(-p.W_MAX:2pi*dw:10.0p.T)
        G = PV_Green(set_PV_Gk(id_w,p,H)...)
        w = p.w_mesh[id_w]
        if(w < 10p.T)
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
    end
    return p.dw*G0/p.Ω^2
end

#linear polarized
#@everywhere 
function Green_SHG_nonlinear(p::Parm, H::Hamiltonian)
    G0::Float64 = 0.0
    for id_w in 1:p.w_size#collect(-p.W_MAX:2pi*dw:10.0p.T)
        G = PV_Green(set_PV_Gk(id_w,p,H)...)
        w = p.w_mesh[id_w]
        if(w < 10p.T)
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
    end
    return p.dw*G0/p.Ω^2
end

#circular polarized
#@everywhere 
function Green_SHG_nonlinear_CP(p::Parm, H::Hamiltonian)
    G0::Float64 = 0.0
    for id_w in 1:p.w_size#collect(-p.W_MAX:2pi*dw:10.0p.T)
        G = PV_Green(set_PV_Gk(id_w,p,H)...)
        w = p.w_mesh[id_w]
        if(w < 10p.T)
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
    end
    return p.dw*G0/p.Ω^2
end

#=
function main(arg::Array{String,1})
    p = Parm(set_parm_Wdep(arg, 0.2)...)
    println(p.Σw[1])
    kk = get_kk(p.K_SIZE)
    H = Hamiltonian(HandV_fd(kk[1],p)...)
    println(H.Hk)
    G = PV_Green(set_PV_Gk(10, p, H)...)
    println(G.GR)
end

@time main(ARGS)=#