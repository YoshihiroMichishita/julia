include("model_2D_IB.jl")

using SparseIR, Plots
#using OMEinsum
#using FastGaussQuadrature
import SparseIR: valueim, value
import SparseIR: fit
#using LinearAlgebra


struct IR_params
    U::Float64
    beta::Float64
    bw::Float64
    basis::FiniteTempBasis
    #{Fermionic, LogisticKernel, Float64, Float64}
    
    smpl_matsu::MatsubaraSampling64F
    smpl_tau::TauSampling64

    n_matsu::Int
    n_tau::Int

    smpl_wn::Vector{ComplexF64}
end

function set_IR(U::Float64, beta::Float64, bw::Float64)
    basis = FiniteTempBasis(Fermionic(), beta, bw, 1e-10)

    smpl_matsu = MatsubaraSampling(basis)
    n_matsu = size(smpl_matsu.sampling_points)[1]

    smpl_tau = TauSampling(basis)
    n_tau = size(smpl_tau.sampling_points)[1]

    smpl_wn = zeros(ComplexF64, n_matsu)
    for w in 1:n_matsu
        smpl_wn[w] = valueim(smpl_matsu.sampling_points[w], beta)
    end

    return U, beta, bw, basis, smpl_matsu, smpl_tau, n_matsu, n_tau, smpl_wn
end

mutable struct Green_Sigma
    g0_ir::Vector{Matrix{ComplexF64}}
    g0_ir_vec::Matrix{Float64}
    g0_tau::Vector{Matrix{ComplexF64}}
    g0_matsu::Vector{Matrix{ComplexF64}}

    g_ir::Vector{Matrix{ComplexF64}}
    g_ir_vec::Matrix{Float64}
    g_matsu::Vector{Matrix{ComplexF64}}

    sigma_ir::Vector{Matrix{ComplexF64}}
    sigma_tau::Vector{Matrix{ComplexF64}}
    sigma_matsu::Vector{Matrix{ComplexF64}}

    n_ir::Int
end

function init_zero_g(ir::IR_params)
    
    g0_ir::Vector{Matrix{ComplexF64}} = []
    g0_ir_vec = zeros(Float64, 43, 4)
    g0_tau::Vector{Matrix{ComplexF64}} = []
    g0_matsu::Vector{Matrix{ComplexF64}} = []

    g_ir::Vector{Matrix{ComplexF64}} = []
    g_ir_vec = zeros(Float64, 43, 4)
    g_matsu::Vector{Matrix{ComplexF64}} = []

    sigma_ir::Vector{Matrix{ComplexF64}} = []
    sigma_tau::Vector{Matrix{ComplexF64}} = []
    sigma_matsu::Vector{Matrix{ComplexF64}} = []

    n_ir::Int = 0

    return g0_ir, g0_ir_vec, g0_tau, g0_matsu, g_ir, g_ir_vec, g_matsu, sigma_ir, sigma_tau, sigma_matsu, n_ir
end

function get_G0mlocal!(p::Parm, k_BZ::Vector{Vector{Float64}}, sw::Int,ir::IR_params, g::Green_Sigma)
    if(sw == 1)
        for wn in 1:ir.n_matsu
            gw_l = zeros(ComplexF64, 2, 2)
            gl = zeros(ComplexF64, 2, 2)
            for i in 1:length(k_BZ)
                e = set_H(k_BZ[i],p) - p.mu*Matrix{Complex{Float64}}(I,2,2)
                #gk = 1.0/(w - e + p.eta*1.0im*sign(imag(w)))
                gk = inv(ir.smpl_wn[wn]*Matrix{Complex{Float64}}(I,2,2) - e)
                gw_l += p.dk2 * gk
            end
            push!(g.g0_matsu, gw_l)
            push!(g.g_matsu, gl)
        end
    else
        for wn in 1:ir.n_matsu
            gw_l = zeros(ComplexF64, 2, 2)
            gl = zeros(ComplexF64, 2, 2)
            for i in 1:length(k_BZ)
                e = set_H(k_BZ[i],p) - p.mu*Matrix{Complex{Float64}}(I,2,2)
                #gk = 1.0/(w - e + p.eta*1.0im*sign(imag(w)))
                gk = inv(ir.smpl_wn[wn]*Matrix{Complex{Float64}}(I,2,2) - e - g.sigma_matsu[wn])
                gl += p.dk2 * gk
            end
            gw_l = inv(inv(gl) + g.sigma_matsu[wn])
            g.g0_matsu[wn] = gw_l
            g.g_matsu[wn] = gl
        end
    end
    return nothing
end

function MatsuToTau!(ir::IR_params, g::Green_Sigma)
    g.g0_ir = fit(ir.smpl_matsu, g.g0_matsu, dim=1)
    g.g_ir = fit(ir.smpl_matsu, g.g_matsu, dim=1)
    g.n_ir = size(g.g0_ir)[1]
    g.g0_tau = evaluate(ir.smpl_tau, g.g0_ir, dim=1)
end

function calc_sigma!(sw::Int, ir::IR_params, g::Green_Sigma)
    for tau in 1:ir.n_tau
        if(sw == 1)
            test = zeros(ComplexF64, 2, 2)
            for i in 1:2, j in 1:2 
                test[i,j] = ir.U^2 * (g.g0_tau[tau])[i,j] * (g.g0_tau[tau])[3-i,3-j] * (g.g0_tau[end+1-tau])[3-i,3-j]
                #test[i,j] = ir.U^2 * (g.g0_tau[tau])[i,j] * (g.g0_tau[tau])[i,j] * (g.g0_tau[end+1-tau])[i,j]
            end
            push!(g.sigma_tau, test)
        else
            for i in 1:2, j in 1:2 
                (g.sigma_tau[tau])[i,j] = ir.U^2 * (g.g0_tau[tau])[i,j] * (g.g0_tau[tau])[3-i,3-j] * (g.g0_tau[end+1-tau])[3-i,3-j]
                #(g.sigma_tau[tau])[i,j] = ir.U^2 * (g.g0_tau[tau])[i,j] * (g.g0_tau[tau])[i,j] * (g.g0_tau[end+1-tau])[i,j]
            end
        end
    end
    return nothing
end

function TauToMatsu!(sw::Int, ir::IR_params, g::Green_Sigma, γ::Float64)
    ir_new = fit(ir.smpl_tau, g.sigma_tau)
    diff0 =0.0
    sum0 = 0.0
    if(sw == 1)
        for i in 1:g.n_ir
            diff0 += sum(abs.(ir_new[i]))
            push!(g.sigma_ir, zeros(ComplexF64, 2, 2))
        end
        sum0 = 1.0
    else
        for i in 1:g.n_ir
            diff0 += sum(abs.(ir_new[i] .- g.sigma_ir[i]))
            sum0 += sum(abs.(g.sigma_ir[i]))
        end
    end
    diff = diff0/sum0
    g.sigma_ir = (1.0-γ) .* g.sigma_ir .+ γ .* ir_new
    g.sigma_matsu = evaluate(ir.smpl_matsu, g.sigma_ir, dim=1)

    return diff
end

function update_g!(p::Parm, k_BZ::Vector{Vector{Float64}},sw::Int, ir::IR_params, g::Green_Sigma, γ::Float64)
    get_G0mlocal!(p, k_BZ, sw, ir, g)
    MatsuToTau!(ir, g)
    calc_sigma!(sw, ir, g)
    # .+ ir.U .* g.g0_tau
    diff = TauToMatsu!(sw, ir, g, γ)
    return diff
end

function MtoV!(g::Green_Sigma)
    g.g0_ir_vec = zeros(Float64, g.n_ir, 4)
    g.g_ir_vec = zeros(Float64, g.n_ir, 4)
    for it in 1:g.n_ir
        for mm in 1:4
            g.g0_ir_vec[it,mm] = real(tr(sigma[mm]*g.g0_ir[it]))/2
            g.g_ir_vec[it,mm] = real(tr(sigma[mm]*g.g_ir[it]))/2
        end
    end
end

function MtoV(MM::Vector{Matrix{ComplexF64}})
    l = size(MM)[1]
    VV = zeros(ComplexF64, l, 4)
    for it in 1:l
        for mm in 1:4
            VV[it,mm] = tr(sigma[mm]*MM[it])/2
        end
    end
    return VV
end

using Flux

function F_rho0(ir::IR_params, g::Green_Sigma, rho_ls, λ)
    vec = g.g0_ir_vec - (ir.basis.s .* rho_ls)
    
    return f = 0.5*sum((vec.^2)) + λ*sum(abs.(rho_ls))
end

function fit_rho0w(ir::IR_params, g::Green_Sigma, l_num::Int, batch_num::Int, tr_w, it_MAX::Int)
    sn = range(-12.0, 0.0, length=l_num)
    lam_test = 10 .^ (sn)
    opt = ADAM()
    s_rho_l::Vector{Matrix{Float64}} = []
    #rand(Float64, l_num, g.n_ir)
    s_F_rho = 1000.0 * ones(Float64, l_num)
    for ll in 1:l_num
        lam = lam_test[ll]
        for b in 1:batch_num
            rho_ll = rand(Float64, g.n_ir, 4)
            F_old = 1000.0
            F_new = 1000.0
            for i in 1:it_MAX
                grads = Flux.gradient(Flux.params(rho_ll)) do
                    F_rho0(ir, g, rho_ll, lam)
                end
                Flux.Optimise.update!(opt, Flux.params(rho_ll), grads)
                F_old = F_new
                F_new = F_rho0(ir, g, rho_ll, lam)
                if(abs(F_old-F_new)/abs(F_old)<1e-6)
                    break
                end
            end
            if(s_F_rho[ll] > F_new)
                if(b==1)
                    push!(s_rho_l, rho_ll)
                else
                    s_rho_l[ll] = rho_ll
                end
                s_F_rho[ll] = F_new
            end 
        end
    end
    b = (log(s_F_rho[end])-log(s_F_rho[1]))/(log(lam_test[end])-log(lam_test[1]))
    a = s_F_rho[1]*lam_test[1]^(-b)
    ev = a .* lam_test .^ b ./ s_F_rho

    p1 = plot(lam_test, ev, xaxis=:log, yaxis=:log, marker=:circle)
    savefig(p1,"./lambda_opt_G0.png")

    it = findmax(ev)[2]
    println("it:$it")
    max1 = s_F_rho[it]
    println("it:$it,  s_F:$max1")
    println(size(s_rho_l[it]))
    rho_omega = -tr_w * s_rho_l[it]
    return rho_omega
end

function F_rho(ir::IR_params, g::Green_Sigma, rho_ls, λ)
    vec = g.g_ir_vec - (ir.basis.s .* rho_ls)
    
    return f = 0.5*sum((vec.^2)) + λ*sum(abs.(rho_ls))
end

function fit_rhow(ir::IR_params, g::Green_Sigma, l_num::Int, batch_num::Int, tr_w, it_MAX::Int)
    sn = range(-12.0, 0.0, length=l_num)
    lam_test = 10 .^ (sn)
    opt = ADAM()
    s_rho_l::Vector{Matrix{Float64}} = []
    #rand(Float64, l_num, g.n_ir)
    s_F_rho = 1000.0 * ones(Float64, l_num)
    for ll in 1:l_num
        lam = lam_test[ll]
        count::Int = 0
        for b in 1:batch_num
            rho_ll = rand(Float64, g.n_ir, 4)
            
            #rho_ll = rand(Float64, g.n_ir)
            F_old = 1000.0
            F_new = 1000.0
            for i in 1:it_MAX
                grads = Flux.gradient(Flux.params(rho_ll)) do
                    F_rho(ir, g, rho_ll, lam)
                end
                Flux.Optimise.update!(opt, Flux.params(rho_ll), grads)
                F_old = F_new
                F_new = F_rho(ir, g, rho_ll, lam)
                if(abs(F_old-F_new)/abs(F_old)<1e-6)
                    break
                end
                if(i==it_MAX)
                    count += 1
                end
            end
            if(s_F_rho[ll] > F_new)
                if(b==1)
                    push!(s_rho_l, rho_ll)
                else
                    s_rho_l[ll] = rho_ll
                end
                s_F_rho[ll] = F_new
            end 
        end
        println(count)
    end
    b = (log(s_F_rho[end])-log(s_F_rho[1]))/(log(lam_test[end])-log(lam_test[1]))
    a = s_F_rho[1]*lam_test[1]^(-b)
    ev = a .* lam_test .^ b ./ s_F_rho

    p1 = plot(lam_test, ev, xaxis=:log, yaxis=:log, marker=:circle)
    #savefig(p1,"./lambda_opt_G.png")

    it = findmax(ev)[2]
    max1 = s_F_rho[it]
    println("it:$it,  s_F:$max1")
    rho_omega = -(tr_w * s_rho_l[it])
    return rho_omega
end

function reshape(rho::Vector{Float64}, cutoff::Float64)
    v_it::Vector{Int} = []
    rho_rep = rho
    sw = true
    for w in 1:length(rho_rep)
        if(rho_rep[w]<0)
            if(sw)
                for it in v_it
                    rho_rep[it] = 0.002          
                end
                empty!(v_it)
                sw = false
            elseif(w<length(rho_rep) && rho_rep[w+1]>0.0)
                sw = true
            end
            rho_rep[w] = 0.002

        elseif(rho_rep[w]<cutoff)
            if(sw)
                push!(v_it,w)
            #elseif(w<length(rho_rep) && rho_rep[w+1]>rho_rep[w])
            #    sw = true
            #    push!(v_it,w)
            end
        elseif(rho_rep[w]>=cutoff && sw)
            empty!(v_it)
            sw = false
        end
        if(w == length(rho_rep))
            if(sw)
                for it in v_it
                    rho_rep[it] = 0.002             
                end
            end
        end
    end
    return rho_rep
end

f(beta::Float64, w::Float64) = 1.0/(1.0+exp(beta*w))
function renorm_rho(beta::Float64, w::Vector{Float64}, rho::Vector{Float64})
    n = 0.0
    dw = w[2]-w[1]
    for i in 1:length(w)
        n += dw * rho[i] * f(beta, w[i])
    end
    #rho1 = (0.5/n) * rho
    #return rho1
    return n
end


function KK_GR(w::Vector{Float64}, rho::Vector{Matrix{ComplexF64}})
    #GR_ = zeros(ComplexF64, length(w))
    GR_::Vector{Matrix{ComplexF64}} = []
    dw = w[2]-w[1]
    for w_re in 1:length(w)
        re = zeros(ComplexF64, 2, 2)
        for w_im in 1:length(w)
            if(w_im != w_re)
                re += dw * rho[w_im] / (w[w_re] - w[w_im])
            end 
        end
        ggg = re - 1.0im * rho[w_re] * pi
        push!(GR_, ggg)
    end
    return GR_
end

function VtoM(Vec::Matrix{Float64})
    MM::Vector{Matrix{ComplexF64}} = []
    for it in 1:size(Vec)[1]
        Mat = Vec[it,:]' * sigma
        push!(MM, Mat)
    end
    return MM
end

function reshape_g(w::Vector{Float64}, beta::Float64, Aw::Matrix{Float64})
    g11 = reshape(Aw[:,1]+Aw[:,4],0.3)
    g22 = reshape(Aw[:,1]-Aw[:,4],0.3)

    c_Aw = Aw
    c_Aw[:,1] = (g11 + g22)/2
    c_Aw[:,4] = (g11 - g22)/2

    n = renorm_rho(beta, w, Aw[:,1])
    Aw = (0.5/n) * Aw

    iG = VtoM(Aw)
    GR = KK_GR(w, iG)
    return GR
end

ENV["GKSwstype"]="nul"
Plots.scalefontsizes(1.4)



using DataFrames
using CSV

function main(arg::Vector{String})
    p = Parm(set_parm(arg)...)
    println(p)
    ir = IR_params(set_IR(parse(Float64,arg[7]),parse(Float64,arg[8]),parse(Float64,arg[9]))...)
    lamda_num = parse(Int,arg[10])
    batch_num = parse(Int,arg[11])
    w_num = parse(Int,arg[12])
    g = Green_Sigma(init_zero_g(ir)...)
    w_mesh = collect(-ir.bw:2ir.bw/(w_num-1):ir.bw)

    tr_w = transpose(ir.basis.v(w_mesh))

    kk = get_kk(p.K_SIZE)
    Disp_HSL(p)

    for it in 1:1000
        L1 = update_g!(p,kk,it,ir,g, 0.2)
        if(L1<1e-7)
            println(it)
            break
        end
    end

    MtoV!(g)
    g0 = fit_rho0w(ir, g, lamda_num, batch_num, tr_w, 20000)
    gi = fit_rhow(ir, g, lamda_num, batch_num, tr_w, 20000)

    

    GR_int = reshape_g(w_mesh, ir.beta, gi)
    GR_0 = reshape_g(w_mesh, ir.beta, g0)
    
    #sigma_w = zeros(ComplexF64, w_num)
    sigma_w::Vector{Matrix{ComplexF64}} = []
    for ww in 1:w_num
        if(abs(tr(GR_0[ww]))<1e-4 || abs(tr(GR_int[ww]))<1e-4)
            ss = zeros(ComplexF64, 2, 2)
            push!(sigma_w, ss)
        else
            ss = inv(GR_0[ww]) - inv(GR_int[ww]) - p.mu*Matrix{Complex{Float64}}(I,2,2)
            push!(sigma_w, ss)
        end
    end

    GR_int_vec = MtoV(GR_int)
    SR_w = MtoV(sigma_w)
    save_data_g = DataFrame(w=w_mesh,img1=imag.(GR_int_vec[:,1]),reg1=real.(GR_int_vec[:,1]),img2=imag.(GR_int_vec[:,2]),reg2=real.(GR_int_vec[:,2]),img3=imag.(GR_int_vec[:,3]),reg3=real.(GR_int_vec[:,3]),img4=imag.(GR_int_vec[:,4]),reg4=real.(GR_int_vec[:,4]))


    save_data_s = DataFrame(w=w_mesh, ims1=imag.(SR_w[:,1]),res1=real.(SR_w[:,1]), ims2=imag.(SR_w[:,2]), res2=real.(SR_w[:,2]), ims3=imag.(SR_w[:,3]),res3=real.(SR_w[:,3]), ims4=imag.(SR_w[:,4]),res4=real.(SR_w[:,4]))

    #pg0 = plot(w_mesh, imag.(GR_0), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    #pg0 = plot!(w_mesh, real.(GR_0), linewidth=3.0)
    #savefig(pg0,"./LDOS_free.png")

    pg = plot(w_mesh, imag.(GR_int_vec[:,1]+ GR_int_vec[:,4]), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    pg = plot!(w_mesh, imag.(GR_int_vec[:,1]- GR_int_vec[:,4]), linewidth=3.0)
    savefig(pg,"./LDOS.png")

    ps = plot(w_mesh, imag.(SR_w[:,1]+ SR_w[:,4]), linewidth=3.0, xlabel="ω", ylabel="Σ(ω)", title="self-energy")
    ps = plot!(w_mesh, imag.(SR_w[:,1]- SR_w[:,4]), linewidth=3.0)
    ps = plot!(w_mesh, real.(SR_w[:,1]+ SR_w[:,4]), linewidth=3.0)
    ps = plot!(w_mesh, real.(SR_w[:,1]- SR_w[:,4]), linewidth=3.0)
    savefig(ps,"./self-energy.png")
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./GF_U$(ir.U)_b$(ir.beta).csv", save_data_g)
    CSV.write("./Sigma_U$(ir.U)_b$(ir.beta).csv", save_data_s)

    #Spectral_HSL(p, w_mesh, sigma_w)

end
#=
function main_3D(arg::Vector{String})
    p = Parm(set_parm(arg)...)
    println(p)
    ir = IR_params(set_IR(parse(Float64,arg[6]),parse(Float64,arg[7]),parse(Float64,arg[8]))...)
    lamda_num = parse(Int,arg[9])
    batch_num = parse(Int,arg[10])
    w_num = parse(Int,arg[11])
    g = Green_Sigma(init_zero_g(ir)...)
    w_mesh = collect(-ir.bw:2ir.bw/(w_num-1):ir.bw)

    #kk = get_kk(p.K_SIZE)
    #Disp_HSL(p)

    for it in 1:600
        L1 = update_g_3D!(p,it,ir,g,0.2)
        if(L1<1e-8)
            println(it)
            break
        end
    end

    g0 = -1.0im .* fit_rho0w(ir, g, lamda_num, batch_num, w_mesh)
    g = -1.0im .* fit_rhow(ir, g, lamda_num, batch_num, w_mesh)
    sigma_w = 1.0 ./ g0 .- 1.0 ./ g .- p.mu
    save_data_g = DataFrame(w=w_mesh,img=imag.(g),reg=real.(g))
    save_data_s = DataFrame(w=w_mesh,ims=imag.(sigma_w),res=real.(sigma_w))

    pg0 = plot(w_mesh, imag.(g0), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    pg0 = plot!(w_mesh, real.(g0), linewidth=3.0)
    savefig(pg0,"./LDOS_free.png")

    pg = plot(w_mesh, imag.(g), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    pg = plot!(w_mesh, real.(g), linewidth=3.0)
    savefig(pg,"./LDOS.png")

    ps = plot(w_mesh, imag.(sigma_w), linewidth=3.0, xlabel="ω", ylabel="Σ(ω)", title="self-energy")
    ps = plot!(w_mesh, real.(sigma_w), linewidth=3.0)
    savefig(ps,"./self-energy.png")
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./GF_U$(ir.U)_b$(ir.beta).csv", save_data_g)
    CSV.write("./Sigma_U$(ir.U)_b$(ir.beta).csv", save_data_s)

    #Spectral_HSL(p, w_mesh, sigma_w)

end
=#

@time main(ARGS)