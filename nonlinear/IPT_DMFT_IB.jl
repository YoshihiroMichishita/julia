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
    g0_tau::Vector{Matrix{ComplexF64}}
    g0_matsu::Vector{Matrix{ComplexF64}}

    g_ir::Vector{Matrix{ComplexF64}}
    g_matsu::Vector{Matrix{ComplexF64}}

    sigma_ir::Vector{Matrix{ComplexF64}}
    sigma_tau::Vector{Matrix{ComplexF64}}
    sigma_matsu::Vector{Matrix{ComplexF64}}

    n_ir::Int
end

function init_zero_g(ir::IR_params)
    
    g0_ir::Vector{Matrix{ComplexF64}} = []
    g0_tau::Vector{Matrix{ComplexF64}} = []
    g0_matsu::Vector{Matrix{ComplexF64}} = []

    g_ir::Vector{Matrix{ComplexF64}} = []
    g_matsu::Vector{Matrix{ComplexF64}} = []

    sigma_ir::Vector{Matrix{ComplexF64}} = []
    sigma_tau::Vector{Matrix{ComplexF64}} = []
    sigma_matsu::Vector{Matrix{ComplexF64}} = []

    n_ir::Int = 0

    return g0_ir, g0_tau, g0_matsu, g_ir, g_matsu, sigma_ir, sigma_tau, sigma_matsu, n_ir
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

using Flux

function F_rho0(ir::IR_params, g::Green_Sigma, rho_ls, λ)
    vec::Vector{Matrix{ComplexF64}} = []
    for i in 1:g.n_ir
        test = g.g0_ir[i] - ir.basis.s .* rho_ls[i]
        push!(vec, test)
    end
    f = 0.0
    for i in 1:g.n_ir
        f += 0.5*real(sum(abs.((vec[i])'*vec[i]))) + λ*sum(abs.(rho_ls))
    end
    
    return f
end

function fit_rho0w(ir::IR_params, g::Green_Sigma, l_num::Int, batch_num::Int, w_mesh::Vector{Float64})
    sn = range(-12.0, 0.0, length=l_num)
    lam_test = 10 .^ (sn)
    opt = ADAM()
    s_rho_l = rand(Float64, l_num, g.n_ir)
    s_F_rho = 1000.0 * ones(Float64, l_num)
    for ll in 1:l_num
        lam = lam_test[ll]
        for b in 1:batch_num
            rho_ll = rand(Float64, g.n_ir)
            F_old = 1000.0
            F_new = 1000.0
            for i in 1:10000
                grads = Flux.gradient(Flux.params(rho_ll)) do
                    F_rho0(ir, g, rho_ll, lam)
                end
                Flux.Optimise.update!(opt, Flux.params(rho_ll), grads)
                F_old = F_new
                F_new = F_rho(ir, g, rho_ll, lam)
                if(abs(F_old-F_new)/abs(F_old)<1e-6)
                    break
                end
            end
            if(s_F_rho[ll] > F_rho0(ir, g, rho_ll, lam))
                s_rho_l[ll,:] = rho_ll
                s_F_rho[ll] = F_rho0(ir, g, rho_ll, lam)
            end
        end
    end
    b = (log(s_F_rho[end])-log(s_F_rho[1]))/(log(lam_test[end])-log(lam_test[1]))
    a = s_F_rho[1]*lam_test[1]^(-b)
    ev = a .* lam_test .^ b ./ s_F_rho

    p1 = plot(lam_test, ev, xaxis=:log, yaxis=:log, marker=:circle)
    savefig(p1,"./lambda_opt_G0.png")

    it = findmax(ev)[2]
    max1 = s_F_rho[it]
    println("it:$it,  s_F:$max1")
    rho_omega = -transpose(ir.basis.v(w_mesh)) * s_rho_l[it,:]
    return rho_omega
end

function F_rho(ir::IR_params, g::Green_Sigma, rho_ls, λ)
    vec::Vector{Matrix{ComplexF64}} = []
    for i in 1:g.n_ir
        test = g.g_ir[i] - ir.basis.s .* rho_ls[i]
        push!(vec, test)
    end
    f = 0.0
    for i in 1:g.n_ir
        f += 0.5*real(sum(abs.((vec[i])'*vec[i]))) + λ*sum(abs.(rho_ls))
    end
    return f
end

function fit_rhow(ir::IR_params, g::Green_Sigma, l_num::Int, batch_num::Int, w_mesh::Vector{Float64})
    sn = range(-12.0, 0.0, length=l_num)
    lam_test = 10 .^ (sn)
    opt = ADAM()
    s_rho_l = rand(Float64, l_num, g.n_ir)
    s_F_rho = 1000.0 * ones(Float64, l_num)
    for ll in 1:l_num
        lam = lam_test[ll]
        for b in 1:batch_num
            rho_ll = rand(Float64, g.n_ir)
            F_old = 1000.0
            F_new = 1000.0
            for i in 1:10000
                grads = Flux.gradient(Flux.params(rho_ll)) do
                    F_rho(ir, g, rho_ll, lam)
                end
                Flux.Optimise.update!(opt, Flux.params(rho_ll), grads)
                F_old = F_new
                F_new = F_rho(ir, g, rho_ll, lam)
                if(abs(F_old-F_new)/abs(F_old)<1e-6)
                    break
                end
            end
            if(s_F_rho[ll] > F_rho(ir, g, rho_ll, lam))
                s_rho_l[ll,:] = rho_ll
                s_F_rho[ll] = F_rho(ir, g, rho_ll, lam)
            end
        end
    end
    b = (log(s_F_rho[end])-log(s_F_rho[1]))/(log(lam_test[end])-log(lam_test[1]))
    a = s_F_rho[1]*lam_test[1]^(-b)
    ev = a .* lam_test .^ b ./ s_F_rho

    p1 = plot(lam_test, ev, xaxis=:log, yaxis=:log, marker=:circle)
    savefig(p1,"./lambda_opt_G.png")

    it = findmax(ev)[2]+1
    max1 = s_F_rho[it]
    println("it:$it,  s_F:$max1")
    rho_omega = -transpose(ir.basis.v(w_mesh)) * s_rho_l[it,:]
    return rho_omega
end

function reshape(rho::Vector{Float64}, cutoff::Float64)
    v_it::Vector{Int} = []
    rho_rep = rho
    sw = false
    for w in 1:length(rho_rep)
        if(rho_rep[w]<0)
            if(sw)
                for it in v_it
                    rho_rep[it] = 0.005                
                end
                empty!(v_it)
                sw = false
            elseif(w<length(rho_rep) && rho_rep[w+1]>rho_rep[w])
                sw = true
            end
            rho_rep[w] = 0.005

        elseif(rho_rep[w]<cutoff)
            if(sw)
                push!(v_it,w)
            elseif(w<length(rho_rep) && rho_rep[w+1]>rho_rep[w])
                sw = true
                push!(v_it,w)
            end
        elseif(rho_rep[w]>=cutoff && sw && w<length(rho_rep))
            empty!(v_it)
            sw = false
        end
    end
    return rho_rep
end


function KK_GR(w::Vector{Float64}, rho::Vector{Float64})
    GR_ = zeros(ComplexF64, length(w))
    dw = w[2]-w[1]
    for w_re in 1:length(w)
        re::Float64 = 0.0
        for w_im in 1:length(w)
            if(w_im != w_re)
                re += dw * rho[w_im] / (w[w_re] - w[w_im])
            end 
        end
        GR_[w_re] = re - 1.0im * rho[w_re] * pi
    end
    return GR_
end

ENV["GKSwstype"]="nul"
Plots.scalefontsizes(1.4)



using DataFrames
using CSV

function main(arg::Vector{String})
    p = Parm(set_parm(arg)...)
    println(p)
    ir = IR_params(set_IR(parse(Float64,arg[6]),parse(Float64,arg[7]),parse(Float64,arg[8]))...)
    lamda_num = parse(Int,arg[9])
    batch_num = parse(Int,arg[10])
    w_num = parse(Int,arg[11])
    g = Green_Sigma(init_zero_g(ir)...)
    w_mesh = collect(-ir.bw:2ir.bw/(w_num-1):ir.bw)

    kk = get_kk(p.K_SIZE)
    Disp_HSL(p)

    for it in 1:1000
        L1 = update_g!(p,kk,it,ir,g, 0.2)
        if(L1<1e-7)
            println(it)
            break
        end
    end

    g0 = -1.0im .* fit_rho0w(ir, g, lamda_num, batch_num, w_mesh)
    gi = -1.0im .* fit_rhow(ir, g, lamda_num, batch_num, w_mesh)
    #=
    sigma_w = zeros(ComplexF64, w_num)
    for ww in 1:w_num
        if(abs(g0)<1e-4 || abs(gi)<1e-4)
        else
            sigma_w[ww] = 1.0/g0[ww] - 1.0/gi[ww] - p.mu
        end
    end=#
    save_data_g = DataFrame(w=w_mesh,img=imag.(gi),reg=real.(gi))
    #save_data_s = DataFrame(w=w_mesh,ims=imag.(sigma_w),res=real.(sigma_w))

    pg0 = plot(w_mesh, imag.(g0), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    pg0 = plot!(w_mesh, real.(g0), linewidth=3.0)
    savefig(pg0,"./LDOS_free.png")

    pg = plot(w_mesh, imag.(gi), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    pg = plot!(w_mesh, real.(gi), linewidth=3.0)
    savefig(pg,"./LDOS.png")

    #ps = plot(w_mesh, imag.(sigma_w), linewidth=3.0, xlabel="ω", ylabel="Σ(ω)", title="self-energy")
    #ps = plot!(w_mesh, real.(sigma_w), linewidth=3.0)
    #savefig(ps,"./self-energy.png")
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./GF_U$(ir.U)_b$(ir.beta).csv", save_data_g)
    #CSV.write("./Sigma_U$(ir.U)_b$(ir.beta).csv", save_data_s)

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