include("model_2D_test.jl")

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
    basis = FiniteTempBasis(Fermionic(), beta, bw, 1e-8)

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
    g0_ir::Vector{ComplexF64}
    g0_tau::Vector{ComplexF64}
    g0_matsu::Vector{ComplexF64}

    g_ir::Vector{ComplexF64}
    g_matsu::Vector{ComplexF64}

    sigma_ir::Vector{ComplexF64}
    sigma_tau::Vector{ComplexF64}
    sigma_matsu::Vector{ComplexF64}

    n_ir::Int
end

function init_zero_g(ir::IR_params)
    
    g0_tau = zeros(ComplexF64, ir.n_tau)
    g0_matsu = zeros(ComplexF64, ir.n_matsu)
    g0_ir = fit(ir.smpl_matsu, g0_matsu, dim=1)

    n_ir = size(g0_ir)[1]

    g_ir = zeros(ComplexF64, n_ir)
    g_matsu = zeros(ComplexF64, ir.n_matsu)

    sigma_ir = zeros(ComplexF64, n_ir)
    sigma_tau = zeros(ComplexF64, ir.n_tau)
    sigma_matsu = zeros(ComplexF64, ir.n_matsu)

    return g0_ir, g0_tau, g0_matsu, g_ir, g_matsu, sigma_ir, sigma_tau, sigma_matsu, n_ir
end


function gk_m(p::Parm, k::Vector{Float64}, w::ComplexF64, g::Green_Sigma)
    e = set_H(k,p)
    gk = 1.0/(w - e - g.sig)
    return gk
end

function get_G0mlocal(p::Parm, k_BZ::Vector{Vector{Float64}}, w::ComplexF64, sw::Int, sigma::ComplexF64)
    gw_l = 0.0
    gl = 0.0
    if(sw == 1)
        for i in 1:length(k_BZ)
            e = set_H(k_BZ[i],p) - p.mu
            #gk = 1.0/(w - e + p.eta*1.0im*sign(imag(w)))
            gk = 1.0/(w - e)
            gw_l += p.dk2 * gk
        end
    else
        for i in 1:length(k_BZ)
            e = set_H(k_BZ[i],p) - p.mu
            #gk = 1.0/(w - e -sigma + p.eta*1.0im*sign(imag(w)))
            gk = 1.0/(w - e -sigma)
            gl += p.dk2 * gk
        end
        gw_l = 1.0/(1.0/gl + sigma)
    end
    return gw_l, gl
end

function get_G0mlocal_3D(p::Parm, w::ComplexF64, sw::Int, sigma::ComplexF64)
    gw_l = 0.0
    gl = 0.0
    dk = 2pi/p.K_SIZE
    if(sw == 1)
        for kz in collect(-pi+dk:dk:pi)
            kk = get_kk(p.K_SIZE, kz)
            for i in 1:length(kk)
                k_BZ = [(kk[i])[1], (kk[i])[2], kz]
                e = set_H(k_BZ,p) - p.mu
                #gk = 1.0/(w - e + p.eta*1.0im*sign(imag(w)))
                gk = 1.0/(w - e)
                gw_l += p.dk2 * gk
            end
        end
    else
        for kz in collect(-pi+dk:dk:pi)
            kk = get_kk(p.K_SIZE, kz)
            for i in 1:length(kk)
                k_BZ = [(kk[i])[1], (kk[i])[2], kz]
                e = set_H(k_BZ,p) - p.mu
                #gk = 1.0/(w - e -sigma + p.eta*1.0im*sign(imag(w)))
                gk = 1.0/(w - e -sigma)
                gl += p.dk2 * gk
            end
            gw_l = 1.0/(1.0/gl + sigma)
        end
    end
    return gw_l, gl
end

function MatsuToTau!(ir::IR_params, g::Green_Sigma)
    g.g0_ir = fit(ir.smpl_matsu, g.g0_matsu, dim=1)
    g.g_ir = fit(ir.smpl_matsu, g.g_matsu, dim=1)
    g.g0_tau = evaluate(ir.smpl_tau, g.g0_ir, dim=1)
end

function TauToMatsu!(ir::IR_params, g::Green_Sigma, γ::Float64)
    ir_new = fit(ir.smpl_tau, g.sigma_tau)
    diff = sum(abs.(ir_new .- g.sigma_ir))/sum(abs.(g.sigma_ir))
    g.sigma_ir = (1.0-γ) .* g.sigma_ir .+ γ .* ir_new
    g.sigma_matsu = evaluate(ir.smpl_matsu, g.sigma_ir, dim=1)

    return diff
end

function update_g!(p::Parm, k_BZ::Vector{Vector{Float64}},sw::Int, ir::IR_params, g::Green_Sigma, γ::Float64)
    for w in 1:ir.n_matsu
        #g.g0_matsu[w], g.g_matsu[w] = get_G0mlocal(p, k_BZ, valueim(ir.smpl_matsu.sampling_points[w], ir.beta), sw, g.sigma_matsu[w])
        g.g0_matsu[w], g.g_matsu[w] = get_G0mlocal(p, k_BZ, ir.smpl_wn[w], sw, g.sigma_matsu[w])
    end
    MatsuToTau!(ir, g)
    g.sigma_tau = ir.U^2 .* (g.g0_tau).^2 .* g.g0_tau[end:-1:1]
    # .+ ir.U .* g.g0_tau
    diff = TauToMatsu!(ir, g, γ)
    return diff
end

function update_g_3D!(p::Parm,sw::Int, ir::IR_params, g::Green_Sigma, γ::Float64)
    for w in 1:ir.n_matsu
        #g.g0_matsu[w], g.g_matsu[w] = get_G0mlocal(p, k_BZ, valueim(ir.smpl_matsu.sampling_points[w], ir.beta), sw, g.sigma_matsu[w])
        g.g0_matsu[w], g.g_matsu[w] = get_G0mlocal_3D(p, ir.smpl_wn[w], sw, g.sigma_matsu[w])
    end
    MatsuToTau!(ir, g)
    g.sigma_tau = ir.U^2 .* (g.g0_tau).^2 .* g.g0_tau[end:-1:1]
    # .+ ir.U .* g.g0_tau
    diff = TauToMatsu!(ir, g, γ)
    return diff
end

using Flux

function F_rho0(ir::IR_params, g::Green_Sigma, rho_ls, λ)
    vec = g.g0_ir - ir.basis.s .* rho_ls
    return f = 0.5*real(vec'*vec) + λ*sum(abs.(rho_ls))
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
    vec = g.g_ir - ir.basis.s .* rho_ls
    return f = 0.5*real(vec'*vec) + λ*sum(abs.(rho_ls))
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

    it = findmax(ev)[2] + 1
    max1 = s_F_rho[it]
    println("it:$it,  s_F:$max1")
    rho_omega = -transpose(ir.basis.v(w_mesh)) * s_rho_l[it,:]

    s_rho_l0 = rand(Float64, l_num, g.n_ir)
    s_F_rho0 = 1000.0 * ones(Float64, l_num)
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
                F_new = F_rho0(ir, g, rho_ll, lam)
                if(abs(F_old-F_new)/abs(F_old)<1e-6)
                    break
                end
            end
            if(s_F_rho0[ll] > F_rho0(ir, g, rho_ll, lam))
                s_rho_l0[ll,:] = rho_ll
                s_F_rho0[ll] = F_rho0(ir, g, rho_ll, lam)
            end
        end
    end
    b0 = (log(s_F_rho0[end])-log(s_F_rho0[1]))/(log(lam_test[end])-log(lam_test[1]))
    a0 = s_F_rho0[1]*lam_test[1]^(-b)
    ev0 = a0 .* lam_test .^ b0 ./ s_F_rho0

    p1 = plot(lam_test, ev0, xaxis=:log, yaxis=:log, marker=:circle)
    savefig(p1,"./lambda_opt_G0.png")

    it0 = findmax(ev0)[2] + 1
    max0 = s_F_rho0[it0]
    println("it:$it0,  s_F:$max0")
    rho_omega0 = -transpose(ir.basis.v(w_mesh)) * s_rho_l0[it0,:]

    #=
    s_rho00 = zeros(Float64, g.n_ir)
    s_F_rho0 = 1000.0
    lam = lam_test[it]
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
            F_new = F_rho0(ir, g, rho_ll, lam)
            if(abs(F_old-F_new)/abs(F_old)<1e-6)
                break
            end
        end
        if(s_F_rho0 > F_rho0(ir, g, rho_ll, lam))
            s_rho00 = rho_ll
            s_F_rho0 = F_rho0(ir, g, rho_ll, lam)
        end
    end
    rho_omega0 = -transpose(ir.basis.v(w_mesh)) * s_rho00
    =#

    return rho_omega, rho_omega0
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
        elseif(rho_rep[w]<cutoff && sw)
            push!(v_it,w)
        elseif(rho_rep[w]>=cutoff && sw && w<length(rho_rep) && rho_rep[w+1]<rho_rep[w])
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

f(beta::Float64, w::Float64) = 1.0/(1.0+exp(beta*w))

function renorm_rho(beta::Float64, w::Vector{Float64}, rho::Vector{Float64})
    n = 0.0
    dw = w[2]-w[1]
    for i in 1:length(w)
        n += dw * rho[i] * f(beta, w[i])
    end
    rho1 = (0.5/n) * rho
    return rho1
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

    for it in 1:8000
        L1 = update_g!(p,kk,it,ir,g, 0.2)
        if(L1<1e-8)
            println(it)
            break
        end
    end

    #g0 = -1.0im .* fit_rho0w(ir, g, lamda_num, batch_num, w_mesh)
    gi, g0 = fit_rhow(ir, g, lamda_num, batch_num, w_mesh)
    #rint_res = reshape(gi, pi/ir.beta)
    #r0_res = reshape(g0, pi/ir.beta)
    rint_res = reshape(gi, 0.2)
    r0_res = reshape(g0, 0.2)

    rint_res = renorm_rho(ir.beta, w_mesh, rint_res)
    r0_res = renorm_rho(ir.beta, w_mesh, r0_res)

    GR_int = KK_GR(w_mesh, rint_res)
    GR_0 = KK_GR(w_mesh, r0_res)

    sigma_w = zeros(ComplexF64, w_num)
    for ww in 1:w_num
        sigma_w[ww] = 1.0/GR_0[ww] - 1.0/GR_int[ww] - p.mu
        if(imag(sigma_w[ww])>0)
            sigma_w[ww] = real(sigma_w[ww]) - 0.01im
        end
    end
    
    save_data_g = DataFrame(w=w_mesh,img=imag.(GR_int),reg=real.(GR_int))
    save_data_s = DataFrame(w=w_mesh,ims=imag.(sigma_w),res=real.(sigma_w))

    pg0 = plot(w_mesh, imag.(GR_0), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    pg0 = plot!(w_mesh, real.(GR_0), linewidth=3.0)
    savefig(pg0,"./LDOS_free.png")

    pg = plot(w_mesh, imag.(GR_int), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    pg = plot!(w_mesh, real.(GR_int), linewidth=3.0)
    savefig(pg,"./LDOS.png")

    ps = plot(w_mesh, imag.(sigma_w), linewidth=3.0, xlabel="ω", ylabel="Σ(ω)", title="self-energy")
    ps = plot!(w_mesh, real.(sigma_w), linewidth=3.0)
    savefig(ps,"./self-energy.png")
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./GF_U$(ir.U)_b$(ir.beta).csv", save_data_g)
    CSV.write("./Sigma_U$(ir.U)_b$(ir.beta).csv", save_data_s)

    g_check = zeros(Float64, w_num)
    for w in 1:w_num
        for i in 1:length(kk)
            e = set_H(kk[i],p)
            gk = 1.0/(w_mesh[w] - e -sigma_w[w] + 1.0im*p.eta)
            g_check[w] += -p.dk2 * imag(gk)/pi
        end        
    end
    pc = plot(w_mesh, g_check, linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    savefig(pc,"./LDOS_check.png")


    Spectral_HSL(p, w_mesh, sigma_w)

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