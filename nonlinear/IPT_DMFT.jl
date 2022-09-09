include("model_2D.jl")

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
    basis::FiniteTempBasis{Fermionic, LogisticKernel, Float64, Float64}
    
    smpl_matsu::MatsubaraSampling64F
    smpl_tau::TauSampling64
    n_matsu::Int
    n_tau::Int
end

function set_IR(U::Float64, beta::Float64, bw::Float64)
    basis = FiniteTempBasis(Fermionic(), beta, bw, 1e-6)

    smpl_matsu = MatsubaraSampling(basis)
    n_matsu = size(smpl_matsu.sampling_points)[1]

    smpl_tau = TauSampling(basis)
    n_tau = size(smpl_tau.sampling_points)[1]

    return U, beta, bw, basis, smpl_matsu, smpl_tau, n_matsu, n_tau
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
end

function init_zero_g(ir::IR_params)
    
    g0_tau = zeros(ComplexF64, ir.n_tau)
    g0_matsu = zeros(ComplexF64, ir.n_matsu)
    g0_ir = fit(ir.smpl_matsu, g0_matsu, dim=1)

    g_ir = zeros(ComplexF64, length(g0_ir))
    g_matsu = zeros(ComplexF64, ir.n_matsu)

    sigma_ir = zeros(ComplexF64, length(g0_ir))
    sigma_tau = zeros(ComplexF64, ir.n_tau)
    sigma_matsu = zeros(ComplexF64, ir.n_matsu)

    return g0_ir, g0_tau, g0_matsu, g_ir, g_matsu, sigma_ir, sigma_tau, sigma_matsu
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
            gk = 1.0/(w - e)
            gw_l += p.dk2 * gk
        end
    else
        for i in 1:length(k_BZ)
            e = set_H(k_BZ[i],p)
            gk = 1.0/(w - e -sigma)
            gl += p.dk2 * gk
        end
        gw_l = 1.0/(1.0/gl + sigma)
    end
    return gw_l, gl
end

function MatsuToTau!(ir::IR_params, g::Green_Sigma)
    g.g0_ir = fit(ir.smpl_matsu, g.g0_matsu, dim=1)
    g.g0_tau = evaluate(ir.smpl_tau, g.g0_ir, dim=1)
end

function TauToMatsu!(ir::IR_params, g::Green_Sigma)
    g.sigma_ir = fit(ir.smpl_tau, g.sigma_tau)
    g.sigma_matsu = evaluate(ir.smpl_matsu, g.sigma_ir, dim=1)
end

function update_g!(p::Parm, k_BZ::Vector{Vector{Float64}},sw::Int, ir::IR_params, g::Green_Sigma)
    for w in ir.n_matsu
        g.g0_matsu[w], g.g_matsu[w] = get_G0mlocal(p, k_BZ, valueim(ir.smpl_matsu.sampling_points[w], ir.beta), sw, g.sigma_matsu[w])
    end
    MatsuToTau!(ir, g)
    g.sigma_tau = ir.U^2 .* (g.g0_tau).^2 .* g.g0_tau[end:-1:1]
    TauToMatsu!(ir, g)
end

using Flux

function F_rho(ir::IR_params, g::Green_Sigma, rho_ls, λ)
    vec = g.g0_ir - ir.basis.s .* rho_ls
    return f = 0.5*real(vec'*vec) + λ*sum(abs.(rho_ls))
end

function fit_rho0w(ir::IR_params, g::Green_Sigma, l_num::Int, batch_num::Int, w_mesh::Vector{Float64})
    sn = range(-6.0, 1.0, length=l_num)
    lam_test = 10 .^ (sn)
    opt = ADAM()
    s_rho_l = rand(Float64, l_num, ir.n_matsu-1)
    s_F_rho = 1000.0 * ones(Float64, l_num)
    for ll in 1:l_num
        lam = lam_test[ll]
        for b in 1:batch_num
            rho_ll = rand(Float64, ir.n_matsu-1)
            for i in 1:8000
                grads = Flux.gradient(Flux.params(rho_ll)) do
                    F_rho(ir, g, rho_ll, lam)
                end
                Flux.Optimise.update!(opt, Flux.params(rho_ll), grads)
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
    savefig(p1,"./lambda_opt_G0.png")

    it = findmax(ev)[2]
    println(it)
    rho_omega = -transpose(ir.basis.v(w_mesh)) * s_rho_l[it,:]
    return rho_omega
end

function fit_rhow(ir::IR_params, g::Green_Sigma, l_num::Int, batch_num::Int, w_mesh::Vector{Float64})
    sn = range(-6.0, 1.0, length=l_num)
    lam_test = 10 .^ (sn)
    opt = ADAM()
    s_rho_l = rand(Float64, l_num, ir.n_matsu-1)
    s_F_rho = 1000.0 * ones(Float64, l_num)
    for ll in 1:l_num
        lam = lam_test[ll]
        for b in 1:batch_num
            rho_ll = rand(Float64, ir.n_matsu-1)
            for i in 1:8000
                grads = Flux.gradient(Flux.params(rho_ll)) do
                    F_rho(ir, g, rho_ll, lam)
                end
                Flux.Optimise.update!(opt, Flux.params(rho_ll), grads)
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

    it = findmax(ev)[2]
    rho_omega = -transpose(ir.basis.v(w_mesh)) * s_rho_l[it,:]
    return rho_omega
end

ENV["GKSwstype"]="nul"
Plots.scalefontsizes(1.4)


function Spectral_HSL(p::Parm,w_mesh::Vector{Float64}, sigma_w::Vector{ComplexF64})
    E = zeros(Float64, 4*p.K_SIZE, length(sigma_w))

    for K0 in 1:p.K_SIZE
        KK = 4pi/3*K0/p.K_SIZE
        k = 4pi/3*a1 + KK*a2
        E[K0,:] = -imag.(1.0 ./ (w_mesh .- set_H(k, p) .- sigma_w))/pi
    end
    for K0 in 1:p.K_SIZE
        KK = 2pi/3*K0/p.K_SIZE
        k = 4pi/3*a3 - KK*a1
        E[K0+p.K_SIZE,:] = -imag.(1.0 ./ (w_mesh .- set_H(k, p) .- sigma_w))/pi
    end
    for K0 in 1:p.K_SIZE
        KK = 4pi/3*K0/p.K_SIZE
        k = 4pi/3*a3 -2pi/3*a1 + KK*(0.5*a1 - a3)
        E[K0+2p.K_SIZE,:] = -imag.(1.0 ./ (w_mesh .- set_H(k, p) .- sigma_w))/pi
    end
    for K0 in 1:p.K_SIZE
        KK = 4pi/3*K0/p.K_SIZE
        k = KK*(a1)
        E[K0+3p.K_SIZE,:] = -imag.(1.0 ./ (w_mesh .- set_H(k, p) .- sigma_w))/pi
    end

    q = 1:4p.K_SIZE

    p1 = plot(q, w_mesh, E, st=:heatmap, xticks=([0, p.K_SIZE/2, p.K_SIZE,2p.K_SIZE,3p.K_SIZE,4p.K_SIZE],["K", "M", "K'", "M'", "Γ", "K"]),xlabel="HSL",ylabel="ω",colorbar_title="A(k,ω)",title="Spectral")
    savefig(p1,"./disp.png")

    return nothing
end

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
    w_mesh = collect(-ir.bw:2ir.bw/w_num:ir.bw)

    kk = get_kk(p.K_SIZE)
    Disp_HSL(p)

    for it in 0:50
        s_old = g.sigma_ir
        update_g!(p,kk,it,ir,g)
        diff = g.sigma_ir .- s_old
        L1 = sum(abs.(diff))
        if(L1<1e-6)
            println(it)
            break
        end
    end

    g0 = -1.0im .* fit_rho0w(ir, g, lamda_num, batch_num, w_mesh)
    g = -1.0im .* fit_rhow(ir, g, lamda_num, batch_num, w_mesh)
    sigma_w = 1.0 ./ g0 .- 1.0 ./ g
    save_data_g = DataFrame(w=w_mesh,img=imag.(g),reg=real.(g))
    save_data_s = DataFrame(w=w_mesh,ims=imag.(sigma_w),res=real.(sigma_w))

    pg0 = plot(w_mesh, imag.(g0), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    pg0 = plot!(w_mesh, real.(g0), linewidth=3.0)
    savefig(pg0,"./LDOS_free.png")

    pg = plot(w_mesh, imag.(g), linewidth=3.0, xlabel="ω", ylabel="A(ω)", title="local DOS")
    savefig(pg,"./LDOS.png")

    ps = plot(w_mesh, imag.(sigma_w), linewidth=3.0, xlabel="ω", ylabel="Σ(ω)", title="self-energy")
    ps = plot!(w_mesh, real.(sigma_w), linewidth=3.0)
    savefig(ps,"./self-energy.png")
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./GF_U$(ir.U)_b$(ir.beta).csv", save_data_g)
    CSV.write("./Sigma_U$(ir.U)_b$(ir.beta).csv", save_data_s)

    Spectral_HSL(p, w_mesh, sigma_w)

end

@time main(ARGS)