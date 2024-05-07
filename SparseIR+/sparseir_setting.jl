using SparseIR
import SparseIR: valueim, value


include("gmm.jl")

struct IR_params
    beta::Float64 #inverse temperature
    bw::Float64 #Band-width

    basis::FiniteTempBasis #this is the struct defined in SparseIR. This holds the information about Fermionic/Bosonic, and the matrix which can translate the Matsubara or imaginary time to IR-basis and the inverse.
    #{Fermionic, LogisticKernel, Float64, Float64}
    
    smpl_matsu::MatsubaraSampling64F #the sampling points of Matsubara frequency
    smpl_tau::TauSampling64 #the sampling points of imaginary time

    n_matsu::Int # the number of the sampling points of Matsubara frequency
    n_tau::Int # the number of the sampling points of imaginary time

    smpl_wn::Vector{ComplexF64}#the sampling points of Matsubara frequency(type is different from smpl_matsu)
end

#IR_paramsをセットするための関数
function set_IR(beta::Float64, bw::Float64, del::Float64)
    #SparseIRの関数を使ってbasisを定義する。
    #FiniteTempBasis(粒子の種類, 逆温度, バンド幅, 許容する誤差(特異値分解の際に使用))
    basis = FiniteTempBasis(Fermionic(), beta, bw, del)

    #よく使うことになるので、basisから松原と虚時間のsampling pointsを取り出しておく
    smpl_matsu = MatsubaraSampling(basis)
    n_matsu = size(smpl_matsu.sampling_points)[1]

    smpl_tau = TauSampling(basis)
    n_tau = size(smpl_tau.sampling_points)[1]

    #smpl_matsuのままだとtypeがよく分からず後に不便を感じたのでVector{Complex}の型に直しておく
    smpl_wn = zeros(ComplexF64, n_matsu)
    for w in 1:n_matsu
        smpl_wn[w] = valueim(smpl_matsu.sampling_points[w], beta)
    end

    return IR_params(beta, bw, basis, smpl_matsu, smpl_tau, n_matsu, n_tau, smpl_wn)
end

function cross_integration_sympson(f::Vector{Float64}, g::Vector{Float64}, dw::Float64)
    l = length(f)
    return dw/3*(f[1]*g[1] + 4*sum(f[2:2:end].*g[2:2:end]) + 2*sum(f[3:2:end-1].*g[3:2:end-1]) + f[end]*g[end])
end

function calc_rhol(ws::Vector{Float64} ,ρ::Vector{Float64}, ir_w_basis)
    ir_ls = ir_w_basis(ws)
    dw = (ws[2] - ws[1])
    #return dw*ir_ls'*ρ
    return cross_integration_sympson(ir_ls, ρ, dw)
end

function rho2gl(ws::Vector{Float64} ,ρ::Vector{Float64}, ir::IR_params)
    l = length(ir.basis.s)
    ρl = [calc_rhol(ws, ρ, ir.basis.v[i]) for i in 1:l]
    gl = -ir.basis.s .* ρl
    return gl
end

function check_rhol(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw, ir.bw, length=w_size)...]
    gmm_params = rand_init_params(n_gauss)
    gmm_rho0 = gmm_rho(ws, gmm_params)
    l = length(ir.basis.s)
    ρl = [calc_rhol(ws, gmm_rho0, ir.basis.v[i]) for i in 1:l]
    ρ_recov = ρl' * ir.basis.v(ws)
    return gmm_rho0, ρl, ρ_recov
end

function step_fn(x::Float64)
    if(x > 0)
        return x
    else
        return 0.0
    end
end

function D_kl(ws::Vector{Float64}, true_ρ::Vector{Float64}, ρ::Vector{Float64})
    #f(x) = gmm1(x, p)
    #K = length(ws)
    dw = ws[2]-ws[1]
    return sum(dw* true_ρ .* (log.(true_ρ.+1e-8).-log.(step_fn.(ρ) .+ 1e-8)))
end

function check_rhol_dkl(w_size::Int, n_gauss::Int, ir::IR_params, gp::gParams)
    ws = [range(-ir.bw, ir.bw, length=w_size)...]
    #gmm_params = rand_init_params(n_gauss)
    gmm_rho0 = gmm_rho(ws, gp)
    l = length(ir.basis.s)
    ρl = [calc_rhol(ws, gmm_rho0, ir.basis.v[i]) for i in 1:l]
    ρ_recov = ir.basis.v(ws)' * ρl
    dkl = D_kl(ws, gmm_rho0, ρ_recov)
    return ρl, dkl
end

function check_rhol_edge(w_size::Int, ir::IR_params, gp::gParams)
    ws = [range(-ir.bw, ir.bw, length=w_size)...]
    #gmm_params = rand_init_params(n_gauss)
    gmm_rho0 = gmm_rho(ws, gp)
    l = length(ir.basis.s)
    ρl = [calc_rhol(ws, gmm_rho0, ir.basis.v[i]) for i in 1:l]
    ρ_recov = ir.basis.v(ws)' * ρl
    dkl = D_kl(ws, gmm_rho0, ρ_recov)
    return ρl, dkl
end

function create_data(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    gmm_params = rand_init_params(n_gauss)
    gmm_rho0 = gmm_rho(ws, gmm_params)
    gl_gmm = Float32.(rho2gl(ws, gmm_rho0, ir))
    data = gparams2data(gmm_params)
    return (gl_gmm, data)
end
function create_data2(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    gmm_params = rand_init_params(n_gauss)
    gmm_rho0 = gmm_rho(ws, gmm_params)
    gl_gmm = Float32.(rho2gl(ws, gmm_rho0, ir))
    #data = gparams2data(gmm_params)
    return gl_gmm, gmm_rho0
end
function create_data3(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    gmm_params = rand_init_params(n_gauss)
    gmm_rho0 = gmm_rho(ws, gmm_params)
    gl_gmm = Float32.(rho2gl(ws, gmm_rho0, ir))
    #data = gparams2data(gmm_params)
    return gl_gmm
end
function create_data4(w_size::Int, n_gauss::Int, ir::IR_params)
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        rhol, dkl = check_rhol_dkl(w_size, n_gauss, ir, gmm_params)
        if(dkl < 1f-2)
            #gmm_rho0 = gmm_rho(ws, gmm_params)
            gl_gmm = -Float32.(ir.basis.s .* rhol)
            break
        end
    end
    return gl_gmm
end

#データの前処理
function loginv(x)
    if(x > 0)
        return 1.0/(1.0-log(x))
    else
        return -1.0/(1.0-log(-x))
    end
end

function create_data5(w_size::Int, n_gauss::Int, ir::IR_params)
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 3n_gauss)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        rhol, dkl = check_rhol_dkl(w_size, n_gauss, ir, gmm_params)
        if(dkl < 1f-2)
            gl_gmm = -Float32.(loginv.(ir.basis.s .* rhol))
            data = gparams2data(gmm_params)
            break
        end
    end
    return gl_gmm, data
end

function check_orth(ir::IR_params, w_size::Int, l0::Int)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    ws = ws[2:end-1]
    dw = (ws[2] - ws[1])
    l = length(ir.basis.s)
    orth = zeros(Float64, l)
    println(l)
    for i in 1:l
        #orth[i] = dw*(ir.basis.v[i](ws))'*(ir.basis.v[l0](ws))
        orth[i] = cross_integration_sympson(ir.basis.v[i](ws), ir.basis.v[l0](ws), dw)
    end
    return orth
end

function create_data6(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 3n_gauss)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            gl_gmm = Float32.(loginv.(rho2gl(ws, gmm_rho0, ir)))
            data = gparams2data(gmm_params)
            break
        end
    end
    #data = gparams2data(gmm_params)
    return gl_gmm, data
end

function check_rhol_dkl(ws::Vector{Float64}, gmm_rho0::Vector{Float64}, ir::IR_params)
    l = length(ir.basis.s)
    ρl = [calc_rhol(ws, gmm_rho0, ir.basis.v[i]) for i in 1:l]
    ρ_recov = ir.basis.v(ws)' * ρl
    dkl = D_kl(ws, gmm_rho0, ρ_recov)
    return ρl, dkl
end

function create_data7(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 3n_gauss)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            rhol, dkl = check_rhol_dkl(ws, gmm_rho0, ir)
            if(dkl < 1f-1)
                gl_gmm = -Float32.(loginv.(ir.basis.s .* rhol))
                #gl_gmm = Float32.(rho2gl(ws, gmm_rho0, ir))
                data = gparams2data(gmm_params)
                break
            end
        end
    end
    #data = gparams2data(gmm_params)
    return gl_gmm, data
end