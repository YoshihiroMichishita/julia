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


using Combinatorics
function set_mu_order(K::Int, μ::Vector{Float32})
    ord = combinations(μ, K)
    return ord
end

function set_mu_order2(K::Int, μ::Vector{Float32})
    data = []
    l = length(μ)
    for i in 1:l^K
        vec = zeros(Float32,K)
        for j in 1:K
            n = l^(K-j)
            s = (i-1)%(l*n)
            x = div(s,n)
            vec[j] = μ[x+1]
        end
        push!(data, vec)
    end
    return data
end

function set_σ_order(K::Int, σ::Vector{Float32})
    data = []
    l = length(σ)
    for i in 1:l^K
        vec = zeros(Float32,K)
        for j in 1:K
            n = l^(K-j)
            s = (i-1)%(l*n)
            x = div(s,n)
            vec[j] = σ[x+1]
        end
        push!(data, vec)
    end
    return data
end

function set_ϕ_order(K::Int, ϕ::Vector{Float32})
    data = []
    l = length(ϕ)
    for i in 1:l^K
        vec = zeros(Float32,K)
        for j in 1:K
            n = l^(K-j)
            s = (i-1)%(l*n)
            x = div(s,n)
            vec[j] = ϕ[x+1]
        end
        push!(data, vec)
    end
    return data
end

function add_noise(K::Int, μ::Vector{Float32}, σ::Vector{Float32}, ϕ::Vector{Float32}, η::Vector{Float32})
    μ1 = μ + η[1] * randn(Float32, K)
    σ1 = exp.(σ + η[2] * randn(Float32, K))
    ϕ1 = softmax0(ϕ + η[3] * randn(Float32, K))
    return μ1, σ1, ϕ1
end

function create_data6_orderly(w_size::Int, n_gauss::Int, ir::IR_params, μ::Vector{Float32}, σ::Vector{Float32}, ϕ::Vector{Float32})
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 3n_gauss)
    it = 0
    while(true)
        if(it< 20)
            μ1, σ1, ϕ1 = add_noise(n_gauss, μ, σ, ϕ, [0.1f0, 1f0, 1f0])
            gmm_params = gParams(n_gauss, μ1, σ1, ϕ1)
        else
            gmm_params = rand_init_params(n_gauss)
        end
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            if(sum(gmm_rho0) < 0.9f0)
                continue
            else
                gl_gmm = Float32.(loginv.(rho2gl(ws, gmm_rho0, ir)))
                data = gparams2data(gmm_params)
                break
            end
        end
        it+=1
    end
    return gl_gmm, data
end

function create_data6_orderly2(w_size::Int, n_gauss::Int, ir::IR_params, μ::Vector{Float32}, σ::Vector{Float32}, ϕ::Vector{Float32}, noise_vec::Vector{Float32})
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 3n_gauss)
    it = 0
    while(true)
        if(it< 20)
            μ1, σ1, ϕ1 = add_noise(n_gauss, μ, σ, ϕ, noise_vec)
            gmm_params = gParams(n_gauss, μ1, σ1, ϕ1)
        else
            gmm_params = rand_init_params(n_gauss)
        end
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            if(sum(gmm_rho0) < 0.9f0)
                continue
            else
                gl_gmm = Float32.(loginv.(rho2gl(ws, gmm_rho0, ir)))
                data = gparams2data(gmm_params)
                break
            end
        end
        it+=1
    end
    return gl_gmm, data
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
include("cauchy.jl")
function create_data6_c(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_ccm = zeros(Float32, l)
    data = zeros(Float32, 3n_gauss)
    while(true)
        ccm_params = init_rand_params(n_gauss)
        ccm_rho0 = ccm_rho(ws, ccm_params)
        if(ccm_rho0[1] < 1f-2 && ccm_rho0[end] < 1f-2)
            gl_ccm = Float32.(loginv.(rho2gl(ws, ccm_rho0, ir)))
            data = cparams2data(ccm_params)
            break
        end
    end
    #data = gparams2data(gmm_params)
    return gl_ccm, data
end

function create_data6_2(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 3n_gauss)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            gl_gmm = Float32.(loginv.(rho2gl(ws, gmm_rho0, ir)))
            #data = gparams2data_μϕperm(gmm_params)
            data = gparams2data_μperm(gmm_params)
            break
        end
    end
    #data = gparams2data(gmm_params)
    return gl_gmm, data
end

function create_data6_3(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 3n_gauss)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            gl_gmm = Float32.(loginv.(rho2gl(ws, gmm_rho0, ir)))
            data = gparams2data_σperm(gmm_params)
            break
        end
    end
    #data = gparams2data(gmm_params)
    return gl_gmm, data
end

function create_data6_4(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 3n_gauss)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            gl_gmm = Float32.(loginv.(rho2gl(ws, gmm_rho0, ir)))
            data = gparams2data_μσperm(gmm_params)
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

function create_data8(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 9)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            gl_gmm = Float32.(loginv.(rho2gl(ws, gmm_rho0, ir)))
            data = gparams2data_sym2(gmm_params)
            break
        end
    end
    #data = gparams2data(gmm_params)
    return gl_gmm, data
end

function create_data8_ver2(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 9)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            gl_gmm = Float32.(loginv.(rho2gl(ws, gmm_rho0, ir)))
            data = gparams2data_sym2_2(gmm_params)
            break
        end
    end
    #data = gparams2data(gmm_params)
    return gl_gmm, data
end

function create_data8_check(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 9)
    data_orig = zeros(Float32, 3n_gauss)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            gl_gmm = Float32.(loginv.(rho2gl(ws, gmm_rho0, ir)))
            data = gparams2data_sym2(gmm_params)
            data_orig = gparams2data(gmm_params)
            break
        end
    end
    #data = gparams2data(gmm_params)
    return gl_gmm, data, data_orig
end

function create_data9(w_size::Int, n_gauss::Int, ir::IR_params)
    ws = [range(-ir.bw,ir.bw, length=w_size)...]
    l = length(ir.basis.s)
    gl_gmm = zeros(Float32, l)
    data = zeros(Float32, 9)
    while(true)
        gmm_params = rand_init_params(n_gauss)
        gmm_rho0 = gmm_rho(ws, gmm_params)
        if(gmm_rho0[1] < 1f-2 && gmm_rho0[end] < 1f-2)
            rhol, dkl = check_rhol_dkl(ws, gmm_rho0, ir)
            if(dkl < 1f-1)
                gl_gmm = -Float32.(loginv.(ir.basis.s .* rhol))
                #gl_gmm = Float32.(rho2gl(ws, gmm_rho0, ir))
                data = gparams2data_sym2_2(gmm_params)
                break
            end
        end
    end
    #data = gparams2data(gmm_params)
    return gl_gmm, data
end

