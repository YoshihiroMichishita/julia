include("gmm.jl")

struct H_params
    t::Float32
    μ::Float32
    β::Float32
    K_size::Int
    ws::Vector{Float32}
end

function Ham(k::Vector{Float32}, p::H_params)
    return -p.t * sum(cos.(k)) + p.μ
end

mutable struct Hamiltonian
    Hk::Float32
    Vx::Float32
    Vy::Float32
    #ws::Vector{Float32}
end

function HandV(k::Vector{Float32}, p::H_params)
    Hk = Ham(k, p)
    Vx = p.t * sin(k[1])
    Vy = p.t * sin(k[2])
    #ws = [range(-1.0f0,1.0f0, length=501)...]
    return Hamiltonian(Hk, Vx, Vy)
end

function ρwk(k::Vector{Float32}, w_e::Float32, η::Float32)
    return η/((w_e)^2+ η^2)/pi
end

function ρwk_vec(k::Vector{Float32}, ws::Vector{Float32}, e::Float32, η::Float32)
    return [ρwk(k, w-e, η) for w in ws]
end

function EM_fit(itr_max::Int, X::Int, ws::Vector{Float32}, ρ::Vector{Float32})
    th = 1e-6
    gp = init_params(X)
    old_logpx = logpws(ws, true_ρ, gp)

    for i in 1:itr_max
        qs = E_step(ws, gp)
        M_step!(ws, ρ, qs, gp)
        new_logpx = logpws(ws, ρ, gp)
        dkl = D_kl(ws, ρ, gp)
        if (abs(new_logpx - old_logpx) < th && dkl < 1f-3)
            break
        end
        old_logpx = new_logpx
    end

    dkl = D_kl(ws, ρ, gp)
    if(dkl > 0.1f0)
        println("D_KL: ", dkl)
    end
    p_app = p_now(ws, gp)

    return p_app
end

function fd(β::Float32, w::Float32)
    return Float32(1/(exp(β*w) + 1))
end
function dfd(β::Float32, w::Float32)
    return Float32(1/(exp(β*w) + 1)/(exp(-β*w) + 1))
end
function fds(β::Float32, ws::Vector{Float32})
    return [fd(β, w) for w in ws]
end
function dfds(β::Float32, ws::Vector{Float32})
    return [dfd(β, w) for w in ws]
end

function check_xx(μ::Float32, β::Float32, k_size::Int)
    ph = H_params(0.2f0, μ, β, k_size, [range(-1.0f0,1.0f0, length=501)...])
    σxx_true = 0.0f0
    σxx_gmm = 0.0f0
    dfdv = dfds(ph.β, ph.ws)

    kxv = collect(Float32,-pi:2*pi/ph.K_size:pi)
    kyv = collect(Float32,-pi:2*pi/ph.K_size:pi)
    dk = 1.0f0/ph.K_size
    #=
    kx = Float32(pi/2)
    ky = Float32(pi/2)
    Hs = HandV([kx, ky], ph)
    ρw = ρwk_vec([kx, ky], ws, Hs.Hk, 0.02f0)
    σxx_true = dk^2*Hs.Vx^2*sum(ρw.^2 .* fdv)
    ρ_gmm = EM_fit(3000, 5, ph.ws, ρw)
    σxx_gmm = dk^2*Hs.Vx^2*sum(ρ_gmm.^2 .* fdv)=#
    
    for kx in kxv
        for ky in kyv
            Hs = HandV([kx, ky], ph)
            ρw = ρwk_vec([kx, ky], ws, Hs.Hk, 0.02f0)
            σxx_true += dk^2*Hs.Vx^2*sum(ρw.^2 .* dfdv)
            ρ_gmm = EM_fit(2000, 10, ph.ws, ρw)
            σxx_gmm += dk^2*Hs.Vx^2*sum(ρ_gmm.^2 .* dfdv)
        end
    end
    dkl = ρw'*[log(ρw[i]+1f-8)-log(ρ_gmm[i]+1f-8) for i in 1:length(ρw)]
    dkl *= (ph.ws[2] - ph.ws[1])

    return σxx_true, σxx_gmm, dkl
end