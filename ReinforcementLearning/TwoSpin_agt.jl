using Flux
include("TwoSpin_env.jl")


struct agtQ
    in_size::Int
    out_size::Int
    n_dense::Int
    ϵ::Float64
    γ::Float64
end

function init_nQ(en::TS_env)
    in_size::Int = en.num_parm + en.HS_size^2 + 1
    out_size::Int = en.HS_size^2
    n_dense::Int = 32
    ϵ::Float64 = 0.1
    γ::Float64 = 0.9

    return in_size, out_size, n_dense, ϵ, γ
end

mutable struct models
    model
    opt
    loss
end

function build_model(nq::agtQ)
    #model = Chain(Flux.flatten', Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))
    model = Chain(Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_dense, relu), Dense(nq.n_dense, nq.out_size))
    opt = ADAM()
    loss(x,y) = Flux.mse(model(x),y)

    return model, opt, loss
end

function get_U(m::models , obs)
    U = m.model(Flux.flatten(obs)')
    return U
end


using RandomMatrices

function decide_action(nq::netQAgt, m::models, obs)

    if(rand()< nq.ϵ)
        her = GaussianHermite(2)
        U = rand(her,4)
        act = matrix_to_vec(U)
    else
        act = get_U(m, obs)
    end

    return act
end