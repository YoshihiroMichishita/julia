using Flux

include("coreEnv.jl")

struct netQAgt
    n_act::Int
    input_size::Int
    n_dense::Int
    ϵ::Float64
    γ::Float64
    filepath::String
end

function init_nQ()
    n_act::Int = 3
    input_size::Int = 4
    n_dense::Int = 32
    ϵ::Float64 = 0.1
    γ::Float64 = 0.9
    filepath::String = "test"

    return n_act, input_size, n_dense, ϵ, γ, filepath
end

function model(nq::netQAgt, x)
    m = Chain(Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))
    return m(x)
end

function model_f(nq::netQAgt, x)
    m = Chain(Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))
    return m(Flux.flatten(x)')
end

mutable struct models
    model
    opt
    loss
end

function build_Qnet(nq::netQAgt)
    model = Chain(Flux.flatten, Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))
    opt = ADAM()
    loss(x,y) = Flux.mse(model(x),y)

    return model
end

function get_Q(nq::netQAgt, obs)
    Q = model(nq, model)
    return Q
end

function decide_action(nq::netQAgt, q::Q_table, obs)

    if(rand()< nq.ϵ)
        act = rand(0:nq.n_act)
    else
        q.Q = get_Q(obs)
        act= argmax(q.Q)
    end

    return act
end

function learn(nq::netQAgt, obs, act, rwd, done, next_obs)
    if(isnothing(rwd))
        return
    end

    y = get_Q(obs)
    target = copy(y)

    if(!done)
        next_y = get_Q(next_obs)
        target_act = rwd + nq.γ*maximum(next_y)
    else
        target_act = rwd
    end

    target[act] = target_act
end

