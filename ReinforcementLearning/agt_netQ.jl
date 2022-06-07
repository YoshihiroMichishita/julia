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
#=
function init_nQ()
    n_act::Int = 3
    input_size::Int = 4
    n_dense::Int = 32
    ϵ::Float64 = 0.1
    γ::Float64 = 0.9
    filepath::String = "test"

    return n_act, input_size, n_dense, ϵ, γ, filepath
end=#

function init_nQ(s::String = "test")
    n_act::Int = 3
    input_size::Int = 4
    n_dense::Int = 32
    ϵ::Float64 = 0.1
    γ::Float64 = 0.9
    filepath::String = s

    return n_act, input_size, n_dense, ϵ, γ, filepath
end
#=
function model(nq::netQAgt, x)
    m = Chain(Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))
    return m(x)
end

function model_f(nq::netQAgt, x)
    m = Chain(Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))
    return m(Flux.flatten(x)')
end=#

mutable struct models
    model
    opt
    loss
end

function build_model(nq::netQAgt)
    #model = Chain(Flux.flatten', Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))
    model = Chain(Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))
    opt = ADAM()
    loss(x,y) = Flux.mse(model(x),y)

    return model, opt, loss
end

function get_Q(m::models , obs)
    Q = m.model(Flux.flatten(obs)')
    return Q
end

function decide_action(nq::netQAgt, m::models, obs)

    if(rand()< nq.ϵ)
        act = rand(0:nq.n_act)
    else
        Q = get_Q(m, obs)
        act= argmax(Q[1])
    end

    return act
end

function learn(nq::netQAgt, m::models, obs, act, rwd, done, next_obs)
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

    Flux.train!(m.loss,Flux.params(m.model),obs, m.opt)
end

using BSON: @save
using BSON: @load

function save_weight(nq::netQAgt, m::models, filepath=nothing)
    if(isnothing(filepath))
        filepath = nq.filepath * ".bson"
    end
    x = m.model |> cpu
    @save filepath x
end

function load_weight(nq::netQAgt, m::models, filepath=nothing)
    if(isnothing(filepath))
        filepath = nq.filepath*".bson"
    end
    x = Chain(Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))

    @load filepath x
    m.model = x
end

function main_nq()
    nq = netQAgt(init_nQ()...)
    m = models(build_model(nq)...)
    obs = ones(Int,4)
    act = decide_action(nq, m, obs)

    println("act: "*string(act))

    rwd::Float64 =1.0
    done = false
    next_obs = [1, 1, 1, 2]
    learn(obs, act, rwd, done, next_obs)

    println(m.model)

    save_weight(nq, m)
    load_weight(nq, m)

    y = m.model(obs)
    println(y)
end


@time main_nq()