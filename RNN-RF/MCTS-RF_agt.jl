using Flux
include("MCTS-RF_env.jl")

mutable struct Agent
    #counts
    Q::Matrix{Float32}
    N::Matrix{Int}
    W::Matrix{Float32}
    P::Matrix{Float32}
end

function init_agt(env::Env)
    Q = zeros(Float32, env.act_ind, env.input_dim)
    N = zeros(Int, env.act_ind, env.input_dim)
    W = zeros(Float32, env.act_ind, env.input_dim)
    P = zeros(Float32, env.act_ind, env.input_dim)

    return Agent(Q, N, W, P)
end

function loss(out::Vector{Float32}, target::Vector{Float32}, env::Env, model)
    return (out[end]-target[end])^2 - target[1:end-1]' * log(out[1:end-1]) + env.C * sum(Flux.params(model)^2)
end

function s_rate(env::Env, agt::Agent, ind_s::Int)
    return log((1 + sum(agt.N[:,ind_s]) + env.Cb)/env.Cb) + env.Ci
end

function UCB(env::Env, agt::Agent, ind_s::Int, ind_a::Int)
    return agt.Q[ind_a, ind_s] + s_rate(env, agt, ind_s) * sqrt(sum(agt.N[:,ind_s])) / (1 + agt.N[ind_a, ind_s])
end

function decide_action!(env::Env, agt::Agent, ind_s::Int)
    act = argmax(UCB(env, agt, ind_s, i) for i in 1:env.act_ind)
    agt.N[act, ind_s] += 1
    return act
end

function play_physics!(env::Env, agt::Agent)
    st = zeros(Float32, env.input_dim)
    for turn in 1:env.max_turn
        
    end
end