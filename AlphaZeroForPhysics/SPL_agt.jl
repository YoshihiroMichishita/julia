

#include("AZP_env.jl")


#it stores sigle game history and child visit pi
mutable struct Agent
    #counts
    history::Vector{Int}
    surprise::Vector{Float32}
    branch_left::Vector{Int}
    child_visit_pi::Vector{Vector{Float32}}
    num_calc::Int
end

function init_agt()
    return Agent([], [], [-1], [], 0)
end

mutable struct Storage
    storage::Dict{Int, Chain}
    random_out::Chain
    scores::Dict{Vector{Int}, Float32}
    scores_size::Int
end

function init_storage(env, size::Int)
    return Storage(Dict(), Chain(Dense(zeros(Float32, env.output,env.input_dim))), Dict(), size)
end

function score_save!(storage::Storage, hist::Vector{Int}, score::Float32)
    if length(storage.scores) > storage.scores_size
        pop!(storage.scores, collect(keys(storage.scores))[end])
    end
    storage.scores[hist] = score
end

function latest_model(storage::Storage)
    if(isempty(storage.storage))
        return storage.random_out
    else
        return storage.storage[rand(keys(storage.storage))]
    end
end

function calc_score_his(history::Vector{Int}, env::Env, scores::Dict{Vector{Int}, Float32})
    if haskey(scores, history)
        return scores[history]
    else
        score = calc_score(history, env)
        scores[history] = score
        return score
    end
end

function calc_score_his(history::Vector{Int}, env::Env, storage::Storage)
    if haskey(storage.scores, history)
        return storage.scores[history]
    else
        score = calc_score(history, env)
        score_save!(storage, history, score)
        #scores[history] = score
        return score
    end
end

function make_target(env::Env,agt::Agent, scores::Dict{Vector{Int}, Float32}, turn::Int)
    return [agt.child_visit_pi[turn]; calc_score_his(agt.history, env, scores)]
end

function make_target(env::Env,agt::Agent, storage::Storage, turn::Int)
    return [agt.child_visit_pi[turn]; calc_score_his(agt.history, env, storage)]
end