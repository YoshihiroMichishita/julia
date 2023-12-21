

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
    scores::Dict{Vector{Int}, Float32}
    scores_size::Int
    tp_trees::Dict{Vector{Int}, Float32}
end

function init_storage(size::Int)
    return Storage(Dict(), size, Dict())
end

function score_save!(storage::Storage, hist::Vector{Int}, score::Float32)
    if length(storage.scores) > storage.scores_size
        pop!(storage.scores, collect(keys(storage.scores))[end])
    end
    storage.scores[hist] = score
end

function tp_trees_update!(storage::Storage, hist::Vector{Int}, score::Float32)
    if(length(storage.tp_trees)=5)
        pop!(storage.tp_trees, collect(keys(storage.tp_trees))[end])
    end
    storage.tp_trees[hist] = score
    sort!(storage.tp_trees,)
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