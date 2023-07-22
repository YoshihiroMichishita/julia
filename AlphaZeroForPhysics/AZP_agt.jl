

include("AZP_env.jl")


#it stores sigle game history and child visit pi
mutable struct Agent
    #counts
    history::Vector{Int}
    branch_left::Vector{Int}
    child_visit_pi::Vector{Vector{Float32}}
end

function init_agt()
    return Agent([], [-1], [])
end

function make_image(env::Env, agt::Agent, turn::Int)
    input_data = zeros(Int, env.input_dim, 1)
    for act_ind in 1:env.act_ind
        ind = findall(x->x==act_ind, agt.history[1:turn])
        for it in ind
            input_data[(act_ind-1)*env.max_turn+it,1] = 1
        end
    end
    return input_data
    #append!(copy(agt.history[1:turn]), zeros(Int, env.max_turn-turn))
end
function make_image(env::Env, history::Vector{Int}, turn::Int)
    input_data = zeros(Int, env.input_dim, 1)
    for act_ind in 1:env.act_ind
        ind = findall(x->x==act_ind, history[1:turn])
        for it in ind
            input_data[(act_ind-1)*env.max_turn+it,1] = 1
        end
    end
    return input_data
end
function make_image(env::Env, history::Vector{Int})
    input_data = zeros(Int, env.input_dim, 1)
    for act_ind in 1:env.act_ind
        ind = findall(x->x==act_ind, history)
        for it in ind
            input_data[(act_ind-1)*env.max_turn+it,1] = 1
        end
    end
    return input_data
end

function make_target(env::Env,agt::Agent, turn::Int)
    return [agt.child_visit_pi[turn]; calc_score(agt.history, env)]
end

function calc_score_his(history::Vector{Int}, env::Env, scores::Dict{Vector{Int}, Float32})
    if haskey(scores, history)
        return scores[history]
    else
        score = calc_score(history, env)
        #scores[history] = score
        return score
    end
end

function make_target(env::Env,agt::Agent, scores::Dict{Vector{Int}, Float32}, turn::Int)
    return [agt.child_visit_pi[turn]; calc_score_his(agt.history, env, scores)]
end
