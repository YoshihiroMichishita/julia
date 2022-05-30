using "coreEnv.jl"
using Random


struct TableQAgt
    n_act::Int
    init_val_Q::Float64
    ϵ::Float64
    α::Float64
    γ::Float64
    max_memory::Int
    filepath::String
end

function init_tableQAgt(path::String)
    n_act::Int = 2
    init_val_Q::Float64 = 0.0
    ϵ::Float64 = 0.1
    α::Float64 = 0.1
    γ::Float64 = 0.9
    max_memory::Int = 500
    filepath::String = path

    return n_act, init_val_Q, ϵ, α, γ, max_memory, filepath
end

mutable struct Q_table
    Q::Dict
    len_Q::Int
end

function init_Qtable()
    Q = Dict()
    lenQ::Int = 0
end

function check_and_add_obs(qagt::TableQAgt, q::Q_table, obs::String)
    if(haskey(q.Q,obs))
        q.Q[obs] = [qagt.init_val_Q] * qagt.n_act
        q.len_Q += 1

        if(q.len_Q > qagt.max_memory)
            println("観測の登録数が上限"*string(qagt.max_memory)*"に達しました")
            exit()
        end

        if((q.len_Q < 100 && q.len_Q%10 == 0)||q.len_Q%100 == 0 )
            println("the number of Q-table "*string(q.lenQ))
        end
    end
end


function decide_action(qagt::TableQAgt, q::Q_table, obs)
    
    obs = string(obs)
    check_and_add_obs(qagt, q, obs)

    if(rand()< qagt.ϵ)
        act = rand(0:qagt.n_act)
    else
        act= findall(isequal(maximum(q.Q[obs])), q.Q[obs])
    end

    return act
end

