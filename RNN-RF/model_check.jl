using Dates

include("MCTS-RF_env.jl")
include("MCTS-RF_agt.jl")

using BSON: @load

function make_image_from_vec(env::Env, state::Vector{Int})
    input_data = zeros(Int, env.input_dim, 1)
    for act_ind in 1:env.act_ind
        ind = findall(x->x==act_ind, state)
        for it in ind
            input_data[(act_ind-1)*env.max_turn+it,1] = 1
        end
    end
    return input_data
    #append!(copy(agt.history[1:turn]), zeros(Int, env.max_turn-turn))
end

function legal_action_vec(env::Env, state::Vector{Int})
    if(isempty(state))
        return [i for i in 1:env.act_ind]
    #elseif(env.max_turn-length(state)<=length(agt.branch_left)+1)
    #    return [i for i in 1:env.val_num]
    elseif(state[end]>env.val_num && state[end]<=env.val_num+env.br_num)
        return [i for i in 1:env.act_ind if(i!=state[end])]
    elseif(state[end]==6)
        return [i for i in 2:env.act_ind-1]
    else
        return [i for i in 1:env.act_ind]
    end
end

function main(args::Vector{String})
    println("Start! at $(now())")
    env = init_Env(args)
    @load "/home/yoshihiro/Documents/Codes/julia/RNN-RF/AlphaZero_ForPhysics_new.bson" model
    for it in 1:env.max_turn
        state::Vector{Int} = parse.(Int, split(readline()))
        if(isempty(state))
            break
        end
        input = make_image_from_vec(env, state)
        println("=================")
        println("input: $(input)")
        output = model(input)
        println("output: $(output)")
        value = output[end] 
        pol_log = output[1:end-1]
        A = legal_action_vec(env, state)
        println("legal_action: $(A)")
        policy = softmax([pol_log[a] for a in A])
        println("value: $(value),   policy: $(policy)")
    end
    score_test()
end

main(ARGS)