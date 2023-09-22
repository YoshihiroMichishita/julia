using Dates
using CUDA

include("PPO_env.jl")
include("PPO_agt.jl")

#1局1局の情報をストックする
mutable struct ReplayBuffer
    buffer::Vector{Agent}
    buffer_size::Int
    batch_size::Int
    scores::Dict{Vector{Int}, Float32}
    #count::Int
end

function init_buffer(buffer_size::Int, batch_size::Int)
    return ReplayBuffer([], buffer_size, batch_size, Dict{Vector{Int}, Float32}())
end

function save_game!(buffer::ReplayBuffer, agt::Agent)
    if length(buffer.buffer) > buffer.buffer_size
        popfirst!(buffer.buffer)
    end
    push!(buffer.buffer, agt)
end

function play_physics!(env::Env, model::Chain, scores::Dict{Vector{Int}, Float32})
    game = init_agt()
    while(true)
        input = make_image(env, game) |> gpu
        output = model(input) |> cpu
        policy_log = output[1:end-1, 1]
        legal = legal_action(env, game.history, game.branch_left)
        policy = softmax([policy_log[i] for i in legal])
        action = legal[sample(Categorical(policy))]
        apply!(env, game, action)
        if(is_finish(env, game))
            break
        end
    end
    if(haskey(scores, game.history))
        game.reward = scores[game.history]
    else
        game.reward = calc_score(game.history, env)
        scores[game.history] = game.reward
    end
    
    return game
end

#cpu並列化予定
function sample_batch(env::Env, buffer::ReplayBuffer)
    games = sample(buffer.buffer, env.batch_size, replace=true)
    return games
end


#cpu並列化予定
function run_selfplay!(env::Env, buffer::ReplayBuffer, model::Chain, best_score::Vector{Float32})
    for it in 1:env.num_player
        game = play_physics!(env, model, buffer.scores)
        if(game.reward > best_score[end])
            push!(best_score, game.reward)
        else
            push!(best_score, best_score[end])
        end
        save_game!(buffer, game)
    end
end

function clip(rt::Float32, ϵ::Float32)
    if(rt > 1.0f0 + ϵ)
        return 1.0f0 + ϵ
    elseif(rt < 1.0f0 - ϵ)
        return 1.0f0 - ϵ
    else
        return rt
    end
end

function loss(env::Env, games::Vector{Agent}, model::Chain, old_model::Chain)
    L::Float32 = 0.0f0
    
    for game in games
        val_next = 0.0f0
        adv = 0.0f0
        hist = copy(game.history)
        for it in 1:length(hist)
            image = make_image(env, hist) |> gpu
            a = pop!(hist)
            y = model(image) |> cpu
            y_old = old_model(image) |> cpu
            policy_log = y[1:end-1]
            val = y[end]
            policy_log_old = y_old[1:end-1]

            legal = legal_action(env, game.history, game.branch_left)
            policy = softmax([policy_log[i] for i in legal])
            policy_old = softmax([policy_log_old[i] for i in legal])

            delta = game.reward + env.γ * val_next - val

            adv = delta + env.γ * env.λ * adv
            rt = policy[a]/policy_old[a]
            
            L += min(rt*adv, clip(rt, env.ϵ)*adv) + delta^2  - env.E * (policy' * log.(policy))

            val_next = val
        end
    end
    return L / env.batch_size
    # + env.C * sum(sqnorm, Flux.params(model))
end



tanh10(x) = Float32(12)*tanh(x/10)
tanh2(x) = Float32(4)*tanh(x/4)

#gpu並列化予定
function train_model!(env::Env, buffer::ReplayBuffer, model::Chain, old_model::Chain, opt)

    
    games = sample_batch(env, buffer)
    val, grads = Flux.withgradient(Flux.params(model)) do
        loss(env, games, model, old_model)
    end
    Flux.Optimise.update!(opt, Flux.params(model), grads)
    return val
end




function AlphaZero_ForPhysics(env::Env)
    ld = []
    max_hist::Vector{Float32} = [-12.0f0]

    model = Chain(Dense(env.input_dim, env.middle_dim), BatchNorm(env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu),Dense(env.middle_dim, env.middle_dim, relu)), identity)) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), env.act_ind, tanh2)), Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), 1, tanh10)))) |> gpu

    replay_buffer = init_buffer(1000, env.batch_size)

    old_model = deepcopy(model)

    for it in 1:env.num_simulation
        if(it%(env.num_simulation/10)==0)
            println("=============")
            println("it=$(it);")
            println("max score: $(max_hist[end])")
            k = [keys(replay_buffer.scores)...]
            inds = findall(s -> replay_buffer.scores[s] == max_hist[end], k)
            for i in inds
                println("$(k[i])")
            end
        end
        
        run_selfplay!(env, replay_buffer, model, max_hist)

        opt = Flux.Optimiser(WeightDecay(env.C), Adam(env.η))
        ll = train_model!(env, replay_buffer, model, old_model, opt)
        push!(ld,ll)
        old_model = deepcopy(model)
    end
    
    return ld, max_hist, model
end

using BSON: @save
using BSON: @load
using Plots
ENV["GKSwstype"]="nul"

using JLD2
using FileIO

date = 922

function main(args::Vector{String})
    #args = ARGS
    println("Start! at $(now())")
    env = init_Env(args)
    
    max_hists = []
    for dd in 1:1
        ld, max_hist, model = AlphaZero_ForPhysics(env)
        push!(max_hists, max_hist)

        for test in 1:10
            game = play_physics!(env, model)
            println("History:$(hist2eq(game.history)), Reward:$(game.reward)")
        end
    end
    p0 = plot(max_hists[1], linewidth=3.0)
    for i in 2:length(max_hists)
        p0 = plot!(max_hists[i], linewidth=3.0)
    end
    savefig(p0, "/Users/johnbrother/Documents/Codes/julia/AlphaZeroForPhysics/PPO_valMAX_itr_mt$(env.max_turn)_$(date).png")
    println("PPO Finish!")
end



@time main(ARGS)