using Dates
using CUDA
using Distributions
using StatsBase

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
        input = make_image(env, game.history) |> gpu
        output = model(input) |> cpu
        policy_log = output[1:end-1, 1]
        legal = legal_action(env, game.history, game.branch_left)
        policy = softmax([policy_log[i] for i in legal])
        action = legal[rand(Categorical(policy))]
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

function play_physics!(env::Env, model::Chain, scores::Dict{Vector{Int}, Float32}, best_scores::Vector{Float32})
    game = init_agt()
    while(true)
        input = make_image(env, game.history) |> gpu
        output = model(input) |> cpu
        policy_log = output[1:end-1, 1]
        legal = legal_action(env, game.history, game.branch_left)
        policy = softmax([policy_log[i] for i in legal])
        action = legal[rand(Categorical(policy))]
        apply!(env, game, action)
        if(is_finish(env, game))
            break
        end
    end
    if(haskey(scores, game.history))
        game.reward = scores[game.history]
    else
        game.reward = calc_score(game.history, env)
        if(game.reward > best_scores[end])
            push!(best_scores, game.reward)
        else
            push!(best_scores, best_scores[end])
        end
        scores[game.history] = game.reward
    end
    
    return game
end

#cpu並列化予定
function sample_batch(env::Env, buffer::ReplayBuffer)
    games = sample(buffer.buffer, env.batch_size, replace=true)
    images_batch::Vector{Vector{Matrix{Int}}} = []
    legals_batch::Vector{Vector{Vector{Int}}} = []
    actions_batch::Vector{Vector{Int}} = []
    rewards_batch::Vector{Float32} = []
    for game in games
        l = length(game.history)
        image_batch::Vector{Matrix{Int}} = []
        legal_batch::Vector{Vector{Int}} = []
        branch_left::Vector{Int} = [-1]
        for turn in 1:l
            push!(image_batch, make_image(env, game.history, turn))
            legal = legal_action(env, game.history[1:turn], branch_left)
            push!(legal_batch, legal)
            if(game.history[turn] <= env.val_num)
                pop!(branch_left)
            elseif(game.history[turn] <= env.val_num+env.br_num)
                push!(branch_left, game.history[turn])
            end
        end
        push!(images_batch, reverse(image_batch))
        push!(legals_batch, reverse(legal_batch))
        push!(actions_batch, reverse(game.history))
        push!(rewards_batch, game.reward)
    end
    return images_batch, legals_batch, actions_batch, rewards_batch
end

#cpu並列化予定
function run_selfplay!(env::Env, buffer::ReplayBuffer, model::Chain, best_score::Vector{Float32})
    for it in 1:env.num_player
        game = play_physics!(env, model, buffer.scores, best_score)
        #print("#")
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

function step(x::Int)
    if(x>1)
        return 1
    else
        return 0
    end
end

function loss(env::Env, images_batch,legals_batch, actions_batch, rewards_batch, model::Chain, old_model::Chain)
    L::Float32 = 0.0f0
    
    for b in 1:env.batch_size
        val_next = 0.0f0
        adv = 0.0f0
        reward = rewards_batch[b]
        #hist = copy(game.history)
        for it in 1:length(actions_batch[b])
            image = images_batch[b][it] |> gpu
            y = model(image) |> cpu
            y_old = old_model(image) |> cpu

            policy_log = y[1:end-1, 1]
            val = y[end, 1]
            policy_log_old = y_old[1:end-1, 1]

            #legal = legal_action(env, actions_batch[b][1:end-it+1], game.branch_left)
            policy = softmax([policy_log[i] for i in legals_batch[b][it]])
            #policy_old = softmax([policy_log_old[i] for i in legal])

            delta = reward + env.γ * val_next - val

            adv = delta + env.γ * env.λ * adv
            a = actions_batch[b][it]
            rt = exp(step(it)*(policy_log[a,1]-policy_log_old[a,1]))
            
            L += min(rt*adv, clip(rt, env.ϵ)*adv) + delta^2  - step(it) * env.E * (policy' * log.(policy))
            #L += delta^2  - step(it)*env.E * (policy' * log.(policy))

            val_next = val
            reward = 0.0f0
        end
    end
    return L / env.batch_size
    # + env.C * sum(sqnorm, Flux.params(model))
end

tanh10(x) = Float32(12)*tanh(x/10)
tanh2(x) = Float32(4)*tanh(x/4)

#gpu並列化予定
function train_model!(env::Env, buffer::ReplayBuffer, model::Chain, old_model::Chain, opt)

    
    images_batch, legals_batch, actions_batch, rewards_batch = sample_batch(env, buffer)
    val, grads = Flux.withgradient(Flux.params(model)) do
        loss(env, images_batch, legals_batch, actions_batch, rewards_batch, model, old_model)
    end
    Flux.Optimise.update!(opt, Flux.params(model), grads)
    return val
end

function AlphaZero_ForPhysics(env::Env)
    ld = []
    max_hist::Vector{Float32} = [-12.0f0]

    #model = Chain(Dense(env.input_dim, env.middle_dim), BatchNorm(env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu),Dense(env.middle_dim, env.middle_dim, relu)), identity)) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), env.act_ind, tanh2)), Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), 1, tanh10)))) |> gpu
    model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(Dense(env.middle_dim, env.middle_dim, relu),Dense(env.middle_dim, env.middle_dim, relu)), identity)) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), env.act_ind, tanh2)), Chain(Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), 1, tanh10)))) |> gpu
    opt = Flux.Optimiser(WeightDecay(1f-6), Adam(env.η))

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

        
        ll = train_model!(env, replay_buffer, model, old_model, opt)
        push!(ld,ll)
        old_model = deepcopy(model)
    end
    
    return ld, max_hist, model, replay_buffer.scores
end

function AlphaZero_ForPhysics_hind(env::Env)
    ld = []
    max_hist::Vector{Float32} = [-12.0f0]

    #model = Chain(Dense(env.input_dim, env.middle_dim), BatchNorm(env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu),Dense(env.middle_dim, env.middle_dim, relu)), identity)) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), env.act_ind, tanh2)), Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), 1, tanh10)))) |> gpu
    model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(Dense(env.middle_dim, env.middle_dim, relu),Dense(env.middle_dim, env.middle_dim, relu)), identity)) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), env.act_ind, tanh2)), Chain(Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), 1, tanh10)))) |> gpu
    opt = Flux.Optimiser(WeightDecay(1f-6), Adam(env.η))

    replay_buffer = init_buffer(200, env.batch_size)

    old_model = deepcopy(model)

    for it in 1:env.num_simulation
        
        if(it%(env.num_simulation/10)==0)
            println("=============")
            println("it=$(it);")
            println("max score: $(max_hist[end])")
            #k = [keys(replay_buffer.scores)...]
            #inds = findall(s -> replay_buffer.scores[s] == max_hist[end], k)
            #for i in inds
            #    println("$(k[i])")
            #end
        end
        
        run_selfplay!(env, replay_buffer, model, max_hist)

        
        ll = train_model!(env, replay_buffer, model, old_model, opt)
        push!(ld,ll)
        old_model = deepcopy(model)
    end
    
    return ld, max_hist, model, replay_buffer.scores
end

#using BSON: @save
#using BSON: @load
using Plots
ENV["GKSwstype"]="nul"
Plots.scalefontsizes(1.3)

using DataFrames
using CSV

#=
using JLD2
using FileIO
=#
date = 1202

function PPO(args::Vector{String})
    #args = ARGS
    println("Start! at $(now())")
    env = init_Env(args)
    
    max_hists = []
    lds = []
    dist = 5
    for dd in 1:dist
        @time ld, max_hist, model, scores = AlphaZero_ForPhysics(env)
        push!(max_hists, max_hist)
        push!(lds, ld)

        println("search count: $(length(scores))")
        println("max score: $(max_hist[end])")
        k = [keys(scores)...]
        inds = findall(s -> scores[s] == max_hist[end], k)
        for i in inds
            println("$(k[i])")
        end
        for test in 1:5
            game = play_physics!(env, model, scores)
            println("History:$(hist2eq(game.history)), Reward:$(game.reward)")
        end
    end
    p0 = plot(max_hists[1],linewidth=3.0, gridwidth=2.0, xaxis=:log, xticks=([1,10,100,1000]), xlabel="iterate", yrange=(0,12), ylabel="max score", legend=:bottomright)
    for i in 2:dist
        p0 = plot!(max_hists[i], linewidth=3.0, xaxis=:log, yrange=(0,12))
    end
    savefig(p0, "./PPO_valMAX_itr_mt$(env.max_turn)_$(date).pdf")

    p1 = plot(lds[1], linewidth=3.0)
    for i in 2:dist
        p1 = plot!(lds[i], linewidth=3.0)
    end
    savefig(p1, "./PPO_loss_itr_mt$(env.max_turn)_$(date).png")

    save_data = DataFrame(hist1=max_hists[1][1:2000],hist2=max_hists[2][1:2000],hist3=max_hists[3][1:2000],hist4=max_hists[4][1:2000],hist5=max_hists[5][1:2000])
    CSV.write("./PPO_hists_mt$(env.max_turn)_$(date).csv", save_data)

    println("PPO Finish!")
end



@time PPO(ARGS)