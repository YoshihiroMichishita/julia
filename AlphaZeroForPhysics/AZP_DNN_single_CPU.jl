using Dates

#include("AZP_env.jl")
include("AZP_env.jl")
include("AZP_agt.jl")
include("AZP_mcts_single_CPU.jl")

#1局1局の情報をストックする
mutable struct ReplayBuffer
    buffer::Vector{Agent}
    buffer_size::Int
    batch_size::Int
    scores::Dict{Vector{Int}, Float32}
    #count::Int
end

function init_buffer(buffer_size::Int, batch_size::Int)
    return ReplayBuffer([], buffer_size, batch_size, Dict())
end

function save_game!(buffer::ReplayBuffer, agt::Agent)
    if length(buffer.buffer) > buffer.buffer_size
        popfirst!(buffer.buffer)
    end
    push!(buffer.buffer, agt)
end

mutable struct Storage
    storage::Dict{Int, Chain}
    random_out::Chain
    scores::Dict{Vector{Int}, Float32}
end

function init_storage(env)
    return Storage(Dict(), Chain(Dense(zeros(Float32, env.output,env.input_dim))), Dict())
end

function latest_model(storage::Storage)
    if(isempty(storage.storage))
        return storage.random_out
    else
        return storage.storage[rand(keys(storage.storage))]
    end
end

#=
function WeightSample(hist::Vector{Int})
    s = [i for i in 1:length(hist)]
    ww = s.^2
    ww = ww/sum(ww)
    return sample(s, ProbabilityWeights(ww))
end=#

function WeightSample(hist::Vector{Int})
    s = [i for i in 1:length(hist)]
    w = [2^i for i in 1:length(hist)]
    ww = w/sum(w)
    return sample(s, ProbabilityWeights(ww))
end
function WeightSample(agt::Agent)
    s = [i for i in 1:length(agt.history)]
    w = agt.surprise
    ww = w/sum(w)
    return sample(s, ProbabilityWeights(ww))
end

#cpu並列化予定
function sample_batch!(env::Env, buffer::ReplayBuffer, scores::Dict{Vector{Int}, Float32})
    games = sample(buffer.buffer, weights([length(agt.history) for agt in buffer.buffer]), env.batch_size, replace=true)
    g_turn = [(g, WeightSample(g.history)) for g in games]

    if(isempty(scores))
        imag = zeros(Int, env.input_dim, env.batch_size)
        target = zeros(Float32, env.output, env.batch_size)
        for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, scores, turn)
        end
    else
        imag = zeros(Int, env.input_dim, 4env.batch_size)
        target = zeros(Float32, env.output, 4env.batch_size)
        for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, scores, turn)
        end
        for it in 1:3env.batch_size
            hist = rand(keys(scores))
            imag[:,env.batch_size+it] = make_image(env, hist)
            target[end,env.batch_size+it] = scores[hist]
        end
    end

    tar_data = copy(target)
    for it in 1:env.batch_size
        g, turn = g_turn[it]
        for l in 1:length(g.history)
            his = g.history[1:l]
            if(haskey(scores, his))
                if(scores[his] < tar_data[end,it])
                    #println("score_data:$(scores[his]), tar_data:$(tar_data[end,it]), max:$(max(scores[his], tar_data[end,it]))")
                    scores[his] = tar_data[end,it]
                end
                #max(scores[his], tar_data[end,it])
            else
                scores[his] = tar_data[end,it]
            end
        end
    end
    return imag, tar_data
end


#cpu並列化予定
function run_selfplay(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32)
    model = latest_model(storage)
    for it in 1:env.num_player
        #=
        if(it%(env.num_player/10)==0)
            print("#")
        end=#
        #model = gpu(latest_model(storage))
        game = play_physics!(env, model, ratio, noise_r)
        save_game!(buffer, game)
    end
end

function run_selfplay!(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32, max_hist::Vector{Float32})
    model = latest_model(storage)
    for it in 1:env.num_player
        #=
        if(it%(env.num_player/10)==0)
            print("#")
        end=#
        #model = gpu(latest_model(storage))
        game = play_physics!(env, model, ratio, noise_r, storage.scores, max_hist)
        save_game!(buffer, game)
    end
end


function loss(image::Matrix{Int}, target::Matrix{Float32}, env::Env, model::Chain)
    y1 = model(image)
    return sum([(((y1[end,i]-target[end,i]))^2 - target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]))) for i in 1:env.batch_size])/env.batch_size
    # + env.C * sum(sqnorm, Flux.params(model))
end

function loss_check(image::Matrix{Int}, target::Matrix{Float32}, env::Env, model::Chain)
    y1 = model(image)
    val = sum([(((y1[end,i]-target[end,i]))^2) for i in 1:env.batch_size])/env.batch_size
    pol = sum([(-target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]))) for i in 1:env.batch_size])/env.batch_size
    return val, pol
    # + env.C * sum(sqnorm, Flux.params(model))
end


tanh10(x) = Float32(12)*tanh(x/10)
tanh2(x) = Float32(4)*tanh(x/4)

#gpu並列化予定
function train_model!(env::Env, buffer::ReplayBuffer, storage::Storage)
    #ll = zeros(Float32, env.batch_num)
    ll = zeros(Float32, env.batch_num, env.training_step)
    for b_num in 1:env.batch_num
        if(haskey(storage.storage, b_num))
            model = storage.storage[b_num]
        else
            #model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu)),Dense(env.middle_dim, env.middle_dim, relu)), identity) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(Dense(env.middle_dim, div(env.middle_dim,4), relu), Dense(div(env.middle_dim,4), env.act_ind, tanh2)), Chain(Dense(env.middle_dim, div(env.middle_dim,4), relu), Dense(div(env.middle_dim,4), 1, tanh10)))) |> gpu
            model = Chain(Dense(env.input_dim, env.middle_dim), BatchNorm(env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu),Dense(env.middle_dim, env.middle_dim, relu)), identity)) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), env.act_ind, tanh2)), Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), 1, tanh10))))
            #model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu)),Dense(env.middle_dim, env.middle_dim, relu)), identity) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(Dense(env.middle_dim, env.middle_dim, relu), Dense(env.middle_dim, env.act_ind, tanh2)), Chain(Dense(env.middle_dim, env.middle_dim, relu), Dense(env.middle_dim, 1, tanh10)))) |> gpu
        end
        opt = Flux.Optimiser(WeightDecay(env.C), Adam(2f-5))
        #ParameterSchedulers.Scheduler(env.scheduler, Momentum())
        for it in 1:env.training_step
            if(it%(env.checkpoint_interval)==0)
                opt = Flux.Optimiser(WeightDecay(env.C), Adam(2f-5))
            end
            image_batch, target_batch = sample_batch!(env, buffer, storage.scores)
            val, grads = Flux.withgradient(Flux.params(model)) do
                loss(image_batch,target_batch,env,model)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            ll[b_num, it] = val
            #=
            if(it > env.training_step-6)
                val, pol = loss_check(image_batch,target_batch,env,model)
                println("val=$(val), pol=$(pol)")
            end=#
        end
        storage.storage[b_num] = model
    end
    return ll
end




function AlphaZero_ForPhysics(env::Env, envf::Env, storage::Storage)
    ld = []
    max_hist::Vector{Float32} = [-12.0f0]
    itn = 3
    lastit = 0
    ratio = env.ratio
    randr = env.ratio_r
    for it in 1:itn
        println("=============")
        println("it=$(it);")

        replay_buffer = init_buffer(1200, env.batch_size)
        
        if(it<5)
            #ratio -= Float32(1.0)
            #randr -= Float32(1.0f-1)
            #@time run_selfplay_palss(env, replay_buffer, storage, ratio, randr)
            @time run_selfplay!(env, replay_buffer, storage, ratio, randr, max_hist)
            #@time run_selfplay_pals(env, replay_buffer, storage, ratio, 1.0f0)
            @time ll = train_model!(env, replay_buffer, storage)
        else
            #ratio -= Float32(2.0)
            #randr -= Float32(1.0f-1)
            #ratio = Float32(6.0)
            #@time run_selfplay_palss(env, replay_buffer, storage, ratio, randr)
            @time run_selfplay!(env, replay_buffer, storage, ratio, randr, max_hist)
            @time ll = train_model!(env, replay_buffer, storage)
        end
        #@report_call run_selfplay(env, replay_buffer, storage)
        #ll = @report_call train_model!(env, replay_buffer, storage)
        #println("loss_average: $(ll)")
        push!(ld,ll)
        println("store data")
        println(length(storage.scores))
        #if(it%2==0)
        for bb in 1:env.batch_num
            model0 = storage.storage[bb]
            println("------------")
            println("head = $(bb);")
            for tes in 1:5
                game = play_physics!(envf, model0)
                score = calc_score(game.history, envf)
                val = model0(make_image(envf, game, length(game.history)))[end, 1]
                println("$(hist2eq(game.history)), score:$(score), val(NN):$(val)")
            end
        end
        val = findmax(storage.scores)[1]
        println("max score: $(val)")
        #if(max_hist[end]>10.0f0)
        #    break
        #end
    end
    
    return ld, max_hist, latest_model(storage)
end

function AlphaZero_ForPhysics_hind(env::Env, storage::Storage)
    ld = []
    max_hist::Vector{Float32} = [-12.0f0]
    itn = 2
    ratio = env.ratio
    randr = env.ratio_r
    for it in 1:itn
        replay_buffer = init_buffer(1200, env.batch_size)
        
        run_selfplay!(env, replay_buffer, storage, ratio, randr, max_hist)
        ll = train_model!(env, replay_buffer, storage)

        push!(ld,ll)
    end
    
    return max_hist
end

function dict_copy(orig::Dict{Vector{Int}, Float32})
    c_dict = Dict{String, Float32}()
    for k in keys(orig)
        c_dict["$(k)"] = orig[k] 
    end
    return c_dict
end

#=
using BSON: @save
using BSON: @load
using Plots
ENV["GKSwstype"]="nul"

using JLD2
using FileIO

date = 1004

function main(args::Vector{String})
    #args = ARGS
    println("Start! at $(now())")
    env = init_Env(args)
    env_fc = init_Env_forcheck(args)
    lds = []
    max_hists = []
    for dd in 1:5
        storage = init_storage(env)
        ld, max_hist, model = AlphaZero_ForPhysics(env, env_fc, storage)
        push!(max_hists, max_hist)
        push!(lds, ld)
        #string_score = dict_copy(storage.scores)
        k = [keys(storage.scores)...]
        inds = findall(s -> storage.scores[s] == findmax(storage.scores)[1], k)
        println("max score:")
        for i in inds
            println("$(hist2eq(k[i])), $(storage.scores[k[i]])")
        end
    end
    p0 = plot(max_hists[1], linewidth=3.0)
    for i in 2:length(max_hists)
        p0 = plot!(max_hists[i], linewidth=3.0)
    end
    #savefig(p0, "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/valMAX_itr_mt$(env.max_turn)_$(date).png")
    savefig(p0, "/Users/johnbrother/Documents/Codes/julia/AlphaZeroForPhysics/valMAX_itr_mt$(env.max_turn)_$(date).png")
    
    p = plot(lds[1], yaxis=:log, linewidth=3.0)
    for i in 2:length(lds)
        plot!(lds[i], yaxis=:log, linewidth=3.0)
    end
    #savefig(p, "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/loss_valMAX_mt$(env.max_turn)_$(date).png")
    savefig(p, "/Users/johnbrother/Documents/Codes/julia/AlphaZeroForPhysics/loss_valMAX_mt$(env.max_turn)_$(date).png")
    
    println("AlphaZero Finish!")
    #println("loss-dynamics: $(ld)")
    #=
    for head in 1:env.batch_num
        model0 = storage.storage[head] |> cpu
        @save "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/AZP_valMAX_mt$(env.max_turn)_$(date)_head$(st+head).bson" model0
    end=#

    #=
    for it in 1:10
        game = play_physics!(env_fc, model)
        score = calc_score(game.history, env_fc)
        println("$(game.history), score:$(score)")
    end

    string_score = dict_copy(storage.scores)
    k = [keys(string_score)...]
    inds = findall(s -> string_score[s] == findmax(string_score)[1], k)
    println("max score:")
    for i in inds
        println("$(k[i]), $(string_score[k[i]])")
    end
    
    println("check")
    h = [3, 6, 2, 6, 4, 6, 2, 1]
    println("$(h), $(calc_score(h, env_fc))")
    h = [3, 6, 2, 6, 4, 1, 6, 2]
    println("$(h), $(calc_score(h, env_fc))")
    h = [6, 3, 2, 6, 4, 2, 1]
    println("$(h), $(calc_score(h, env_fc))")
    h = [6, 3, 6, 4, 5, 2, 1, 1, 2]
    println("$(h), $(calc_score(h, env_fc))")
    
    if(st == 0)
        save("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores_mt$(env.max_turn)_$(date).jld2", string_score)
    else
        save("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores_new.jld2", string_score)
    end=#
end



@time main(ARGS)=#