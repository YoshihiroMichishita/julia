using Distributed
using Dates
using SharedArrays
#using JET
addprocs(12)

@everywhere include("AZP_env.jl")
@everywhere include("AZP_agt.jl")
@everywhere include("AZP_mcts_valMAX.jl")

#1局1局の情報をストックする
mutable struct ReplayBuffer
    buffer::Vector{Agent}
    buffer_size::Int
    batch_size::Int
    #count::Int
end

function init_buffer(buffer_size::Int, batch_size::Int)
    return ReplayBuffer([], buffer_size, batch_size)
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
    #g_turn = [(g, rand(1:length(g.history))) for g in games]

    #imag = SharedArray(zeros(Int, env.input_dim, buffer.batch_size))
    #target = SharedArray(zeros(Float32, env.output, buffer.batch_size))
    if(isempty(scores))
        imag = SharedArray(zeros(Int, env.input_dim, env.batch_size))
        target = SharedArray(zeros(Float32, env.output, env.batch_size))
        @sync @distributed for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, scores, turn)
        end
    else
        imag = SharedArray(zeros(Int, env.input_dim, 4env.batch_size))
        target = SharedArray(zeros(Float32, env.output, 4env.batch_size))
        @sync @distributed for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, scores, turn)
        end
        @sync @distributed for it in 1:3env.batch_size
            hist = rand(keys(scores))
            imag[:,env.batch_size+it] = make_image(env, hist)
            target[end,env.batch_size+it] = scores[hist]
        end
    end

    tar_data = sdata(target)
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
    return sdata(imag), tar_data
end

function sample_batch_s!(env::Env, buffer::ReplayBuffer, scores::Dict{Vector{Int}, Float32})
    games = sample(buffer.buffer, weights([length(agt.history) for agt in buffer.buffer]), env.batch_size, replace=true)
    g_turn = [(g, WeightSample(g)) for g in games]
    #g_turn = [(g, rand(1:length(g.history))) for g in games]

    #imag = SharedArray(zeros(Int, env.input_dim, buffer.batch_size))
    #target = SharedArray(zeros(Float32, env.output, buffer.batch_size))
    if(isempty(scores))
        imag = SharedArray(zeros(Int, env.input_dim, env.batch_size))
        target = SharedArray(zeros(Float32, env.output, env.batch_size))
        @sync @distributed for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, scores, turn)
        end
    else
        imag = SharedArray(zeros(Int, env.input_dim, 4env.batch_size))
        target = SharedArray(zeros(Float32, env.output, 4env.batch_size))
        @sync @distributed for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, scores, turn)
        end
        @sync @distributed for it in 1:3env.batch_size
            hist = rand(keys(scores))
            imag[:,env.batch_size+it] = make_image(env, hist)
            target[end,env.batch_size+it] = scores[hist]
        end
    end

    tar_data = sdata(target)
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
    return sdata(imag), tar_data
end

#cpu並列化予定
function run_selfplay(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32)
    model = latest_model(storage) |> gpu
    for it in 1:env.num_player
        if(it%(env.num_player/10)==0)
            print("#")
        end
        #model = gpu(latest_model(storage))
        game = play_physics!(env, model, ratio)
        save_game!(buffer, game)
    end
end

function run_selfplay(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32)
    model = latest_model(storage) |> gpu
    for it in 1:env.num_player
        if(it%(env.num_player/10)==0)
            print("#")
        end
        #model = gpu(latest_model(storage))
        game = play_physics!(env, model, ratio, noise_r)
        save_game!(buffer, game)
    end
end

@everywhere function run_selfplay_worker(env::Env, model::Chain, ratio::Float32, noise_r::Float32)
    games = Agent[]
    for it in 1:div(env.num_player, nworkers())
        game = play_physics!(env, model, ratio, noise_r)
        push!(games, game)
    end
    return games
end

@everywhere function run_selfplay_worker(env::Env, model::Chain, ratio::Float32, noise_r::Float32, scores::Dict{Vector{Int}, Float32})
    games = Agent[]
    for it in 1:div(env.num_player, nworkers())
        game = play_physics!(env, model, ratio, noise_r, scores)
        push!(games, game)
    end
    return games
end

@everywhere function run_selfplay_worker_withS(env::Env, model::Chain, ratio::Float32, noise_r::Float32, scores::Dict{Vector{Int}, Float32})
    games = Agent[]
    for it in 1:div(env.num_player, nworkers())
        game = play_physics_s!(env, model, ratio, noise_r, scores)
        push!(games, game)
    end
    return games
end

function run_selfplay_pal(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32)
    model = latest_model(storage) |> gpu
    futures = Future[]
    for i in workers()
        push!(futures, remotecall(run_selfplay_worker, i, env, model, ratio, noise_r))
    end
    for f in futures
        games = fetch(f)
        for g in games
            save_game!(buffer, g)
        end
    end
end

function run_selfplay_pals(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32)
    model = latest_model(storage) |> gpu
    futures = Future[]
    for i in workers()
        push!(futures, remotecall(run_selfplay_worker, i, env, model, ratio, noise_r, storage.scores))
    end
    for f in futures
        games = fetch(f)
        for g in games
            save_game!(buffer, g)
        end
    end
end

function run_selfplay_palss(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32)
    model = latest_model(storage) |> gpu
    futures = Future[]
    for i in workers()
        push!(futures, remotecall(run_selfplay_worker_withS, i, env, model, ratio, noise_r, storage.scores))
    end
    for f in futures
        games = fetch(f)
        for g in games
            save_game!(buffer, g)
        end
    end
end

function loss(image::CuArray{Int, 2}, target::Matrix{Float32}, env::Env, model::Chain)
    y1 = cpu(model(image))
    return sum([(((y1[end,i]-target[end,i]))^2 - target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]))) for i in 1:env.batch_size])/env.batch_size
    # + env.C * sum(sqnorm, Flux.params(model))
end

function loss_check(image::CuArray{Int, 2}, target::Matrix{Float32}, env::Env, model::Chain)
    y1 = cpu(model(image))
    val = sum([(((y1[end,i]-target[end,i]))^2) for i in 1:env.batch_size])/env.batch_size
    pol = sum([(-target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]))) for i in 1:env.batch_size])/env.batch_size
    return val, pol
    # + env.C * sum(sqnorm, Flux.params(model))
end


@everywhere tanh10(x) = Float32(12)*tanh(x/10)
@everywhere tanh2(x) = Float32(4)*tanh(x/4)

#gpu並列化予定
function train_model!(env::Env, buffer::ReplayBuffer, storage::Storage)
    #ll = zeros(Float32, env.batch_num)
    ll = zeros(Float32, env.batch_num, env.training_step)
    for b_num in 1:env.batch_num
        if(haskey(storage.storage, b_num))
            model = storage.storage[b_num] |> gpu
        else
            #model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu)),Dense(env.middle_dim, env.middle_dim, relu)), identity) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(Dense(env.middle_dim, div(env.middle_dim,4), relu), Dense(div(env.middle_dim,4), env.act_ind, tanh2)), Chain(Dense(env.middle_dim, div(env.middle_dim,4), relu), Dense(div(env.middle_dim,4), 1, tanh10)))) |> gpu
            model = Chain(Dense(env.input_dim, env.middle_dim), BatchNorm(env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu),Dense(env.middle_dim, env.middle_dim, relu)), identity)) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), env.act_ind, tanh2)), Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), 1, tanh10)))) |> gpu
            #model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu)),Dense(env.middle_dim, env.middle_dim, relu)), identity) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(Dense(env.middle_dim, env.middle_dim, relu), Dense(env.middle_dim, env.act_ind, tanh2)), Chain(Dense(env.middle_dim, env.middle_dim, relu), Dense(env.middle_dim, 1, tanh10)))) |> gpu
        end
        opt = Flux.Optimiser(WeightDecay(env.C), Adam(1f-5))
        #ParameterSchedulers.Scheduler(env.scheduler, Momentum())
        for it in 1:env.training_step
            if(it%(env.checkpoint_interval)==0)
                opt = Flux.Optimiser(WeightDecay(env.C), Adam(1f-5))
            end
            image_batch, target_batch = sample_batch!(env, buffer, storage.scores)
            val, grads = Flux.withgradient(Flux.params(model)) do
                loss(cu(image_batch),target_batch,env,model)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            ll[b_num, it] = val
            if(it > env.training_step-6)
                val, pol = loss_check(cu(image_batch),target_batch,env,model)
                println("val=$(val), pol=$(pol)")
            end
        end
        storage.storage[b_num] = model
    end
    return ll
end

function AlphaZero_ForPhysics(env::Env, envf::Env, storage::Storage)
    ld = []
    max_val = []
    
    ratio = env.ratio
    randr = env.ratio_r
    for it in 1:4
        println("=============")
        println("it=$(it);")

        replay_buffer = init_buffer(1000, env.batch_size)
        
        if(it<5)
            ratio -= Float32(1.0)
            randr -= Float32(5.0f-2)
            #@time run_selfplay_palss(env, replay_buffer, storage, ratio, randr)
            @time run_selfplay_pals(env, replay_buffer, storage, ratio, randr)
            #@time run_selfplay_pals(env, replay_buffer, storage, ratio, 1.0f0)
            @time ll = train_model!(env, replay_buffer, storage)
        else
            ratio -= Float32(2.0)
            randr -= Float32(1.0f-1)
            #ratio = Float32(6.0)
            #@time run_selfplay_palss(env, replay_buffer, storage, ratio, randr)
            @time run_selfplay_pals(env, replay_buffer, storage, ratio, randr)
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
            model0 = storage.storage[bb] |> gpu
            println("------------")
            println("head = $(bb);")
            for tes in 1:5
                game = play_physics!(envf, model0)
                score = calc_score(game.history, envf)
                val = cpu(model0(cu(make_image(envf, game, length(game.history)))))[end, 1]
                println("$(game.history), score:$(score), val(NN):$(val)")
            end
        end
        val = findmax(storage.scores)[1]
        println("max score: $(val)")
        push!(max_val, val)
        #end
    end
    
    return ld, max_val, latest_model(storage)
end

function dict_copy(orig::Dict{Vector{Int}, Float32})
    c_dict = Dict{String, Float32}()
    for k in keys(orig)
        c_dict["$(k)"] = orig[k] 
    end
    return c_dict
end

using BSON: @save
using BSON: @load
using Plots
ENV["GKSwstype"]="nul"

using JLD2
using FileIO

function main(args::Vector{String})
    #args = ARGS
    println("Start! at $(now())")
    env = init_Env(args)
    env_fc = init_Env_forcheck(args)
    storage = init_storage(env)
    st = parse(Int, args[23])
    if(st!=0)
        @load "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/AZP_valMAX_t_head$(st).bson" model0
        storage.storage[1] = gpu(model0)
        s_old = load("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores.jld2")
        for k in keys(s_old)
            storage.scores[parse.(Int, split(k[2:end-1], ","))] = Float32(s_old[k])
        end
        println("load model!")
        println("score length: $(length(storage.scores))")
    end
    
    ld, max_val, model = AlphaZero_ForPhysics(env, env_fc, storage)

    p = plot((ld[1])[1,:], yaxis=:log, linewidth=3.0)
    for i in 2:length(ld)
        plot!((ld[i])[1,:], yaxis=:log, linewidth=3.0)
    end
    savefig(p, "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/loss_valMAX_s1.png")
    println("AlphaZero Finish!")
    #println("loss-dynamics: $(ld)")
    
    for head in 1:env.batch_num
        model0 = storage.storage[head] |> cpu
        @save "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/AZP_valMAX_s1_head$(st+head).bson" model0
    end


    for it in 1:10
        game = play_physics!(env_fc, model)
        score = calc_score(game.history, env_fc)
        println("$(game.history), score:$(score)")
    end

    string_score = dict_copy(storage.scores)
    k = [keys(string_score)...]
    inds = findall(s -> string_score[s] == findmax(string_score)[1], k)
    println("max_process: $(max_val)")
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
        save("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores_s1.jld2", string_score)
    else
        save("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores_new.jld2", string_score)
    end
end



@time main(ARGS)