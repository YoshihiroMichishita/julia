using Dates

include("AZP_env.jl")
include("AZP_agt.jl")
include("AZP_mcts_single_CPU.jl")

#Stock the games in serious trials. 
mutable struct ReplayBuffer
    buffer::Vector{Agent}
    buffer_size::Int
    batch_size::Int
    scores::Dict{Vector{Int}, Float32}
    #count::Int
end

#initialization(Constructor)
function init_buffer(buffer_size::Int, batch_size::Int)
    return ReplayBuffer([], buffer_size, batch_size, Dict())
end

function show_buffer(buffer::ReplayBuffer)
    println("buffer_size: $(length(buffer.buffer))")
    println("scores: $(length(buffer.scores))")
end

#save games in buffer
function save_game!(buffer::ReplayBuffer, agt::Agent)
    #if size of the games saved in buffer is larger than its maximum, delete a date for a old game.
    if length(buffer.buffer) > buffer.buffer_size
        popfirst!(buffer.buffer)
    end
    push!(buffer.buffer, agt)
end


#=
mutable struct Storage
    storage::Dict{Int, Chain}
    random_out::Chain
    scores::Dict{Vector{Int}, Float32}
    scores_size::Int
end

function save_score!(storage::Storage, history::Vector{Int}, score::Float32)
    if length(storage.score) > storage.scores_size
        popfirst!(storage.score)
    end
    push!(buffer.buffer, agt)
end

function init_storage(env)
    return Storage(Dict(), Chain(Dense(zeros(Float32, env.output,env.input_dim))), Dict())
end=#

#=
function latest_model(storage::Storage)
    if(isempty(storage.storage))
        return storage.random_out
    else
        return storage.storage[rand(keys(storage.storage))]
    end
end=#

#=
function WeightSample(hist::Vector{Int})
    s = [i for i in 1:length(hist)]
    ww = s.^2
    ww = ww/sum(ww)
    return sample(s, ProbabilityWeights(ww))
end=#

#we can weight the probability of sampling the experience from the buffer as the difference between the true value and estimated value is large.(Prioritzed Experience Replay)
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

#Sampling situations of the games and its statistics form the buffer.
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

function sample_batch!(env::Env, buffer::ReplayBuffer, storage::Storage)
    games = sample(buffer.buffer, weights([length(agt.history) for agt in buffer.buffer]), env.batch_size, replace=true)
    g_turn = [(g, WeightSample(g.history)) for g in games]

    if(isempty(storage.scores))
        imag = zeros(Int, env.input_dim, env.batch_size)
        target = zeros(Float32, env.output, env.batch_size)
        for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, storage, turn)
        end
    else
        imag = zeros(Int, env.input_dim, 4env.batch_size)
        target = zeros(Float32, env.output, 4env.batch_size)
        for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, storage, turn)
        end
        for it in 1:3env.batch_size
            hist = rand(keys(storage.scores))
            imag[:,env.batch_size+it] = make_image(env, hist)
            target[end,env.batch_size+it] = storage.scores[hist]
        end
    end

    tar_data = copy(target)
    for it in 1:env.batch_size
        g, turn = g_turn[it]
        for l in 1:length(g.history)
            his = g.history[1:l]
            if(haskey(storage.scores, his))
                if(storage.scores[his] < tar_data[end,it])
                    #println("score_data:$(scores[his]), tar_data:$(tar_data[end,it]), max:$(max(scores[his], tar_data[end,it]))")
                    storage.scores[his] = tar_data[end,it]
                end
                #max(scores[his], tar_data[end,it])
            else
                storage.scores[his] = tar_data[end,it]
            end
        end
    end
    return imag, tar_data
end


#Play and save the game by AZfP
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
        
        if(it%(env.num_player/10)==0)
            print("#")
        end
        #model = gpu(latest_model(storage))
        game = play_physics!(env, model, ratio, noise_r, storage, max_hist)
        save_game!(buffer, game)
        if(length(max_hist)>2200)
            break
        end
    end
end


function loss(image::Matrix{Int}, target::Matrix{Float32}, env::Env, model::Chain)
    y1 = model(image)
    return sum([((env.ratio*(y1[end,i]-target[end,i]))^2 - target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]))) for i in 1:env.batch_size])/env.batch_size
    # + env.C * sum(sqnorm, Flux.params(model))
end

function loss_check(image::Matrix{Int}, target::Matrix{Float32}, env::Env, model::Chain)
    y1 = model(image)
    val = sum([((env.ratio*(y1[end,i]-target[end,i]))^2) for i in 1:env.batch_size])/env.batch_size
    pol = sum([(-target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]))) for i in 1:env.batch_size])/env.batch_size
    return val, pol
    # + env.C * sum(sqnorm, Flux.params(model))
end


tanh10(x) = Float32(15)*tanh(x/10)
tanh2(x) = Float32(4)*tanh(x/4)

#training
function train_model!(env::Env, buffer::ReplayBuffer, storage::Storage)
    ll = zeros(Float32, env.training_step)
    #ll = zeros(Float32, env.batch_num, env.training_step)
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
            image_batch, target_batch = sample_batch!(env, buffer, storage)
            val, grads = Flux.withgradient(Flux.params(model)) do
                loss(image_batch,target_batch,env,model)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            ll[it] = val
        end
        storage.storage[b_num] = model
    end
    return ll
end




function AlphaZero_ForPhysics(env::Env, envf::Env, storage::Storage)
    ld = []
    max_hist::Vector{Float32} = [-12.0f0]
    itn = 4
    ratio = env.ratio
    randr = env.ratio_r
    for it in 1:itn
        println("=============")
        println("it=$(it);")

        replay_buffer = init_buffer(1200, env.batch_size)
        
        if(it<5)
            @time run_selfplay!(env, replay_buffer, storage, ratio, randr, max_hist)
            @time ll = train_model!(env, replay_buffer, storage)
        else
            #ratio -= Float32(2.0)
            #randr -= Float32(1.0f-1)
            #ratio = Float32(6.0)
            #@time run_selfplay_palss(env, replay_buffer, storage, ratio, randr)
            @time run_selfplay!(env, replay_buffer, storage, ratio, randr, max_hist)
            @time ll = train_model!(env, replay_buffer, storage, ratio)
        end

        show_buffer(replay_buffer)
        push!(ld,ll)
        println("store data")
        println(length(storage.scores))

        for bb in 1:env.batch_num
            model0 = storage.storage[bb]
            println("------------")
            println("head = $(bb);")
            for tes in 1:2
                game = play_physics!(envf, model0)
                score = calc_score(game.history, envf)
                val = model0(make_image(envf, game, length(game.history)))[end, 1]
                println("$(hist2eq(game.history)), score:$(score), val(NN):$(val)")
            end
        end

        val, key = findmax(storage.scores)
        println("max score: $(val)")
        println(key)
        if(length(max_hist)>2200)
            break
        end
    end
    
    return ld, max_hist, latest_model(storage)
end

#When we try the optimization of the hyperparameters by genetic algorithms, we can use this.
function AlphaZero_ForPhysics_hind(env::Env, storage::Storage)
    ld = []
    max_hist::Vector{Float32} = [-12.0f0]
    itn = 4
    ratio = env.ratio
    randr = env.ratio_r
    for it in 1:itn
        replay_buffer = init_buffer(1500, env.batch_size)
        
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


#using BSON: @save
#using BSON: @load
using Plots
ENV["GKSwstype"]="nul"
Plots.scalefontsizes(1.3)

#using JLD2
#using FileIO
using DataFrames
using CSV

date = 1118

function main(args::Vector{String})
    #args = ARGS
    println("Start! at $(now())")
    env = init_Env(args)
    env_fc = init_Env_forcheck(args)
    lds = []
    max_hists = []
    
    bb = 5 #number of trials by AZfP
    for dd in 1:bb
        storage = init_storage(env, 1200)
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

    h_max = 2000
    p0 = plot(max_hists[1], linewidth=3.0, xaxis=:log, xrange=(1,h_max))
    for i in 2:length(max_hists)
        p0 = plot!(max_hists[i], linewidth=3.0, xaxis=:log, xrange=(1,h_max))
    end
    savefig(p0, "valMAX_itr_mt$(env.max_turn)_$(date).png")

    save_data = DataFrame(hist1=max_hists[1][1:h_max],hist2=max_hists[2][1:h_max],hist3=max_hists[3][1:h_max],hist4=max_hists[4][1:h_max],hist5=max_hists[5][1:h_max])
    CSV.write("./hists_$(date).csv", save_data)
    #savefig(p0, "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/valMAX_itr_mt$(env.max_turn)_$(date).png")
    #savefig(p0, "/Users/johnbrother/Documents/Codes/julia/AlphaZeroForPhysics/valMAX_itr_mt$(env.max_turn)_$(date).png")
    
    
    p1 = plot((lds[1])[1], yaxis=:log, linewidth=3.0)
    for i in 2:size(lds[1])[1]
        p1 = plot!((lds[1])[i], yaxis=:log, linewidth=3.0)
    end
    #savefig(p, "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/loss_valMAX_mt$(env.max_turn)_$(date).png")
    #savefig(p, "/Users/johnbrother/Documents/Codes/julia/AlphaZeroForPhysics/loss_valMAX_mt$(env.max_turn)_$(date).png")
    savefig(p1, "loss_valMAX_mt$(env.max_turn)_$(date).png")
    
    p2 = plot((lds[1])[1], linewidth=3.0)
    for i in 2:size(lds[1])[1]
        p2 = plot!((lds[1])[i], linewidth=3.0)
    end
    savefig(p2, "loss_valMAX_mt$(env.max_turn)_$(date)_normal.png")

    println("AlphaZero Finish!")
    
    #=
    if(st == 0)
        save("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores_mt$(env.max_turn)_$(date).jld2", string_score)
    else
        save("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores_new.jld2", string_score)
    end=#
end



@time main(ARGS)