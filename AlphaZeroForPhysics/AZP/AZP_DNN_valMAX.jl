using Distributed
using Dates
#using JET
addprocs(10)

@everywhere include("AZP_env.jl")
@everywhere include("AZP_agt.jl")
@everywhere include("AZP_mcts_single.jl")
#include("AZP_mcts_valMAX.jl")

#1局1局の情報をストックする
@everywhere mutable struct ReplayBuffer
    buffer::Vector{Agent}
    buffer_size::Int
    batch_size::Int
    #count::Int
end

@everywhere function init_buffer(buffer_size::Int, batch_size::Int)
    return ReplayBuffer([], buffer_size, batch_size)
end

@everywhere function show_buffer(buffer::ReplayBuffer)
    println("buffer_size: $(length(buffer.buffer))")
end

@everywhere function save_game!(buffer::ReplayBuffer, agt::Agent)
    if length(buffer.buffer) > buffer.buffer_size
        popfirst!(buffer.buffer)
    end
    push!(buffer.buffer, agt)
end


#=
function WeightSample(hist::Vector{Int})
    s = [i for i in 1:length(hist)]
    ww = s.^2
    ww = ww/sum(ww)
    return sample(s, ProbabilityWeights(ww))
end=#

@everywhere function WeightSample(hist::Vector{Int})
    s = [i for i in 1:length(hist)]
    w = [2^i for i in 1:length(hist)]
    ww = w/sum(w)
    return sample(s, ProbabilityWeights(ww))
end
@everywhere function WeightSample(agt::Agent)
    s = [i for i in 1:length(agt.history)]
    w = agt.surprise
    ww = w/sum(w)
    return sample(s, ProbabilityWeights(ww))
end

#cpu並列化予定
@everywhere function sample_batch!(env::Env, buffer::ReplayBuffer, scores::Dict{Vector{Int}, Float32})
    games = sample(buffer.buffer, weights([length(agt.history) for agt in buffer.buffer]), env.batch_size, replace=true)
    g_turn = [(g, WeightSample(g.history)) for g in games]

    if(isempty(scores))
        #imag = SharedArray(zeros(Int, env.input_dim, env.batch_size))
        #target = SharedArray(zeros(Float32, env.output, env.batch_size))
        #@sync @distributed for it in 1:env.batch_size
        imag = zeros(Int, env.input_dim, env.batch_size)
        target = zeros(Float32, env.output, env.batch_size)
        for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, scores, turn)
        end
    else
        #imag = SharedArray(zeros(Int, env.input_dim, 4env.batch_size))
        #target = SharedArray(zeros(Float32, env.output, 4env.batch_size))
        #@sync @distributed for it in 1:env.batch_size
        imag = zeros(Int, env.input_dim, 4env.batch_size)
        target = zeros(Float32, env.output, 4env.batch_size)
        for it in 1:env.batch_size
            g, turn = g_turn[it]
            imag[:,it] = make_image(env, g, turn)
            target[:,it] = make_target(env, g, scores, turn)
        end
        #@sync @distributed 
        for it in 1:3env.batch_size
            hist = rand(keys(scores))
            imag[:,env.batch_size+it] = make_image(env, hist)
            target[end,env.batch_size+it] = scores[hist]
        end
    end
    tar_data = copy(target)
    #tar_data = sdata(target)
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
    #return sdata(imag), tar_data
    return imag, tar_data
end


@everywhere function sample_batch!(env::Env, buffer::ReplayBuffer, storage::Storage)
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
#=
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
end=#

@everywhere lmax_hist::Int = 2200

#cpu並列化予定
#=
@everywhere function run_selfplay!(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32)
    model = latest_model(storage) |> gpu
    for it in 1:env.num_player
        #=
        par = div(env.num_player,10)
        if(it%par==0)
            print("#")
        end=#
        #model = gpu(latest_model(storage))
        game = play_physics!(env, model, ratio)
        save_game!(buffer, game)
        if(length(max_hist)>lmax_hist)
            #println("max_hist: $(max_hist[end])")
            #println("lmax_hist: $(lmax_hist)")
            break
        end
    end
end

@everywhere function run_selfplay!(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32)
    model = latest_model(storage) |> gpu
    for it in 1:env.num_player
        #=
        par = div(env.num_player,10)
        if(it%par==0)
            print("#")
        end=#
        #model = gpu(latest_model(storage))
        game = play_physics!(env, model, ratio, noise_r)
        save_game!(buffer, game)
        if(length(max_hist)>lmax_hist)
            #println("max_hist: $(max_hist[end])")
            #println("lmax_hist: $(lmax_hist)")
            break
        end
    end
end=#

@everywhere function run_selfplay!(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32, max_hist::Vector{Float32})
    model = latest_model(storage) |> gpu
    for it in 1:env.num_player
        game = play_physics!(env, model, ratio, noise_r, storage, max_hist)
        save_game!(buffer, game)
        if(length(max_hist)>lmax_hist)
            break
        end
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
    max_scores = []
    for it in 1:div(env.num_player, nworkers())
        game = play_physics!(env, model, ratio, noise_r, scores)
        push!(games, game)
        push!(max_scores, findmax(scores)[1])
    end
    return games, max_scores
end
#=
@everywhere function run_selfplay_worker_withS(env::Env, model::Chain, ratio::Float32, noise_r::Float32, scores::Dict{Vector{Int}, Float32})
    games = Agent[]
    for it in 1:div(env.num_player, nworkers())
        game = play_physics_s!(env, model, ratio, noise_r, scores)
        push!(games, game)
    end
    return games
end=#

#=
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
    num_calc = zeros(Int,div(env.num_player,nworkers()))
    max_itr = zeros(Float32,div(env.num_player,nworkers()))
    for i in workers()
        push!(futures, remotecall(run_selfplay_worker, i, env, model, ratio, noise_r, storage.scores))
    end
    for f in futures
        games, max_scores = fetch(f)
        #for g in games
        for it in 1:length(games)
            save_game!(buffer, games[it])
            num_calc[it] += div(games[it].num_calc,nworkers())
            max_itr[it] = max(max_itr[it], max_scores[it])
        end
    end
    return num_calc, max_itr
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
end=#

@everywhere function loss(image::CuArray{Int, 2}, target::Matrix{Float32}, env::Env, model::Chain, ratio::Float32)
    y1 = cpu(model(image))
    return sum([((ratio*(y1[end,i]-target[end,i]))^2 - target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]))) for i in 1:env.batch_size])/env.batch_size
    # + env.C * sum(sqnorm, Flux.params(model))
end

@everywhere function loss_check(image::CuArray{Int, 2}, target::Matrix{Float32}, env::Env, model::Chain, ratio::Float32)
    y1 = cpu(model(image))
    val = sum([((ratio*(y1[end,i]-target[end,i]))^2) for i in 1:env.batch_size])/env.batch_size
    pol = sum([(-target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]))) for i in 1:env.batch_size])/env.batch_size
    return val, pol
    # + env.C * sum(sqnorm, Flux.params(model))
end


@everywhere tanh10(x) = Float32(15)*tanh(x/10)
@everywhere tanh2(x) = Float32(4)*tanh(x/4)

#gpu並列化予定
@everywhere function train_model!(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32)
    #ll = zeros(Float32, env.batch_num)
    ll = zeros(Float32, env.batch_num, env.training_step)
    for b_num in 1:env.batch_num
        if(haskey(storage.storage, b_num))
            model = storage.storage[b_num] |> gpu
        else
            model = Chain(Dense(env.input_dim, env.middle_dim), BatchNorm(env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu),Dense(env.middle_dim, env.middle_dim, relu)), identity)) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), env.act_ind, tanh2)), Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, div(env.middle_dim,2), relu), Tuple(Dense(div(env.middle_dim,2), div(env.middle_dim,2), relu) for i in 1:3)..., Dense(div(env.middle_dim,2), 1, tanh10)))) |> gpu
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
                loss(cu(image_batch),target_batch,env,model, ratio)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            ll[b_num, it] = val
            if(it > env.training_step-4)
                val, pol = loss_check(cu(image_batch),target_batch,env,model, ratio)
                println("val=$(val), pol=$(pol)")
            end
        end
        storage.storage[b_num] = model
    end
    return ll
end

#=
function AlphaZero_ForPhysics(env::Env, storage::Storage)
    ld = []
    max_val = []
    itn = 3
    num_calcs = zeros(Int, itn*div(env.num_player, nworkers()))
    max_itrs = zeros(Float32, itn*div(env.num_player, nworkers()))
    lastit = 0
    ratio = env.ratio
    randr = env.ratio_r
    for it in 1:itn
        println("=============")
        println("it=$(it);")

        replay_buffer = init_buffer(1200, env.batch_size)
        @time run_selfplay!(env, replay_buffer, storage, ratio, randr, max_hist)
        @time ll = train_model!(env, replay_buffer, storage, ratio)
        #=
        if(it<5)
            ratio -= Float32(1.0)
            randr -= Float32(1.0f-1)
            #@time run_selfplay_palss(env, replay_buffer, storage, ratio, randr)
            @time num_calc, max_itr = run_selfplay_pals(env, replay_buffer, storage, ratio, randr)
            #@time run_selfplay_pals(env, replay_buffer, storage, ratio, 1.0f0)
            @time ll = train_model!(env, replay_buffer, storage)
            num_calcs[(it-1)*div(env.num_player, nworkers())+1:it*div(env.num_player, nworkers())] = num_calc .+ lastit
            max_itrs[(it-1)*div(env.num_player, nworkers())+1:it*div(env.num_player, nworkers())] = max_itr
            lastit = num_calcs[it*div(env.num_player, nworkers())]
        else
            ratio -= Float32(2.0)
            randr -= Float32(1.0f-1)
            #ratio = Float32(6.0)
            #@time run_selfplay_palss(env, replay_buffer, storage, ratio, randr)
            @time num_calc, max_itr = run_selfplay_pals(env, replay_buffer, storage, ratio, randr)
            @time ll = train_model!(env, replay_buffer, storage)
            num_calcs[(it-1)*div(env.num_player, nworkers())+1:it*div(env.num_player, nworkers())] = num_calc .+ lastit
            max_itrs[(it-1)*div(env.num_player, nworkers())+1:it*div(env.num_player, nworkers())] = max_itr
            lastit = num_calcs[it*div(env.num_player, nworkers())]
        end=#
        #@report_call run_selfplay(env, replay_buffer, storage)
        #ll = @report_call train_model!(env, replay_buffer, storage)
        #println("loss_average: $(ll)")
        push!(ld,ll)
        println("store data")
        println(length(storage.scores))
        #if(it%2==0)
        #=
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
        end=#
        val = findmax(storage.scores)[1]
        println("max score: $(val)")
        push!(max_val, val)
        #end
    end
    
    return ld, max_val, latest_model(storage), num_calcs, max_itrs
end=#

@everywhere function AlphaZero_ForPhysics(env::Env, storage::Storage)
    ld = []
    max_hist::Vector{Float32} = [-12.0f0]
    itn = 6
    lastit = 0
    ratio = env.ratio
    randr = env.ratio_r
    for it in 1:itn
        #println("=============")
        #println("it=$(it);")

        replay_buffer = init_buffer(1200, env.batch_size)
        
        @time run_selfplay!(env, replay_buffer, storage, ratio, randr, max_hist)
        @time ll = train_model!(env, replay_buffer, storage, ratio)
        #@report_call run_selfplay(env, replay_buffer, storage)
        #ll = @report_call train_model!(env, replay_buffer, storage)
        #println("loss_average: $(ll)")
        push!(ld,ll)
        #println("store data")
        #println(length(storage.scores))
        #if(it%2==0)
        #=
        for bb in 1:env.batch_num
            model0 = storage.storage[bb] |> gpu
            println("------------")
            println("head = $(bb);")
            
            for tes in 1:5
                game = play_physics!(envf, model0)
                score = calc_score(game.history, envf)
                val = cpu(model0(cu(make_image(envf, game, length(game.history)))))[end, 1]
                println("$(hist2eq(game.history)), score:$(score), val(NN):$(val)")
            end
        end=#
        #val, hist = findmax(storage.scores)
        #println("max score: $(val);  hist: $(hist2eq(hist))")
        if(length(max_hist)>lmax_hist)
            break
        end
    end
    
    return ld, max_hist[1:lmax_hist-env.num_player], latest_model(storage)
end

function dict_copy(orig::Dict{Vector{Int}, Float32})
    c_dict = Dict{String, Float32}()
    for k in keys(orig)
        c_dict["$(k)"] = orig[k] 
    end
    return c_dict
end

function mhists2matrix(mhists::Vector{Vector{Float32}}, num_p::Int)
    mat = zeros(Float32, length(mhists[1]), length(mhists))
    for i in 1:length(mhists)
        mat[:,i] = mhists[i][1:lmax_hist-num_p]
    end
    return mat
end

#using BSON: @save
#using BSON: @load
using Plots
ENV["GKSwstype"]="nul"
Plots.scalefontsizes(1.3)

using JLD2
using FileIO

using DataFrames
using CSV


date = 0120
using SharedArrays

function main(args::Vector{String})
    #args = ARGS
    println("Start! at $(now())")
    @everywhere env = init_Env_hind($(args))
    #st = parse(Int, args[23])
    @everywhere ds = 20
    max_hists = SharedArray(zeros(Float32, lmax_hist-env.num_player, ds))
    #=
    if(st!=0)
        @load "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/AZP_valMAX_$(date)_head$(st).bson" model0
        storage.storage[1] = gpu(model0)
        s_old = load("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores.jld2")
        for k in keys(s_old)
            storage.scores[parse.(Int, split(k[2:end-1], ","))] = Float32(s_old[k])
        end
        println("load model!")
        println("score length: $(length(storage.scores))")
    end=#
    
    #ld, max_val, model, num_calcs, max_itr = AlphaZero_ForPhysics(env, env_fc, storage)

    @sync @distributed for dd in 1:ds
        #storage = init_storage(env)
        storage = init_storage(env, 2000)
        ld, max_hist, model = AlphaZero_ForPhysics(env, storage)
        max_hists[:,dd] = max_hist
        #push!(max_hists, max_hist)
        #string_score = dict_copy(storage.scores)
        
        k = [keys(storage.scores)...]
        inds = findall(s -> storage.scores[s] == findmax(storage.scores)[1], k)
        println("max score:")
        for i in inds
            println("$(hist2eq(k[i])), $(storage.scores[k[i]])")
        end
    end
    #p0 = plot(num_calcs, max_itr, linewidth=3.0, marker=:circle)
    #savefig(p0, "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/valMAX_itr_mt$(env.max_turn)_$(date).png")
    #=
    p = plot((ld[1])[1,:], yaxis=:log, linewidth=3.0)
    for i in 2:length(ld)
        plot!((ld[i])[1,:], yaxis=:log, linewidth=3.0)
    end
    savefig(p, "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/loss_valMAX_mt$(env.max_turn)_$(date).png")=#
    m_hists = Matrix(max_hists)
    p0 = plot(m_hists[:,1], linewidth=3.0, xaxis=:log, xrange=(1,lmax_hist), yrange=(0,12))
    for i in 2:ds
        p0 = plot!(m_hists[:,i], linewidth=3.0, xaxis=:log, xrange=(1,lmax_hist), yrange=(0,12))
    end
    savefig(p0, "./valMAX_itr_mt$(env.max_turn)_$(date).png")

    #mm_hist = mhists2matrix(max_hists)
    save_data = DataFrame(m_hists, :auto)
    #save_data = DataFrame(hist1=max_hists[1][1:lmax_hist-100],hist2=max_hists[2][1:lmax_hist-100],hist3=max_hists[3][1:lmax_hist-100],hist4=max_hists[4][1:lmax_hist-100],hist5=max_hists[5][1:lmax_hist-100])
    CSV.write("./hists_mt$(env.max_turn)_$(date).csv", save_data)
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
    end=#
    #=
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
        save("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores_mt$(env.max_turn)_$(date).jld2", string_score)
    else
        save("/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/scores_new.jld2", string_score)
    end=#
end



@time main(ARGS)