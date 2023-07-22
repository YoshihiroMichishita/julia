using Distributed
using Dates
using SharedArrays
#using JET
addprocs(24)

@everywhere include("AZP_env.jl")
@everywhere include("AZP_agt.jl")
include("AZP_mcts.jl")

#1局1局の情報をストックする
mutable struct ReplayBuffer
    buffer::Vector{Agent}
    buffer_size::Int
    batch_size::Int
    lk
end

function init_buffer(buffer_size::Int, batch_size::Int)
    return ReplayBuffer([], buffer_size, batch_size, ReentrantLock())
end

function save_game!(buffer::ReplayBuffer, agt::Agent)
    @lock buffer.lk begin
        if length(buffer.buffer) > buffer.buffer_size
            popfirst!(buffer.buffer)
        end
        push!(buffer.buffer, agt)
    end
end
#=
function save_game!(buffer::ReplayBuffer, agt::Agent)
    if length(buffer.buffer) > buffer.buffer_size
        popfirst!(buffer.buffer)
    end
    push!(buffer.buffer, agt)
end=#
#=
@everywhere function save_game!(buffer, agt::Agent, buffer_size::Int)
    if length(buffer) > buffer_size
        popfirst!(buffer)
    end
    push!(buffer, agt)
end=#



#=function sample_batch!(env::Env, buffer::ReplayBuffer, imag::SharedArray, target::SharedArray)
    games = sample(buffer.buffer, weights([length(agt.history) for agt in buffer.buffer]), buffer.batch_size, replace=false)
    g_turn = [(g, rand(1:length(g.history))) for g in games]
    #imag = SharedArray(zeros(Int, env.input_dim, buffer.batch_size))
    #target = SharedArray(zeros(Float32, env.output, buffer.batch_size))
    @sync @distributed for it in 1:buffer.batch_size
        g, turn = g_turn[it]
        imag[:,it] = make_image(env, g, turn)
        target[:,it] = make_target(env, g, turn)
    end
    #return imag, target
    #return [(make_image(env, g, turn), make_target(env, g, turn)) for (g, turn) in g_turn]
    #return [make_image(env, g, turn) for (g, turn) in g_turn], [make_target(env, g, turn) for (g, turn) in g_turn]
end=#

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


#cpu並列化予定
function sample_batch!(env::Env, buffer::ReplayBuffer, scores::Dict{Vector{Int}, Float32})
    games = sample(buffer.buffer, weights([length(agt.history) for agt in buffer.buffer]), buffer.batch_size, replace=true)
    g_turn = [(g, rand(1:length(g.history))) for g in games]
    imag = SharedArray(zeros(Int, env.input_dim, buffer.batch_size))
    target = SharedArray(zeros(Float32, env.output, buffer.batch_size))
    @sync @distributed for it in 1:buffer.batch_size
        g, turn = g_turn[it]
        imag[:,it] = make_image(env, g, turn)
        target[:,it] = make_target(env, g, scores, turn)
    end
    tar_data = sdata(target)
    for it in 1:buffer.batch_size
        g, turn = g_turn[it]
        scores[g.history] = tar_data[end,it]
    end

    return sdata(imag), tar_data
end
#=
function sample_batch(env::Env, buffer::ReplayBuffer)
    games = sample(buffer.buffer, weights([length(agt.history) for agt in buffer.buffer]), buffer.batch_size, replace=false)
    g_turn = [(g, rand(1:length(g.history))) for g in games]
    imag = SharedArray(zeros(Int, env.input_dim, buffer.batch_size))
    target = SharedArray(zeros(Float32, env.output, buffer.batch_size))
    @sync @distributed for it in 1:buffer.batch_size
        g, turn = g_turn[it]
        imag[:,it] = make_image(env, g, turn)
        target[:,it] = make_target(env, g, turn)
    end
    return sdata(imag), sdata(target)
    #return [(make_image(env, g, turn), make_target(env, g, turn)) for (g, turn) in g_turn]
    #return [make_image(env, g, turn) for (g, turn) in g_turn], [make_target(env, g, turn) for (g, turn) in g_turn]
end=#



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
    synchronize()
    Threads.@threads for it in 1:env.num_player
        if(it%(env.num_player/10)==0)
            print("#")
        end
        #model = gpu(latest_model(storage))
        game = play_physics!(env, model, ratio, noise_r)
        save_game!(buffer, game)
    end
end

#gpu並列化予定
#=function loss(image::SharedMatrix{Int}, target::SharedMatrix{Float32}, env::Env, model)
    y1 = model(image)
    return sum([((y1[end,i]-target[end,i])^2 - target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]).+1f-6)) for i in 1:env.batch_size])/env.batch_size + env.C * sum(Flux.params(model)[1].^2)
end=#

sqnorm(x) = sum(abs2, x)

function loss(image::CuArray{Int, 2}, target::Matrix{Float32}, env::Env, model::Chain)
    y1 = cpu(model(image))
    return sum([(((y1[end,i]-target[end,i])/10)^2 - target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]).+1f-8)) for i in 1:env.batch_size])/env.batch_size
    # + env.C * sum(sqnorm, Flux.params(model))
end


tanh10(x) = Float32(10)*tanh(x)
tanh2(x) = Float32(2)*tanh(x)

#gpu並列化予定
function train_model!(env::Env, buffer::ReplayBuffer, storage::Storage)
    #ll = zeros(Float32, env.batch_num)
    ll = zeros(Float32, env.batch_num, env.training_step)
    for b_num in 1:env.batch_num
        if(haskey(storage.storage, b_num))
            model = storage.storage[b_num] |> gpu
        else
            model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu)),Dense(env.middle_dim, env.middle_dim, relu)), identity) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(Dense(env.middle_dim, env.middle_dim, relu), Dense(env.middle_dim, env.act_ind)), Chain(Dense(env.middle_dim, env.middle_dim, relu), Dense(env.middle_dim, 1, tanh10)))) |> gpu
        end
        opt = Flux.Optimiser(WeightDecay(env.C), Adam(1f-5))
        #ParameterSchedulers.Scheduler(env.scheduler, Momentum())
        iv_batch = []
        tv_batch = []
        bn::Int = 10
        for it in 1:bn
            image_batch, target_batch = sample_batch!(env, buffer, storage.scores)
            push!(iv_batch, image_batch)
            push!(tv_batch, target_batch)
        end
        for it in 1:env.training_step
            if(it%env.checkpoint_interval==0)
                opt = Flux.Optimiser(WeightDecay(env.C), Adam(1f-5))
                print("#")
            end
            for s in 1:bn
                #Flux.train!(loss, Flux.params(model), [(cu(iv_batch[s]), tv_batch[s], env, model, 1.0f0)], opt)
                
                val, grads = Flux.withgradient(Flux.params(model)) do
                    loss(cu(iv_batch[s]),tv_batch[s],env,model)
                end
                Flux.Optimise.update!(opt, Flux.params(model), grads)
                ll[b_num, it] += val/bn
                #ll[b_num] += val/(bn*env.training_step)
            end
        end
        storage.storage[b_num] = model
    end
    return ll
end

function AlphaZero_ForPhysics(env::Env, envf::Env, storage::Storage)
    ld = []
    
    
    for it in 1:8
        println("=============")
        println("it=$(it);")

        replay_buffer = init_buffer(1000, env.batch_size)
        ratio = Float32(12.0)
        if(it<5)
            @time run_selfplay(env, replay_buffer, storage, ratio, 1.0f0)
            @time ll = train_model!(env, replay_buffer, storage)
        else
            run_selfplay(env, replay_buffer, storage, ratio, 5f-1)
            ll = train_model!(env, replay_buffer, storage)
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
                println("$(game.history), score:$(score)")
            end
        end
        #end
    end
    
    return ld, latest_model(storage)
end

using BSON: @save
using BSON: @load
using Plots
ENV["GKSwstype"]="nul"

function main(args::Vector{String})
    #args = ARGS
    println("Start! at $(now())")
    println("nthreads: $(Threads.nthreads())")
    env = init_Env(args)
    env_fc = init_Env_forcheck(args)
    storage = init_storage(env)
    st = parse(Int, args[21])
    if(st!=0)
        @load "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/AZP_head$(st).bson" model0
        storage.storage[1] = gpu(model0)
        println("load model!")
    end
    
    ld, model = AlphaZero_ForPhysics(env, env_fc, storage)

    p = plot((ld[1])[1,:], yaxis=:log, linewidth=3.0)
    for i in 2:length(ld)
        plot!((ld[i])[1,:], yaxis=:log, linewidth=3.0)
    end
    savefig(p, "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/loss.png")
    println("AlphaZero Finish!")
    #println("loss-dynamics: $(ld)")
    
    for head in 1:env.batch_num
        model0 = storage.storage[head] |> cpu
        @save "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/AZP_head$(st+head+30).bson" model0
    end
    #model2 = cpu(model)
    #@save "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/AZP.bson" model2
    for it in 1:10
        game = play_physics!(env_fc, model)
        score = calc_score(game.history, env_fc)
        println("$(game.history), score:$(score)")
    end
end



@time main(ARGS)