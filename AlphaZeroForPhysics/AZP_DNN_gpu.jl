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
    #return Storage(Dict(), Chain(Dense(zeros(Float32, env.output,env.input_dim))))
    return Storage(Dict(), Chain(Dense(zeros(Float32, env.output,env.input_dim))), Dict())
end

function latest_model(storage::Storage)
    if(isempty(storage.storage))
        return storage.random_out
    else
        return storage.storage[maximum(keys(storage.storage))]
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
    model = gpu(latest_model(storage))
    for it in 1:env.num_player
        #model = gpu(latest_model(storage))
        game = play_physics!(env, model, ratio)
        save_game!(buffer, game)
    end
end

function run_selfplay(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32)
    model = gpu(latest_model(storage))
    for it in 1:env.num_player
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
    return sum([((y1[end,i]-target[end,i])^2 - target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]).+1f-6)) for i in 1:env.batch_size])/env.batch_size + env.C * sum(sqnorm, Flux.params(model))
end


tanh10(x) = Float32(10)*tanh(x)
tanh2(x) = Float32(2)*tanh(x)

#gpu並列化予定
function train_model!(env::Env, buffer::ReplayBuffer, storage::Storage)
    model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu)),Dense(env.middle_dim, env.middle_dim, relu)), identity) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Chain(Dense(env.middle_dim, env.middle_dim, relu), Dense(env.middle_dim, env.act_ind)), Chain(Dense(env.middle_dim, env.middle_dim, relu), Dense(env.middle_dim, 1, tanh10)))) |> gpu

    opt = ADAM()
    #ParameterSchedulers.Scheduler(env.scheduler, Momentum())
    iv_batch = []
    tv_batch = []
    for b_num in 1:env.batch_num
        image_batch, target_batch = sample_batch!(env, buffer, storage.scores)
        #image_batch, target_batch = sample_batch(env, buffer)
        push!(iv_batch, image_batch)
        push!(tv_batch, target_batch)
    end
    l = 0.0
    for it in 1:env.training_step
        for b_num in 1:env.batch_num
            Flux.train!(loss, Flux.params(model), [(cu(iv_batch[b_num]), tv_batch[b_num], env, model)], opt)
            #=
            val, grads = Flux.withgradient(Flux.params(model)) do
                loss(cu(iv_batch[b_num]),cu(tv_batch[b_num]),env,model)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            l+=val/(env.batch_num*env.training_step)
            =#
        end
    end
    storage.storage[env.training_step] = model

    return l
end

#gpu並列化予定
#=
function train_model!(env::Env, buffer::ReplayBuffer, storage::Storage)
    model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu)),Dense(env.middle_dim, env.middle_dim, relu)), identity) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Dense(env.middle_dim, env.act_ind, tanh2), Dense(env.middle_dim, 1, tanh10)))

    opt = ADAM()
    #ParameterSchedulers.Scheduler(env.scheduler, Momentum())
    iv_batch = [SharedArray{Int}(env.input_dim, env.batch_size) for i in 1:env.batch_num]
    tv_batch = [SharedArray{Float32}(env.input_dim, env.batch_size) for i in 1:env.batch_num]
    for b_num in 1:env.batch_num
        sample_batch!(env, buffer, iv_batch[b_num], tv_batch[b_num])
    end
    l = 0.0
    for it in 1:env.training_step
        for b_num in 1:env.batch_num
            #Flux.train!(loss, Flux.params(model), [(iv_batch[b_num], tv_batch[b_num], env, model)], opt)
            val, grads = Flux.withgradient(Flux.params(model)) do
                loss(sdata(iv_batch[b_num]),sdata(tv_batch[b_num]),env,model)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            l+=val/(env.batch_num*env.training_step)
        end
    end
    storage.storage[env.training_step] = model
    iv_batch = nothing
    tv_batch = nothing

    return l
end=#
#=
function train_model(env::Env, buffer::ReplayBuffer, storage::Storage)
    model = Chain(Dense(env.input_dim=>env.middle_dim, relu), BatchNorm(env.middle_dim), Dense(env.middle_dim=>env.middle_dim, relu), BatchNorm(env.middle_dim), Dense(env.middle_dim=>env.output, relu))
    opt = ParameterSchedulers.Scheduler(env.scheduler, Momentum())
    for it in 1:env.training_step
        if(it%env.checkpoint_interval==0)
            storage.storage[it] = model
        end
        image_batch, target_batch = sample_batch(env, buffer)
        Flux.train!(loss, Flux.params(model), [(image_batch, target_batch, env, model)], opt)
    end
    Storage.storage[env.training_step] = model
end=#

function AlphaZero_ForPhysics(env::Env, storage::Storage)
    ld = []
    
    
    for it in 1:8
        println("=============")
        println("it=$(it);")

        replay_buffer = init_buffer(1000, env.batch_size)
        
        if(it<4)
            ratio = Float32(12.0)
            @time run_selfplay(env, replay_buffer, storage, ratio, 1.0f0)
            @time ll = train_model!(env, replay_buffer, storage)
        else
            run_selfplay(env, replay_buffer, storage, ratio, 0.5f0)
            ll = train_model!(env, replay_buffer, storage)
        end
        #@report_call run_selfplay(env, replay_buffer, storage)
        #ll = @report_call train_model!(env, replay_buffer, storage)
        println("loss_average: $(ll)")
        push!(ld,ll)
        println("store data")
        println(length(storage.scores))
        if(it%4==0)
            for tes in 1:5
                game = play_physics!(env, latest_model(storage))
                score = calc_score(game.history, env)
                println("$(game.history), score:$(score)")
            end
        end
    end
    
    return ld, latest_model(storage)
end

using BSON: @save
using BSON: @load

function main(args::Vector{String})
    #args = ARGS
    println("Start! at $(now())")
    env = init_Env(args)
    storage = init_storage(env)
    if(args[21]!="0")
        @load "/home/yoshihiro/Documents/Codes/julia/RNN-RF/AlphaZero_ForPhysics_new_0614.bson" model
        storage.storage[env.training_step] = model
        println("load model!")
    end
    ld, model = AlphaZero_ForPhysics(env, storage)
    println("AlphaZero Finish!")
    println("loss-dynamics: $(ld)")
    model2 = cpu(model)
    @save "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/AZP.bson" model2
    for it in 1:10
        game = play_physics!(env, model)
        score = calc_score(game.history, env)
        println("$(game.history), score:$(score)")
    end
end



@time main(ARGS)
#=
function test()
    env = init_Env(ARGS)
    storage = init_storage(env)
    replay_buffer = init_buffer(100, 32)

    run_selfplay(env, replay_buffer, storage)

    for i in 1:10
        @show replay_buffer.buffer[end-i].history
    end
    
end

test()=#