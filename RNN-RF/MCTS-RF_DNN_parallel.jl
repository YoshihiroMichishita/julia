using Distributed
using Dates
using SharedArrays
addprocs(24)

@everywhere include("MCTS-RF_env.jl")
@everywhere include("MCTS-RF_agt.jl")

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

#cpu並列化予定
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
    return imag, target
    #return [(make_image(env, g, turn), make_target(env, g, turn)) for (g, turn) in g_turn]
    #return [make_image(env, g, turn) for (g, turn) in g_turn], [make_target(env, g, turn) for (g, turn) in g_turn]
end

mutable struct Storage
    storage
    random_out
end

function init_storage(env)
    return Storage(Dict(), Chain(Dense(zeros(Float32, env.output,env.input_dim))))
end

function latest_model(storage::Storage)
    if(isempty(storage.storage))
        return storage.random_out
    else
        return storage.storage[maximum(keys(storage.storage))]
    end
end

#cpu並列化予定
function run_selfplay(env::Env, buffer::ReplayBuffer, storage::Storage)
    for it in 1:env.num_player
        model = latest_model(storage)
        game = play_physics!(env, model)
        save_game!(buffer, game)
    end
end

#gpu並列化予定
function loss(image::SharedMatrix{Int}, target::SharedMatrix{Float32}, env::Env, model)
    y1 = model(image)
    return sum([((y1[end,i]-target[end,i])^2 - target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]).+1f-6)) for i in 1:env.batch_size])/env.batch_size + env.C * sum(Flux.params(model)[1].^2)
end

#gpu並列化予定
function train_model!(env::Env, buffer::ReplayBuffer, storage::Storage)
    model = Chain(Dense(env.input_dim, env.middle_dim), Tuple(Chain(Parallel(+, Chain(BatchNorm(env.middle_dim), Dense(env.middle_dim, env.middle_dim, relu)),Dense(env.middle_dim, env.middle_dim, relu)), identity) for i in 1:env.depth)..., Flux.flatten, Flux.Parallel(vcat, Dense(env.middle_dim, env.act_ind, tanh), Chain(Dense(env.middle_dim, env.middle_dim, tanh),Dense(env.middle_dim, 1))))
    #model = Chain(Dense(env.input_dim=>env.middle_dim), BatchNorm(env.middle_dim), Dense(env.middle_dim=>Int(env.middle_dim/2), relu), BatchNorm(Int(env.middle_dim/2)), Dense(Int(env.middle_dim/2)=>Int(env.middle_dim/4), relu), Flux.Parallel(vcat, Chain(Dense(Int(env.middle_dim/4), env.output, tanh), Dense(env.output, env.act_ind)), Chain(Dense(Int(env.middle_dim/4), Int(env.middle_dim/4), tanh),Dense(Int(env.middle_dim/4), 1))))
    opt = ADAM()
    #ParameterSchedulers.Scheduler(env.scheduler, Momentum())
    iv_batch = []
    tv_batch = []
    for b_num in 1:env.batch_num
        image_batch, target_batch = sample_batch(env, buffer)
        push!(iv_batch, image_batch)
        push!(tv_batch, target_batch)
    end
    l = 0.0
    for it in 1:env.training_step
        for b_num in 1:env.batch_num
            #Flux.train!(loss, Flux.params(model), [(iv_batch[b_num], tv_batch[b_num], env, model)], opt)
            val, grads = Flux.withgradient(Flux.params(model)) do
                loss(iv_batch[b_num],tv_batch[b_num],env,model)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            l+=val/(env.batch_num*env.training_step)
        end
    end
    storage.storage[env.training_step] = model

    return l
end
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
    

    for it in 1:40
        println("=============")
        println("it=$(it);")

        replay_buffer = init_buffer(1000, env.batch_size)
        run_selfplay(env, replay_buffer, storage)
        ll = train_model!(env, replay_buffer, storage)
        println("loss_average: $(ll)")
        push!(ld,ll)
        if(it%20==0)
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
        @load "/home/yoshihiro/Documents/Codes/julia/RNN-RF/AlphaZero_ForPhysics_new40_0612.bson" model
        storage.storage[env.training_step] = model
        println("load model!")
    end
    ld, model = AlphaZero_ForPhysics(env, storage)
    println("AlphaZero Finish!")
    println("loss-dynamics: $(ld)")
    @save "/home/yoshihiro/Documents/Codes/julia/RNN-RF/AlphaZero_ForPhysics_new_0614.bson" model
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