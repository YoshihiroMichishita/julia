include("MCTS-RF_env.jl")
include("MCTS-RF_agt.jl")

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

#cpu並列化予定
function sample_batch(env::Env, buffer::ReplayBuffer)
    games = sample(buffer.buffer, weights([length(agt.history) for agt in buffer.buffer]), buffer.batch_size, replace=false)
    g_turn = [(g, rand(1:length(g.history))) for g in games]
    imag = zeros(Int, env.input_dim, buffer.batch_size)
    target = zeros(Float32, env.output, buffer.batch_size)
    for it in 1:buffer.batch_size
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

function latest_model(Storage)
    if(isempty(Storage.storage))
        return Storage.random_out
    else
        return storage[maximum(keys(storage))]
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
function loss(image::Matrix{Int}, target::Matrix{Float32}, env::Env, model)
    y1 = model(image)
    return sum([((y1[end,i]-target[end,i])^2 - target[1:end-1,i]' * log.(softmax(y1[1:end-1,i]))) for i in 1:env.batch_size])/env.batch_size + env.C * sum(Flux.params(model)[1].^2)
end

#gpu並列化予定
function train_model(env::Env, buffer::ReplayBuffer, storage::Storage)
    model = Chain(Dense(env.input_dim=>env.middle_dim), BatchNorm(env.middle_dim), Dense(env.middle_dim=>env.middle_dim, relu), BatchNorm(env.middle_dim), Dense(env.middle_dim=>env.middle_dim, relu), Flux.Parallel(vcat, Chain(Dense(env.middle_dim, env.output, tanh), Dense(env.output, env.act_ind)), Dense(env.middle_dim, 1)))
    opt = ParameterSchedulers.Scheduler(env.scheduler, Momentum())
    for it in 1:env.training_step
        image_batch, target_batch = sample_batch(env, buffer)
        Flux.train!(loss, Flux.params(model), [(image_batch, target_batch, env, model)], opt)
    end
    storage.storage[env.training_step] = model
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

function AlphaZero_ForPhysics(env::Env)
    storage = init_storage(env)
    replay_buffer = init_buffer(1000, env.batch_size)

    @time run_selfplay(env, replay_buffer, storage)

    @time train_model(env, replay_buffer, storage)

    return latest_model(storage)
end

function main(args::Vector{String})
    #args = ARGS
    env = init_Env(args)
    model = AlphaZero_ForPhysics(env)
    @show model
end

main(ARGS)
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