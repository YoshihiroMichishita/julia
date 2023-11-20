using Dates

#include("AZP_env.jl")
include("AZP_env.jl")
include("AZP_agt.jl")
include("AZP_mcts_single_woNN.jl")

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

function show_buffer(buffer::ReplayBuffer)
    println("buffer_size: $(length(buffer.buffer))")
end

function save_game!(buffer::ReplayBuffer, agt::Agent)
    if length(buffer.buffer) > buffer.buffer_size
        popfirst!(buffer.buffer)
    end
    push!(buffer.buffer, agt)
end





#cpu並列化予定
function run_selfplay(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32)
    for it in 1:env.num_player
        if(it%(env.num_player/10)==0)
            print("#")
        end
        #model = gpu(latest_model(storage))
        game = play_physics!(env, ratio, noise_r)
        save_game!(buffer, game)
    end
end

lmax_hist::Int = 2100

function run_selfplay!(env::Env, buffer::ReplayBuffer, storage::Storage, ratio::Float32, noise_r::Float32, max_hist::Vector{Float32})
    for it in 1:env.num_player
        par = div(env.num_player,10)
        #=
        if(it%par==0)
            for ii in 1:div(it, par)
                print("#")
            end
            #show_buffer(buffer)
            #println("score: $(length(storage.scores))")
            #println("now: $(now())")
        end=#
        #model = gpu(latest_model(storage))
        game = play_physics!(env, ratio, noise_r, storage, max_hist)
        save_game!(buffer, game)
        if(length(max_hist)>lmax_hist)
            println("max_hist: $(max_hist[end])")
            println("lmax_hist: $(lmax_hist)")
            break
        end
    end
end






function AlphaZero_ForPhysics(env::Env, envf::Env, storage::Storage)
    ld = []
    max_hist::Vector{Float32} = [-12.0f0]
    itn = 100
    lastit = 0
    ratio = env.ratio
    randr = env.ratio_r
    for it in 1:itn
        #println("=============")
        #println("it=$(it);")

        replay_buffer = init_buffer(1200, env.batch_size)
        
        run_selfplay!(env, replay_buffer, storage, ratio, randr, max_hist)
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
        val, hist = findmax(storage.scores)
        println("max score: $(val);  hist: $(hist2eq(hist))")
        if(length(max_hist)>lmax_hist)
            break
        end
    end
    
    return ld, max_hist, latest_model(storage)
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

date = 1120

function main(args::Vector{String})
    println("Start! at $(now())")
    env = init_Env(args)
    env_fc = init_Env_forcheck(args)
    
    max_hists = []
    for dd in 1:5
        #storage = init_storage(env)
        storage = init_storage(env, 2000)
        ld, max_hist, model = AlphaZero_ForPhysics(env, env_fc, storage)
        push!(max_hists, max_hist)
        #string_score = dict_copy(storage.scores)
        
        k = [keys(storage.scores)...]
        inds = findall(s -> storage.scores[s] == findmax(storage.scores)[1], k)
        println("max score:")
        for i in inds
            println("$(hist2eq(k[i])), $(storage.scores[k[i]])")
        end
    end
    ls = [length(max_hists[i]) for i in 1:length(max_hists)]
    lmin_hist = minimum(ls)
    p0 = plot(max_hists[1], linewidth=3.0, xaxis=:log, xrange=(1,lmax_hist), yrange=(0,12))
    for i in 2:length(max_hists)
        p0 = plot!(max_hists[i], linewidth=3.0, xaxis=:log, xrange=(1,lmax_hist))
    end
    savefig(p0, "/home/yoshihiro/Documents/Codes/julia/AlphaZeroForPhysics/valMAX_woNN_itr_mt$(env.max_turn)_$(date).png")
    save_data = DataFrame(hist1=max_hists[1][1:lmin_hist],hist2=max_hists[2][1:lmin_hist],hist3=max_hists[3][1:lmin_hist],hist4=max_hists[4][1:lmin_hist],hist5=max_hists[5][1:lmin_hist])
    CSV.write("./hists_woNN_mt$(env.max_turn)_$(date).csv", save_data)
    println("AlphaZero Finish!")
    
end



@time main(ARGS)