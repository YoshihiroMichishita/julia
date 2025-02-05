using Distributed
using Dates
#using JET
addprocs(5)

@everywhere include("AZP_env.jl")
@everywhere include("AZP_agt.jl")

#using CUDA
@everywhere using Distributions
@everywhere using StatsBase

@everywhere mutable struct Node
    value_expect::Float32
    children::Dict{Int, Node}
    action::Int
end

@everywhere function init_node(init_val::Float32)
    return Node(init_val, Dict(), 0)
end

@everywhere function has_children(node::Node)
    return length(node.children) > 0
end

#finishの判定
@everywhere function is_finish(env::Env, agt::Agent)
    #max_turnに達したか、branchがなくなって最後のstateがval_numに達したか
    #return env.max_turn == length(agt.history) || (length(agt.branch_left) == 0 && agt.history[end]<=env.val_num)
    return (env.max_turn == length(agt.history) || isempty(agt.branch_left))
end

#行動可能なactionのリストを返す

@everywhere function legal_action(env::Env, agt::Agent)
    if(isempty(agt.history))
        return [i for i in 1:env.act_ind]
    elseif(env.max_turn-length(agt.history)<=length(agt.branch_left)+1)
        return [i for i in 1:env.val_num]
    elseif(agt.history[end]>env.val_num && agt.history[end]<=env.val_num+env.br_num)
        return [i for i in 1:env.act_ind if(i!=agt.history[end])]
    elseif(agt.history[end]==6)
        return [i for i in 2:env.act_ind-1]
    else
        return [i for i in 1:env.act_ind]
    end
end

#historyにactionを追加し、branch_leftを更新
@everywhere function apply!(env::Env, agt::Agent, act::Int)
    push!(agt.history, act)
    if act <= env.val_num
        pop!(agt.branch_left)
    elseif(act <= env.val_num+env.br_num)
        push!(agt.branch_left, act)
    end
end

@everywhere function apply!(env::Env, agt::Agent, act::Int, surprise::Float32)
    push!(agt.history, act)
    push!(agt.surprise, surprise)
    if act <= env.val_num
        pop!(agt.branch_left)
    elseif(act <= env.val_num+env.br_num)
        push!(agt.branch_left, act)
    end
end

@everywhere function coin(ϵ::Float32)
    p = [1-ϵ, ϵ]
    c = [0, 1]
    return sample(c, ProbabilityWeights(p), 1, replace=false)[1]
end

@everywhere function sample_another_action(l::Int, act_ind::Int)
    ac_v = [i for i in 1:l if(i!=act_ind)]
    act = rand(ac_v)
    return act
end

@everywhere function select_child(env::Env, node::Node)
    actions = Int.(keys(node.children))
    children = [node.children[a] for a in actions]
    score_v = [child.value_expect for child in children]
    its = findall(x -> x==maximum(score_v), score_v)
    #println(score_v)
    #println(its)
    l = length(score_v)
    it = rand(its)
    if(coin(env.frac)==0)
        return actions[it], children[it]
    else
        ita = sample_another_action(l, it)
        return actions[ita], children[ita]
    end
    #it = rand(findall(x -> x==maximum(score_v), score_v))
    #return actions[it], children[it]
end

@everywhere function backpropagate!(search_path::Vector{Node}, value::Float32)
    itr=0
    for node in search_path
        if(itr == 0)
            node.value_expect = value
        else
            ks = keys(node.children)
            l = length(k)
            node.value_expect = sum([node.children[k].value_expect for k in ks])/l
        end
    end
end

@everywhere function eval_t!(env::Env, agt::Agent, scores::Dict{Vector{Int}, Float32})
    if haskey(scores, agt.history)
        return scores[agt.history]
    else
        value = calc_score(agt.history, env) 
        scores[agt.history] = value
        return value
    end
end
#=
function best_root(node::Node)
    history = []
    nn = node
    value = 0.0f0
    while(has_children(nn))
        actions = Int.(keys(nn.children))
        children = [nn.children[a] for a in actions]
        score_v = [child.value_expect for child in children]
        it = rand(findall(x -> x==maximum(score_v), score_v))
        push!(history, actions[it])
        nn = children[it]
        value = nn.value_expect
    end
    return value, history
end=#

@everywhere function run_greedy(env::Env, agt::Agent, scores::Dict{Vector{Int}, Float32})
    root = init_node(Float32(20.0))
    
    max_value = [-10.0f0]
    A = legal_action(env, agt)
    for a in A
        root.children[a] = init_node(Float32(20.0))
    end
    score_length = 0
    count = 0
    for it in 1:env.num_simulation
        node = root
        scratch = deepcopy(agt)
        search_path = [node]

        while(!is_finish(env, scratch))
            action, node = select_child(env, node)
            apply!(env, scratch, action)
            A1 = legal_action(env, scratch)
            for a in A1
                if(!haskey(node.children, a))
                    node.children[a] = init_node(Float32(20.0))
                end
            end
            push!(search_path, node)
        end
        value = eval_t!(env, scratch, scores)
        backpropagate!(search_path, value)
        
        if(it%div(env.num_simulation, 10)==0)
            print("#")
        end
        if(score_length == length(scores))
        else
            score_length = length(scores)
            push!(max_value, findmax(scores)[1])
        end
    end
    println("")
    println("score_length: $(length(scores))")
    #return select_action(root), root
    return findmax(scores)[1],findmax(scores)[2], max_value
end

using Plots
ENV["GKSwstype"]="nul"
Plots.scalefontsizes(1.3)
date=0216
using FileIO

using DataFrames
using CSV
@everywhere lmax_hist::Int = 6200
using SharedArrays
function main(args::Vector{String})
    #args = ARGS
    println("Start!")
    @everywhere env = init_Env_hind($(args))
    
    
    @everywhere tes = 20
    #zeros(Float32, env.num_simulation, tes)
    max_hists = SharedArray(zeros(Float32, lmax_hist-env.num_player, tes))
    @sync @distributed for it in 1:tes
        agt = init_agt()
        scores = Dict{Vector{Int}, Float32}()
        @time val, hist, max_value = run_greedy(env, agt, scores)
        L = min(length(max_value), lmax_hist-env.num_player)
        max_hists[1:L,it] = max_value[1:L]
        println("value: $(val)")
        println("history: $(hist)")
        #push!(max_values, max_value)
        #mv = vcat(max_value, 1.0f2*ones(Float32, env.num_simulation-length(max_value)))
        #max_values[:, it] = mv
    end
    max_values = Matrix(max_hists)
    p = plot(max_values[:,1],gridwidth=2.0, linewidth=3.0, xaxis=:log, xticks=([1, 10, 100, 1000]), yrange=(0,12), label="trial1")
    for it in 2:tes
        p=plot!(max_values[:,it], linewidth=3.0, label="trial$(it)") 
    end
    savefig(p, "./ϵ-greedy_ValItr_n$(env.num_simulation)_η$(env.frac)_$(date).png")

    save_data = DataFrame(max_values, :auto)
    #save_data = DataFrame(hist1=max_hists[1][1:lmax_hist-100],hist2=max_hists[2][1:lmax_hist-100],hist3=max_hists[3][1:lmax_hist-100],hist4=max_hists[4][1:lmax_hist-100],hist5=max_hists[5][1:lmax_hist-100])
    CSV.write("./hists_mt$(env.max_turn)_$(date)_egreedy.csv", save_data)
end
  
@time main(ARGS)
