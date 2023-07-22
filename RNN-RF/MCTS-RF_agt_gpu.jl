
using Distributions
using StatsBase
using CUDA
include("MCTS-RF_env.jl")

mutable struct Node
    visit_count::Int
    prior::Float32
    value_sum::Float32
    children::Dict{Int, Node}
    action::Int
end

function init_node(prior::Float32)
    return Node(0, prior, 0.0, Dict(), 0)
end

function has_children(node::Node)
    return length(node.children) > 0
end

function st_value(node::Node)
    if node.visit_count == 0
        return 0.0
    else
        return node.value_sum / node.visit_count
    end
end

#it stores sigle game history and child visit pi
mutable struct Agent
    #counts
    history::Vector{Int}
    branch_left::Vector{Int}
    child_visit_pi::Vector{Vector{Float32}}
end

function init_agt()
    return Agent([], [-1], [])
end

#finishの判定
function is_finish(env::Env, agt::Agent)
    #max_turnに達したか、branchがなくなって最後のstateがval_numに達したか
    #return env.max_turn == length(agt.history) || (length(agt.branch_left) == 0 && agt.history[end]<=env.val_num)
    return (env.max_turn == length(agt.history) || length(agt.branch_left) == 0)
end


#行動可能なactionのリストを返す
function legal_action(env::Env, agt::Agent)
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
function apply!(env::Env, agt::Agent, act::Int)
    push!(agt.history, act)
    if act <= env.val_num
        pop!(agt.branch_left)
    elseif(act <= env.val_num+env.br_num)
        push!(agt.branch_left, act)
    end
end

#child_visit_piの計算
function store_search_statistics!(env::Env, root::Node, agt::Agent)
    visit_pi = zeros(Float32, env.act_ind)
    actions = Int.(keys(root.children))
    sum_visits = sum([root.children[a].visit_count for a in actions])
    for a in actions
        visit_pi[a] = root.children[a].visit_count/sum_visits
        if(isnan(visit_pi[a]))
            println("root.children[a].visit_count: $(root.children[a].visit_count)")
            println("sum_visits: $(sum_visits)")
            visit_pi[a] = 1f-6
        end
    end
    push!(agt.child_visit_pi, visit_pi)
end

function make_image(env::Env, agt::Agent, turn::Int)
    input_data = zeros(Int, env.input_dim, 1)
    for act_ind in 1:env.act_ind
        ind = findall(x->x==act_ind, agt.history[1:turn])
        for it in ind
            input_data[(act_ind-1)*env.max_turn+it,1] = 1
        end
    end
    return input_data
    #append!(copy(agt.history[1:turn]), zeros(Int, env.max_turn-turn))
end

function make_target(env::Env,agt::Agent, turn::Int)
    return [agt.child_visit_pi[turn]; calc_score(agt.history, env)]
end

function calc_score_his(history::Vector{Int}, env::Env, scores::Dict{Vector{Int}, Float32})
    if haskey(scores, history)
        return scores[history]
    else
        score = calc_score(history, env)
        #scores[history] = score
        return score
    end
end

function make_target(env::Env,agt::Agent, scores::Dict{Vector{Int}, Float32}, turn::Int)
    return [agt.child_visit_pi[turn]; calc_score_his(agt.history, env, scores)]
end

function add_exploration_noise!(env::Env, node::Node)
    actions = Int.(keys(node.children))
    noise = rand(Dirichlet(env.α * ones(Float32, length(actions))))
    for it in 1:size(actions)[1]
        node.children[actions[it]].prior = node.children[actions[it]].prior * (1-env.frac) + noise[it] * env.frac
        if(isnan(node.children[actions[it]].prior))
            println("noise: $(noise)")
        end
    end
end

function add_exploration_noise!(env::Env, node::Node, ratio::Float32)
    actions = Int.(keys(node.children))
    noise = rand(Dirichlet(env.α * ones(Float32, length(actions))))
    for it in 1:size(actions)[1]
        node.children[actions[it]].prior = node.children[actions[it]].prior * (1-ratio*env.frac) + noise[it] * ratio* env.frac
        if(isnan(node.children[actions[it]].prior))
            println("noise: $(noise)")
        end
    end
end

function ucb_score(env::Env, parent::Node, child::Node)
    pb_c = log((parent.visit_count + env.Cb + 1) / env.Cb) + env.Ci
    pb_c *= sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = child.value_sum / (child.visit_count + 1)
    ans = prior_score + value_score
    if(isnan(ans))
        println("prior_score: $(prior_score)")
        println("value_score: $(value_score)")
        println("child.visit_count: $(child.visit_count)")
        println("child.value_sum: $(child.value_sum)")
    end
    return ans
    #return prior_score + value_score
end

function ucb_score(env::Env, parent::Node, child::Node, ratio::Float32)
    pb_c = log((parent.visit_count + env.Cb + 1) / env.Cb) + env.Ci
    pb_c *= sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = child.value_sum / (child.visit_count + 1)
    ans = ratio*prior_score + value_score
    if(isnan(ans))
        println("prior_score: $(prior_score)")
        println("value_score: $(value_score)")
        println("child.visit_count: $(child.visit_count)")
        println("child.value_sum: $(child.value_sum)")
    end
    return ans
    #return prior_score + value_score
end
using BSON: @save
function select_child(env::Env, node::Node, model::Chain)
    actions = Int.(keys(node.children))
    children = [node.children[a] for a in actions]
    score_v = [ucb_score(env, node, child) for child in children]
    its = findall(x -> x==maximum(score_v), score_v)
    if(isempty(its))
        println("score_v: $(score_v)")
        @save "BadModel.bson" model
    end
    it = rand(its)
    #it = rand(findall(x -> x==maximum(score_v), score_v))
    return actions[it], children[it]
end

function select_child(env::Env, node::Node, model::Chain, ratio::Float32)
    actions = Int.(keys(node.children))
    children = [node.children[a] for a in actions]
    score_v = [ucb_score(env, node, child, ratio) for child in children]
    its = findall(x -> x==maximum(score_v), score_v)
    if(isempty(its))
        println("score_v: $(score_v)")
        @save "BadModel.bson" model
    end
    it = rand(its)
    #it = rand(findall(x -> x==maximum(score_v), score_v))
    return actions[it], children[it]
end

function backpropagate!(search_path::Vector{Node}, value::Float32)
    for node in search_path
        node.value_sum += value
        if(isnan(node.value_sum))
            println("value Nan!!!!!!!!")
            println("value: $(value)")
            println(node)
        end
        node.visit_count += 1
    end
end
#=
function select_action(root::Node)
    actions = Int.(keys(root.children))
    #visits = [child.visit_count for child in root.children]
    visits = [root.children[a].visit_count for a in actions]
    return actions[argmax(visits)]
end
=#

function select_action(env::Env, root::Node, agt::Agent)
    actions = Int.(keys(root.children))
    visits = [root.children[a].visit_count for a in actions]
    if(length(agt.history)<env.max_turn/2)
        return actions[rand(Categorical(softmax(visits)))]
    else
        return actions[argmax(visits)]
    end
end


function evaluate!(env::Env, agt::Agent,node::Node, model::Chain)
    Y = model(cu(make_image(env, agt, length(agt.history))))
    Yc = cpu(Y)
    value = Yc[end,1] 
    pol_log = Yc[1:end-1,1]
    A = legal_action(env, agt)
    policy = softmax([pol_log[a] for a in A])
    
    for it in 1:size(A)[1]
        if(isnan(policy[it]))
            println("input: $(make_image(env, agt, length(agt.history)))")
            println("Y: $(Yc)")
            println("policy: $(policy)")
            println("pol_log: $(pol_log)")
        end
        node.children[A[it]] = init_node(policy[it])
    end
    return value
end

function run_MCTS(env::Env, agt::Agent, model::Chain)
    root = init_node(Float32(0.0))
    value = evaluate!(env, agt, root, model)
    add_exploration_noise!(env, root)
    for it in 1:env.num_simulation
        node = root
        scratch = deepcopy(agt)
        search_path = [node]
        while(!is_finish(env, scratch) && has_children(node))#has_children(node)
            action, node = select_child(env, node, model)
            apply!(env, scratch, action)
            push!(search_path, node)
        end
        value = evaluate!(env, scratch, node, model)
        backpropagate!(search_path, value)
    end
    #return select_action(root), root
    return select_action(env, root, agt), root
end

function run_MCTS(env::Env, agt::Agent, model::Chain, ratio::Float32)
    root = init_node(Float32(0.0))
    value = evaluate!(env, agt, root, model)
    add_exploration_noise!(env, root)
    for it in 1:env.num_simulation
        node = root
        scratch = deepcopy(agt)
        search_path = [node]
        while(!is_finish(env, scratch) && has_children(node))#has_children(node)
            action, node = select_child(env, node, model, ratio)
            apply!(env, scratch, action)
            push!(search_path, node)
        end
        value = evaluate!(env, scratch, node, model)
        backpropagate!(search_path, value)
    end
    #return select_action(root), root
    return select_action(env, root, agt), root
end


function play_physics!(env::Env, model::Chain)
    agt = init_agt()
    while(!is_finish(env, agt))
        action, root = run_MCTS(env, agt, model)
        apply!(env, agt, action)
        store_search_statistics!(env, root, agt)
    end
    return agt
end

function play_physics!(env::Env, model::Chain, ratio::Float32)
    agt = init_agt()
    while(!is_finish(env, agt))
        action, root = run_MCTS(env, agt, model, ratio)
        apply!(env, agt, action)
        store_search_statistics!(env, root, agt)
    end
    return agt
end

function play_physics!(env::Env,agt::Agent, model::Chain)
    #agt = init_agt()
    while(!is_finish(env, agt))
        action, root = run_MCTS(env, agt, model)
        apply!(env, agt, action)
        store_search_statistics!(env, root, agt)
    end
end

function play_physics!(env::Env,agt::Agent, model::Chain, ratio::Float32)
    #agt = init_agt()
    while(!is_finish(env, agt))
        action, root = run_MCTS(env, agt, model, ratio)
        apply!(env, agt, action)
        store_search_statistics!(env, root, agt)
    end
end

#using BSON: @save
using BSON: @load

function check_RL()
    env = init_Env(ARGS)
    @load ARGS[21] model
    for it in 1:10
        game = play_physics!(env, model)
        score = calc_score(game.history, env)
        println("$(game.history), score:$(score)")
    end
end

#check_RL()
#=
function test()
    env = init_Env(ARGS)
    model = Chain(Dense(zeros(Float32, env.output,env.input_dim)))
    agt = play_physics!(env, model)
    println(agt.history)
end

test()=#
