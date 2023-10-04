include("MZP_env.jl")

mutable struct Node
    visit_count::Int
    prior::Float32
    score_sum::Float32
    children::Vector{Node}
    hidden_state::Vector{Float32}
    reward::Float32
end

function init_node()
    visit_count::Int = 0
    prior::Float32 = 0.0f0
    score_sum::Float32 = 0.0f0
    children::Vector{Node} = []
    hidden_state::Vector{Float32} = []
    reward::Float32 = 0.0f0
    return Node(visit_count, prior, score_sum, children, hidden_state, reward)
end

function node_value(node::Node)
    if(node.visit_count==0)
        return 0.0f0
    end
    return node.score_sum/node.visit_count
end

mutable struct Game
    history::Vector{Int}
    rewards::Vector{Float32}
    child_visits::Vector{Int}
    root_value::Float32
    discount::Float32
    action_size::Int
    done::Bool
end

function init_game(action_size::Int, discount::Float32)
    history::Vector{Int} = []
    rewards::Vector{Float32} = []
    child_visits::Vector{Int} = []
    root_values::Vector{Float32} = [] 
    discount::Float32 = discount
    action_size::Int = action_size
    done::Bool = false
    return Game(history, rewards, child_visits, root_values, discount, action_size, done)
end

function apply!(env::Env, game::Game, action::Int)
    push!(game.history, action)
    if(game.done)
        reward = calc_score(game.history, env)
    else
        reward = 0.0f0
    end
    push!(game.rewards, reward)
end

function store_search_statistics!(game::Game, root::Node, action_size::Int)
    child_visits = zeros(Int, action_size)
    for action in root.children
        child_visits[action] = root.children[action].visit_count
    end
    push!(game.child_visits, child_visits)
    push!(game.root_value, node_value(root))
end