#using "coreEnv.jl"
using Random
using "env_corridor.jl"


struct TableQAgt
    n_act::Int
    init_val_Q::Float64
    ϵ::Float64
    α::Float64
    γ::Float64
    max_memory::Int
    filepath::String
end

function init_tableQAgt(path::String)
    n_act::Int = 2
    init_val_Q::Float64 = 0.0
    ϵ::Float64 = 0.1
    α::Float64 = 0.1
    γ::Float64 = 0.9
    max_memory::Int = 500
    filepath::String = path

    return n_act, init_val_Q, ϵ, α, γ, max_memory, filepath
end

mutable struct Q_table
    Q::Dict
    len_Q::Int
end

function init_Qtable()
    Q = Dict()
    lenQ::Int = 0
end

function check_and_add_obs(qagt::TableQAgt, q::Q_table, obs::String)
    if(haskey(q.Q,obs))
        q.Q[obs] = [qagt.init_val_Q] * qagt.n_act
        q.len_Q += 1

        if(q.len_Q > qagt.max_memory)
            println("観測の登録数が上限"*string(qagt.max_memory)*"に達しました")
            exit()
        end

        if((q.len_Q < 100 && q.len_Q%10 == 0)||q.len_Q%100 == 0 )
            println("the number of Q-table "*string(q.lenQ))
        end
    end
end


function decide_action(qagt::TableQAgt, q::Q_table, obs)
    
    obs = string(obs)
    check_and_add_obs(qagt, q, obs)

    if(rand()< qagt.ϵ)
        act = rand(0:qagt.n_act)
    else
        act= findall(isequal(maximum(q.Q[obs])), q.Q[obs])
    end

    return act
end

function learn(qagt::TableQAgt, q::Q_table, obs, act, rwd, done, next_obs)
    if(isnothing(rwd))
        return
    end
    
    obs = string(obs)
    next_obs = string(next_obs)

    check_and_add_obs(qagt, q, next_obs)

    if(done==true)
        target = rwd
    else
        target = rwd + qagt.γ * maximum(q.Q[next_obs])
    end

    q.Q[obs][act] = (1-qagt.α) * q.Q[obs][act] + qagt.α*target
end


function get_Q(q::Q_table, obs)
    obs = string(obs)
    if(haskey(q.Q, obs))
        val = q.Q[obs]
        Q = copy(val)
    else
        Q = nothing
    end

    return Q
end

function save_weight(qagt::TableQAgt, q::Q_table, filepath=nothing)
    if(isnothing(filepath))
        filepath = qagt.filepath * ".dat"
    end

    serialize(filepath, q.Q)
end

function load_weight(qagt::TableQAgt, q::Q_table, filepath=nothing)
    if(isnothing(filepath))
        filepath = qagt.filepath * ".dat"
    end

    q.Q = open(deserialize,filepath)
end

function mainQ(arg::Array{String,1})
    n_step::Int = 5000
    if(length(arg)>0)
        n_step = parse(Int,arg[1])
    end
    println(string(n_step)*"-stepの学習趣味レーション開始")

    en = CorridorEnv(init_corredor_env()...)
    st = stat(init_stat()...)

    agt = TableQAgt(init_tableQAgt("./test")...)
    q = Q_table(init_Qtable()...)

    obs = reset(en,st)

    for t in 1:n_step
        act = agt.decide_action(agt,q,obs)
        rwd, done, next_obs = step(en, st, act)
        learn(agt,q,obs,act,rwd,done,next_obs)
        obs = next_obs
    end

    obss = ["[1 0 2 0]", "[0 1 2 0]", "[0 0 1 0]", "[0 0 2 1]", "[1 0 0 2]", "[0 1 0 2]", "[0 0 1 2]", "[0 0 0 1]"]

    println()
    println("学習後のQ値")
    for obs0 in obss
        q_vals = get_Q(q,obs0)
        if(!isnothing(q_vals))
            println(obs0*": "*string(q.Q[obs0][0])*", "*string(q.Q[obs0][1]))
        else
            println(obs0*": ")
        end
    end

    #=
    agt.ϵ = 0.0

    t = 0
    obs = reset(en,st)
    act = nothing
    rwd = nothing
    done = nothing
    next_obs = nothing=#

    println("")
    println("学習無しシミュレーション開始")
    main()

    #=
    show_info(t, act, rwd, done, obs, true)

    while(true)
        images = render(en, st)
        imshow(images)
    end=#

end

@time mainQ(ARGS)

