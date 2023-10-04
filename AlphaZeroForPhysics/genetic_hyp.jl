using Distributed
addprocs(5)
@everywhere include("PPO_test.jl")

function PPO_hype(args::Vector{String} ,hyperparams::Vector{Float32})
    env = init_Env_quick(args, hyperparams)
    
    find_ave = 0.0
    @everywhere dist = 10

    find_ave = @distributed (+) for dd in 1:dist
        ld, max_hist, model, scores = AlphaZero_ForPhysics_hind(env)
        if(max_hist[end]>10.0)
            mini_turn = findfirst(max_hist .== minimum(max_hist[end]))
        else
            mini_turn = env.num_simulation
        end
        mini_turn/dist
    end

    return find_ave
end

function gene_search(args::Vector{String})
    schedule::Vector{Float32} = [0.01, 0.005, 0.003]
    init_hype::Vector{Float32} = [0.98, 0.95]
    score_itr = []
    hype_itr = []
    for σ in schedule
        println("===========================")
        println("sigma = ", σ)
        hype_test = []
        score = []
        hyps = collect(Iterators.product(((init_hype[1]-σ):σ:(init_hype[1]+σ)), ((init_hype[2]-2.5f0σ):(2.5f0σ):(init_hype[2]+2.5f0σ))))
        #g = Uniform(-σ, σ)
        #lam = Uniform(-2.5σ, 2.5σ)
        for i in 1:9
            #hype = init_hype + [Float32(rand(g)), Float32(rand(lam))]
            hype = init_hype .+ hyps[i]
            push!(hype_test, hype)
            push!(score, PPO_hype(args, hype))
            print("#")
        end
        println("")
        parm_ind = findmin(score)[2]
        push!(score_itr, score[parm_ind])
        println("score =  $(score[parm_ind])")
        init_hype = hype_test[parm_ind]
        push!(hype_itr, init_hype)
        println("hype =  $(init_hype)")
    end
    println(hype_itr)
    println(score_itr)
end

@time gene_search(ARGS)