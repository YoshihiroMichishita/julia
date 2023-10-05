using Distributed
addprocs(5)

@everywhere include("AZP_DNN_single_CPU.jl")

function AZP_hype(args::Vector{String} ,hyperparams::Vector{Float32})
    env = init_Env_quick(args, hyperparams)
    
    find_ave = 0.0
    @everywhere dist = 10

    find_ave = @distributed (+) for dd in 1:dist
        storage = init_storage(env)
        max_hist = AlphaZero_ForPhysics_hind(env, storage)
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
    dα::Vector{Float32} = [0.2, 0.1]
    dCi::Vector{Float32} = [0.5, 0.25]
    dCb::Vector{Int} = [24, 12]
    init_hype::Vector{Float32} = [0.4, 72, 1.25]
    score_itr = []
    hype_itr = []
    for it in 1:2
        println("===========================")
        println("sigma = ", σ)
        hype_test = []
        score = []
        hyps = collect(Iterators.product(((init_hype[1]-dα[it]),init_hype[1],(init_hype[1]+dα[it])), ((init_hype[2]-dCb[it]),init_hype[2],(init_hype[2]+dCb[it])), ((init_hype[3]-dCi[it]),init_hype[3],(init_hype[3]+dCi[it]))))
        #g = Uniform(-σ, σ)
        #lam = Uniform(-2.5σ, 2.5σ)
        for hype in hyps
            #hype = init_hype + [Float32(rand(g)), Float32(rand(lam))]
            #hype = init_hype .+ hyps[i]
            push!(hype_test, hype)
            push!(score, AZP_hype(args, hype))
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