using LinearAlgebra
using Random
using Flux
using Distributions
using OneHotArrays

using DataFrames
using CSV
using BSON: @save
using BSON: @load
using Plots
ENV["GKSwstype"]="nul"

struct Env
    Ns::Int #siteの数
    num_var::Int  #operatorの数(ψ_iとψ'_i)
    num_br::Int #binary operatorの数。多分上記の +, -, *, -i[,], {,} の５つ？([,]などは*と+で表現できるが、のちに変数は２回以上使わないという制約を課したいので、変数を一回使うだけで交換関係を表現できるように導入しておく)100の位に格納
    #num_fn::Int #unitary operatorの数。 多分exp[], log[], diag[]の３つ。100の位に格納
    num_ter::Int
    num_tot::Int

    n_level::Int
    n_batch::Int
    penalty::Float64
    γ::Float64# discount

    #op_fn
    op_br

    conv_ac
end

function set_fn()
    function daig_mat(M)
        e,v = eigen(M)
        return diagm(e)
    end
    return [x->exp.(x), x->log.(x), x->daig_mat.(x)]
end

function set_br()
    return [(x,y)->(x.+y), (x,y)->(x.-y), (x,y)->(x.*y)]
    #return [(x,y)->(x+y), (x,y)->(x-y), (x,y)->(x*y), (x,y)->-1.0im*(x*y .- y*x), (x,y)->(x*y .+ y*x)/2]
end

function init_Env(N::Int, b::Int, p::Float64, γ::Float64)
    Ns = N #siteの数
    num_var = 2*Ns  #operatorの数(ψ_iとψ'_i)
    num_br = 3 #binary operatorの数。多分上記の +, -, *, -i[,], {,} の５つ？([,]などは*と+で表現できるが、のちに変数は２回以上使わないという制約を課したいので、変数を一回使うだけで交換関係を表現できるように導入しておく)100の位に格納
    #num_fn = 3 #unitary operatorの数。 多分exp[], log[], diag[]の３つ。100の位に格納
    num_ter = 1
    #num_tot = num_var + num_br + num_fn
    num_tot = num_var + num_br

    n_level = 2
    n_batch = b

    penalty = p

    #op_fn = set_fn()
    op_br = set_br()

    conv_ac = zeros(Int, num_tot)
    for i in 1:num_var
        conv_ac[i] = 10i
    end
    for i in 1:num_br
        conv_ac[i+num_var] = i
    end
    #=
    for i in 1:num_fn
        conv_ac[i+num_var+num_br] = 100i
    end=#

    #return Ns, num_var, num_br, num_fn, num_ter, num_tot, n_level, n_batch, op_fn, op_br, conv_ac
    return Ns, num_var, num_br, num_ter, num_tot, n_level, n_batch, penalty, γ, op_br, conv_ac
end

struct DQN 
    width::Int
    act_MAX::Int
    ϵ::Float64# for ϵ-greedy
    #γ::Float64# discount
    prob::Vector{Float64}
    rand_ac::Vector{Float64}
end

function init_DQN(w::Int, a_MAX::Int, ϵ::Float64, en::Env)
    prob = [ϵ, 1-ϵ]
    rand_ac = ones(Float64, en.num_tot)./en.num_tot
    return w, a_MAX, ϵ, prob, rand_ac
end

struct Sample
    var::Vector{Vector{Float64}}
    gauge_sample::Matrix{Float64}
end

function Gene_Rand_Var(en::Env)
    var::Vector{Vector{Float64}} = []
    θ = zeros(Float64,2)
    push!(var, θ)
    for i in 1:en.Ns-1
        θ += [pi*(1.0+rand(Float64))/2en.Ns , pi*(0.5-rand(Float64))/en.Ns]
        push!(var, θ)
    end
    return var
end

function get_Sample(en::Env)
    var = Gene_Rand_Var(en)
    gauge_sample = 2pi*rand(Float64, en.n_batch, en.Ns)
    return var, gauge_sample
end


mutable struct Agt
    #model
    state::Vector{Int}
    branch::Vector{Int}
    q_table::Matrix{Float32}
    ex_table::Array{Int, 3} #経験数(2or3手前までの式, ２手前までのbranch, action)
end

function init_agt(en::Env, dq::DQN)
    #model = Chain(Dense(dq.act_MAX, dq.width, relu, init=Flux.zeros32), Dense(dq.width, dq.width, relu, init=Flux.zeros32), Dense(dq.width, en.num_tot, init=Flux.zeros32))
    state::Vector{Int}=[]
    branch::Vector{Int}=[]
    q_table = zeros(Float32, dq.act_MAX, en.num_tot)
    ex_table = zeros(Int, (en.num_tot+1)^3, (en.num_br+1)^2, en.num_tot)
    #return model, state, branch, q_table
    return state, branch, q_table, ex_table
end

function experience(en::Env,ag::Agt)
    recent_num = 1
    br_num = 1
    for i in 1:min(3, length(ag.state))
        recent=ag.state[end-i+1]
        if(recent<10)#br
            recent_num += ((en.num_tot+1)^(i-1))*(recent+ en.num_var)
        else#var
            recent_num += ((en.num_tot+1)^(i-1))*(div(recent,10))
        end
    end
    for i in 1:min(2, length(ag.branch))
        br = ag.branch[end-i+1]
        br_num += ((en.num_br+1)^(i-1))*br
    end
    return sqrt.(log(sum(ag.ex_table[recent_num, br_num,:])+1)./(ag.ex_table[recent_num, br_num,:].+1))
end

function exp_update!(en::Env,ag::Agt, act::Int)
    recent_num = 1
    br_num = 1
    act_num = 0
    for i in 1:min(3, length(ag.state))
        recent=ag.state[end-i+1]
        if(recent<10)#br
            recent_num += ((en.num_tot+1)^(i-1))*(recent+ en.num_var)
        else#var
            recent_num += ((en.num_tot+1)^(i-1))*(div(recent,10))
        end
    end
    for i in 1:min(2, length(ag.branch))
        br = ag.branch[end-i+1]
        br_num += ((en.num_br+1)^(i-1))*br
    end
    if(act<10)#br
        act_num += act+ en.num_var
    else#var
        act_num += div(act,10)
    end
    ag.ex_table[recent_num, br_num, act_num] += 1
end

function UCT(q_t::Vector{Float32}, ρ::Float64, en::Env, ag::Agt)
    return q_t + ρ*experience(en, ag)
end

function action_vec(q_t::Vector{Float32}, en::Env, ag::Agt)
    ac_prob = softmax(UCT(q_t,sqrt(2.0)*maximum(abs.(q_t)), en, ag))
    act = rand(Categorical(ac_prob))
    return onehot(Int, 1:en.num_tot, act)
end

#function decide_action!(en::Env, dq::DQN, ag::Agt, t::Int)
function decide_action!(en::Env, dq::DQN, ag::Agt, model, t::Int)
    rem_turn = zeros(Int, dq.act_MAX + 1 - t)
    st_vec = vcat(ag.state, rem_turn)
    #q_t = ag.model(st_vec)
    q_t = model(st_vec)
    act = en.conv_ac' * action_vec(q_t, en, ag)
    exp_update!(en, ag, act)
    return q_t, act
end

function decide_action(en::Env, dq::DQN, ag::Agt, model, t::Int)
    rem_turn = zeros(Int, dq.act_MAX + 1 - t)
    st_vec = vcat(ag.state, rem_turn)
    q_t = model(st_vec)
    ac_prob = softmax(q_t)
    act = en.conv_ac' * (onehot(Int, 1:en.num_tot, rand(Categorical(ac_prob))))
    exp_update!(en, ag, act)
    return q_t, act
end

#return the penalty of the rule violation

function rule_violate(ag::Agt, ac::Int)
    if(length(ag.state)>0)
        if(ac<10)
            #if(ac>999) #branch
            if(length(ag.state)>0)
                if(ac == ag.state[end])
                    return true
                elseif(ac<3 && ag.state[end]<3)
                    return true
                else
                    return false
                end
            else
                return false
            end
            #else
                #=
                if((ac==1 && ag.state[end]==2) || (ac==2 && ag.state[end]==1))
                    #println("fn violation!")
                    return true
                elseif(ac== ag.state[end])
                    #println("fn violation!")
                    return true
                else
                    return false
                end
                =#
            #end
        else
            if(length(findall(isequal(ac),ag.state))==0)
                return false
            else
                #println("reuse the same var!")
                return true
            end
        end
    else
        return false
    end
end

function rule_violate(state::Vector{Int}, ac::Int)
    if(length(state)>0)
        if(ac<10)
            #if(ac>999) #branch
            if(length(state)>0)
                if(ac == state[end])
                    return true
                elseif(ac<3 && state[end]<3)
                    return true
                else
                    return false
                end
            else
                return false
            end
            #else
                #=
                if((ac==1 && ag.state[end]==2) || (ac==2 && ag.state[end]==1))
                    #println("fn violation!")
                    return true
                elseif(ac== ag.state[end])
                    #println("fn violation!")
                    return true
                else
                    return false
                end
                =#
            #end
        else
            if(length(findall(isequal(ac),state))==0)
                return false
            else
                #println("reuse the same var!")
                return true
            end
        end
    else
        return false
    end
end

#Check the inveriance and add it to the loss
function VarToLoss(var::Vector{Matrix{ComplexF64}})
    loss = 0.0
    sw = (size(var[1])[1]==size(var[1])[2])
    if(sw)
        for i in 2:size(var)[1]
            for j in 1:i
                loss += abs(tr(var[i])-tr(var[j]))^2
            end
        end
    else
        for i in 2:size(var)[1]
            for j in 1:i
                loss += sum((abs.(var[i]-var[j])).^2)
            end
        end
    end
    return loss/size(var)[1]
end

function VarToLoss(var::Vector{ComplexF64})
    loss = 0.0
    for i in 2:size(var)[1]
        for j in 1:i
            loss += real((var[i]-var[j])'*(var[i]-var[j]))            
        end
    end
    return loss/size(var)[1]
end

function VarToLoss(var::Vector{Vector{ComplexF64}})
    loss = 0.0
    for i in 2:size(var)[1]
        for j in 1:i
            loss += real((var[i]-var[j])'*(var[i]-var[j]))            
        end
    end
    return loss/size(var)[1]
end

function VarToLoss(var::Vector{Adjoint{ComplexF64, Vector{ComplexF64}}})
    loss = 0.0
    for i in 2:size(var)[1]
        for j in 1:i
            loss += real((var[i]-var[j])*(var[i]-var[j])')            
        end
    end
    return loss/size(var)[1]
end

#(θ,ϕ) => ψ
function wave_fn(var::Vector{Float64}, sw::Int)
    if(sw==1)
        wv_fn = ([cos(var[1]), sin(var[1])*exp(1.0im*var[2])])'
    else
        wv_fn = [cos(var[1]), sin(var[1])*exp(1.0im*var[2])]
    end
    return wv_fn
end

#=
function Fn_Gauge(en::Env, sample::Sample, st::Vector{Int}, var_sub, var_now)
    if(length(st)==0)
        return VarToLoss(var_now)
    else
        ac = pop!(st)
    end
    if(ac<100)
        if(var_now != nothing)
            var_sub = var_now
        end
        i_s = ac%(en.Ns+1) + 1
        sw = div(ac, en.Ns+1)
        var_now = [exp(1.0im*sample.gauge_sample[b,ac])*wave_fn(sample.var[i_s], sw) for b in 1:en.n_batch]
    elseif(ac < 100)
        var_now = en.op_fn[ac%100](var_now)
    else
        var_sub, var_now = en.op_br[ac%100](var_sub, var_now)
    end
    Fn_Gauge(en, sample, st, var_sub, var_now)
end
=#


function Fn_Gauge!(en::Env, sample::Sample, st::Vector{Int}, var_now, var_sub1, var_sub2, it::Int)
    if(it==0)
        return VarToLoss(var_now)
    else
        ac = st[it]
    end
    if(rule_violate(st[1:it-1], st[it]))
        return en.penalty
    else
        if(ac>9)
            #println("var")
            var_sub1 = var_now
            var_sub2 = var_sub1
            i_s = (div(ac,10)-1)%en.Ns + 1
            sw = div(div(ac,10), en.Ns+1)
            var_now = [exp((2sw-1)*1.0im*sample.gauge_sample[b,i_s])*wave_fn(sample.var[i_s], sw) for b in 1:en.n_batch]
            #elseif(ac < 100)
            #println("fn")
            #=
            if(typeof(var_now)==Vector{Matrix{ComplexF64}})
                var_now = en.op_fn[div(ac,100)](var_now)
            else
                return 100.0
            end=#
            #=
            try
                var_now = en.op_fn[div(ac,100)](var_now)
            catch
                return 100.0
            end
            =#
        else
            #println("br")
            #=
            if(div(ac,100)<3 && typeof(var_now)!=typeof(var_sub1))
                return 100.0
            elseif(div(ac,100)==3 && typeof(var_now)==typeof(var_sub1) && typeof(var_now)!=Vector{Matrix{ComplexF64}})
                return 100.0
            else
                var_now = en.op_br[div(ac,100)](var_sub1, var_now)
                var_sub1 = nothing
                var_sub1 = var_sub2
            end=#
            try
                #var_now = en.op_br[div(ac,100)](var_sub1, var_now)
                var_now = en.op_br[ac](var_sub1, var_now)
                var_sub1 = nothing
                var_sub1 = var_sub2
            catch
                return en.penalty
            end
        end
        #println("var_now:$(var_now), var1:$(var_sub1), var2:$(var_sub2)")
        Fn_Gauge!(en, sample, st, var_now, var_sub1, var_sub2, it-1)
    end
end



function reward(en::Env, sample::Sample, ag::Agt)
    #var = Gene_Rand_Var()
    T = length(ag.state)
    #gauge_sample = 2pi*rand(Float64, n_batch, Ns)
    #st_copy = copy(ag.state)
    l = Fn_Gauge!(en, sample, ag.state, nothing, nothing, nothing, T)
    return -l + 1.0
end

function q_update!(en::Env, ag::Agt, r::Float64)
    T = length(ag.state)
    q_max = Float32(r)
    #println(q_max)
    for t in 1:T
        ag.q_table[T-t+1, act_ind(ag.state[T-t+1],en)] = q_max
        #println(q_max)
        q_max = en.γ * maximum(ag.q_table[T-t+1,:])
    end
end

#function Search!(en::Env, dq::DQN, sample::Sample, ag::Agt)
function Search!(en::Env, dq::DQN, sample::Sample, ag::Agt, model)
    r = 0.0
    # = zeros(Float32, act_MAX)
    #q_table = []
    ag.state = []
    ag.branch = []
    for turn in 1:dq.act_MAX
        #println(turn)
        #ag.q_table[turn,:], act = decide_action!(en, dq, ag, turn)
        ag.q_table[turn,:], act = decide_action!(en, dq, ag,model, turn)
        #push!(ag.state, act)
        if(act <10) #actionでbinaryを選んだ場合、２つに分岐するので分岐点を覚えておくためにbranchに入れておく
            push!(ag.branch, act)
        else
            if(size(ag.branch)[1]==0)#残りのbranchがなければ関数形が完成しているので終了
                push!(ag.state, act)
                break;
            else #branchが残っていれば、下っ側を埋めていく
                b = pop!(ag.branch)
            end
        end
        if(rule_violate(ag, act)) #rule違反をしていたら、罰則(負の報酬)を与えて終了
            push!(ag.state, act)
            r = -en.penalty
            break;
        end

        push!(ag.state, act)
        if(turn == dq.act_MAX)
            r = -en.penalty
        end
        #r[turn] = reward(state, act)
    end
    if(r==0.0)
        r = reward(en, sample, ag)
    end
    q_update!(en, ag, r)
end
#=
function replay!(en::Env, dq::DQN, sample::Sample, ag::Agt, model, best_state::Vector{Int})
    l = length(best_state)
    st = zeros(Float32, en.num_tot)
    for i in 1:l
        ag.q_table[i,:] = model(st)
        st[i] = best_state[i]        
    end
    ag.state = best_state
end=#

function act_ind(ac::Int, en::Env)
    id = 0
    if(ac<10)
        id = en.num_var + ac
    else
        id = div(ac,10)
    end
    #=
    if(ac<100)
        id = ac
    elseif(ac>999)
        id = en.num_var + div(ac,100)
    else
        id = en.num_var + en.num_br + div(ac,10)
    end=#
    return id
end

#function loss(dq::DQN,ag::Agt)
function loss(dq::DQN,ag::Agt, model)
    T = length(ag.state)
    l=0.0
    for t in 1:T
        if(t==1)
            st_turn::Vector{Int} = []
        else
            st_turn = ag.state[1:t-1]
        end
        rem_turn = zeros(Int, dq.act_MAX - (t-1))
        st = vcat(st_turn, rem_turn) 
        #q = ag.model(st)
        q = model(st)
        l += sum((q - ag.q_table[t, :]).^2)
        #l += q'*q
    end
    return l
end



#=
function train_search(en::Env, dq::DQN, sample::Sample, ag::Agt)
    #state::Vector{Int} = []
    #branch::Vector{Int} = []
    r = 0.0
    # = zeros(Float32, act_MAX)
    #q_table = []
    for turn in 1:dq.act_MAX
        ag.q_table[turn,:], act = decide_action!(en, dq, ag, turn)
        if(act > 999) #actionでbinaryを選んだ場合、２つに分岐するので分岐点を覚えておくためにbranchに入れておく
            push!(ag.branch, act)
        elseif(act < 100)
            if(size(ag.branch)[1]==0)#残りのbranchがなければ関数形が完成しているので終了
                push!(ag.state, act)
                break;
            else #branchが残っていれば、下っ側を埋めていく
                b = pop!(ag.branch)
            end
        end
        if(rule_violate(ag.state, act)) #rule違反をしていたら、罰則(負の報酬)を与えて終了
            r = -100.0
            break;
        end

        push!(ag.state, act)
        #r[turn] = reward(state, act)
    end
    println(length(ag.state))
    r = reward(en, sample, ag)
    println(length(ag.state))
    
    return loss(r, ag.q_table, ag.state)
end
=#

function RandPolitics(en::Env, dq::DQN, sample::Sample, ag::Agt, model)
    ag.state = []
    ag.branch = []
    #branch::Vector{Int} = []
    r = 0.0
    # = zeros(Float32, act_MAX)
    #q_table = []
    for turn in 1:dq.act_MAX
        #ag.q_table[turn,:], act = decide_action!(en, dq, ag, turn)
        #ag.q_table[turn,:], act = decide_action!(en, dq, ag,model, turn)
        ag.q_table[turn,:], act = decide_action(en, dq, ag, model, turn)
        if(rule_violate(ag, act)) #rule違反をしていたら、罰則(負の報酬)を与えて終了
            r = -en.penalty
            push!(ag.state, act)
            break;
        end
        push!(ag.state, act)
        if(act < 10) #actionでbinaryを選んだ場合、２つに分岐するので分岐点を覚えておくためにbranchに入れておく
            push!(ag.branch, act)
        else
            if(size(ag.branch)[1]==0)#残りのbranchがなければ関数形が完成しているので終了
                #push!(ag.state, act)
                break;
            else #branchが残っていれば、下っ側を埋めていく
                b = pop!(ag.branch)
            end
        end
        #push!(ag.state, act)
        
        #r[turn] = reward(state, act)
    end
    if(r == 0.0)
        r = reward(en, sample, ag)
    end
    
    return r, ag.state
end

function init_optimistic!(η::Float64, dq::DQN, model)
    test = zeros(Float32, dq.act_MAX)
    l = 100.0
    while (l<0.1)
        grads = Flux.gradient(Flux.params(model)) do
            #l = loss(dq,ag)
            l = sum(model(test).^2)
        end
        l = sum(model(test).^2)
        Flux.Optimise.update!(ADAM(η), Flux.params(model), grads)
    end
end

function show_formula(state::Vector{Int})
    for i in 1:length(state)
        st = state[i]
        if(st<10)
            if(st==1)
                print("+")
            elseif(st==2)
                print("-")
            elseif(st==3)
                print("*")
            end
        else
            if(st==10)
                print("ψ^†")
            elseif(st==20)
                print("ψ")
            end
        end
    end
    print(": ")
end

function main(arg::Array{String,1})
    #N_s, n_batch
    en = Env(init_Env(parse(Int, arg[1]), parse(Int, arg[2]), parse(Float64, arg[7]), parse(Float64, arg[8]))...)
    #width, act_MAX, ϵ
    dq = DQN(init_DQN(parse(Int, arg[3]), parse(Int, arg[4]), parse(Float64, arg[5]), en)...)
    ag = Agt(init_agt(en, dq)...)

    #model = Chain(Dense(dq.act_MAX, dq.width, relu), Dense(dq.width, dq.width, relu), Dense(dq.width, dq.width, relu) , Dense(dq.width, dq.width, relu), Dense(dq.width, dq.width, relu), Dense(dq.width, en.num_tot))
    model = Chain(Dense(dq.act_MAX, dq.width, relu, init=Flux.randn32), Dense(dq.width, dq.width, relu, init=Flux.randn32), Dense(dq.width, dq.width, relu, init=Flux.randn32) , Dense(dq.width, dq.width, relu, init=Flux.randn32), Dense(dq.width, dq.width, relu, init=Flux.randn32), Dense(dq.width, en.num_tot, init=Flux.randn32))
    #model = Chain(Dense(dq.act_MAX, dq.width), Dense(dq.width, dq.width) , Dense(dq.width, dq.width), Dense(dq.width, en.num_tot))
    #, Dense(dq.width, dq.width, relu) , Dense(dq.width, dq.width, relu),

    
    η = parse(Float64, arg[5])
    #initialization
    y = model(zeros(Float32, dq.act_MAX))
    println("init_optimistic check!")
    println(y)
    init_optimistic!(η, dq, model)
    println(model(zeros(Float32, dq.act_MAX)))
    println("===================")

    ll_MAX = parse(Int, arg[6])
    ll_it = zeros(Float64, ll_MAX)
    best_score = -100.0
    best_state::Vector{Int} = []
    n_replay::Int = 0
    noup::Int = 0
    sample = Sample(get_Sample(en)...)
    for it in 1:ll_MAX
        #var = Gene_Rand_Var()
        #gauge_sample = 2pi*rand(Float64, n_batch, Ns)
        #sample = Sample(get_Sample(en)...)
        #Search!(en, dq, sample, ag)
        Search!(en, dq, sample, ag, model)
        #println(ag.state)
        #ll = 0.0
        #grads = Flux.gradient(Flux.params(ag.model)) do
        grads = Flux.gradient(Flux.params(model)) do
            #l = loss(dq,ag)
            loss(dq,ag, model)/length(ag.state)
        end
        ll_it[it] = loss(dq,ag, model)
        
        #Flux.Optimise.update!(ADAM(), Flux.params(ag.model), grads)
        Flux.Optimise.update!(ADAM(η), Flux.params(model), grads)
        
        if(best_score > reward(en, sample, ag))
            n_replay += 1
            ag.state = best_state
            q_update!(en, ag, best_score)
            grads = Flux.gradient(Flux.params(model)) do
                loss(dq,ag, model)/length(ag.state)
            end
            Flux.Optimise.update!(ADAM(η), Flux.params(model), grads)
        else
            noup +=1
            best_state = ag.state
            best_score = reward(en, sample, ag)
        end
    end
    #println(findmax(ag.ex_table))
    println("replay: $(n_replay)")
    println("noup: $(noup)")
    println(best_state)
    println(model(zeros(Float32, dq.act_MAX)))
    println("==============================")
    println(size(ag.ex_table))
    println(ag.ex_table[1,1,:])
    println(ag.ex_table[6,4,:])
    println(ag.ex_table[30+2,1,:])
    loss_ave::Vector{Float64} = []
    for nn in 1:div(ll_MAX,100)
        if((100*(nn-1)+1000)<ll_MAX)
            l_av = sum(ll_it[(100*(nn-1)+1):(100*(nn-1)+1000)])/1000
            push!(loss_ave, l_av)
        end
    end
    p3 = plot(loss_ave.+1.0, xlabel="it_step", ylabel="loss",yscale=:log10, width=3.0)
    savefig(p3,"./loss_iterate.png")
    #m = copy(ag.model)

    @save "mymodel.bson" model

    sample = Sample(get_Sample(en)...)
    for i in 1:20
        r, pol = RandPolitics(en, dq, sample, ag, model)
        
        show_formula(pol)
        println(r)
        #println(pol)
    end
    ag.state=[3, 10, 20]
    r = reward(en, sample, ag)
    println(r)
end

@time main(ARGS)
