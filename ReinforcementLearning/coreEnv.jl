using LinearAlgebra
#module core
mutable struct coreEnv
    n_act::Int
    done::Bool
end

function coreEnv_init()
    return 2, false
end

function reset(c::coreEnv)
    c.done = false
    obs = zeros(Int,4)
    return obs
end

function step(c::coreEnv, act)
    if(c.done)
        obs = reset()
        return nothing, nothing, obs
    end

    rwd::Float64 = 1.0
    done::Bool = true

    c.doce = done
    obs = zeros(Int,4)

    return rwd, done, obs
end

function render()
    img = zeros(UInt8, 100, 200, 3)
    return img
end

mutable struct coreAgt
    epsilon::Float64
end

coreAgt_init() = 0.4

function select_action(obs::Array{Int,1})
    act::Int = 0
    return act
end

function learn(obs::Array{Int,1}, act::Int, rwd::Float64, done::Bool, next_bos)
    if(isnothing(rwd))
        return
    end
end

function get_Q(obs::Array{Int,1})
    Q = zeros(Float64,2)
    return Q
end

#end