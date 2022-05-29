using "coreEnv.jl"
#using .core

using Images

PATH_BLANK = "image/blank.png"
PATH_ROBOT = "image/robot_right.png"
PATH_CRYSTAL = "image/crystal_small.png"

struct CorridorEnv
    #c::coreEnv
    ID_blank::Int
    ID_robot::Int
    ID_crystal::Int
    field_length::Int
    crystal_candidate::Tuple{Int, Int}
    rwd_fail::Float64
    rwd_move::Float64
    rwd_crystal::Float64
    #robot_pos
    #crystal_pos
    #robot_state
    img_robot::Array{RGB{N0f8},2}
    img_crystal::Array{RGB{N0f8},2}
    img_blank::Array{RGB{N0f8},2}
    unit
end

mutable struct stat
    c::coreEnv
    robot_pos::Int
    crystal_pos::Int
    robot_state::String
end

function init_corridor_env()
    ID_blank::Int = 0
    ID_robot::Int = 1
    ID_crystal::Int =2
    field_length::Int = 4
    crystal_candidate::Tuple{Int, Int} = (2,3)
    rwd_fail::Float64 = -1.0
    rwd_move::Float64 = -1.0
    rwd_crystal::Float64 = 5.0
    #robot_pos = nothing
    #crystal_pos = nothing
    #robot_state = nothing
    img_robot = load(PATH_ROBOT)
    img_crystal = load(PATH_CRYSTAL)
    img_blank = load(PATH_BLANK)
    unit = size(img_robot)
    #c = coreEnv(coreEnv_init()...)

    #return c, ID_blank, ID_robot, ID_crystal, field_length, crystal_candidate, rwd_fail, rwd_move, rwd_crystal, robot_pos, crystal_pos, robot_state, img_robot, img_crystal, img_blank, unit
    return ID_blank, ID_robot, ID_crystal, field_length, crystal_candidate, rwd_fail, rwd_move, rwd_crystal, img_robot, img_crystal, img_blank, unit
end

function init_stat()
    c = coreEnv(coreEnv_init()...)
    r_pos::Int = nothing
    c_pos::Int = nothing
    r_stat::String = nothing
    return c, r_pos, c_pos, r_stat
end
    

function init_corridor_env(length::Int, cand::Tuple{Int, Int}, fail::Float64, move::Float64, cry::Float64)
    ID_blank::Int = 0
    ID_robot::Int = 1
    ID_crystal::Int =2
    field_length::Int = length
    crystal_candidate::Tuple{Int, Int} = cand
    rwd_fail::Float64 = fail
    rwd_move::Float64 = move
    rwd_crystal::Float64 = cry
    #robot_pos = nothing
    #crystal_pos = nothing
    #robot_state = nothing
    img_robot = load(PATH_ROBOT)
    img_crystal = load(PATH_CRYSTAL)
    img_blank = load(PATH_BLANK)
    unit = size(img_robot)
    #c = coreEnv(coreEnv_init()...)

    #return c, ID_blank, ID_robot, ID_crystal, field_length, crystal_candidate, rwd_fail, rwd_move, rwd_crystal, robot_pos, crystal_pos, robot_state, img_robot, img_crystal, img_blank, unit
    return ID_blank, ID_robot, ID_crystal, field_length, crystal_candidate, rwd_fail, rwd_move, rwd_crystal, img_robot, img_crystal, img_blank, unit
end

function make_obs(en::CorridorEnv ,st::stat)
    if(st.c.done)
        obs = ones(Int,en.field_length) * 9
        return obs
    end

    obs = ones(Int,en.field_length) * en.ID_blank
    obs[st.robot_pos] = en.ID_robot
    obs[st.crystal_pos] = en.ID_crystal

    return obs
end

function reset(en::CorridorEnv ,st::stat)
    st.c.done = false
    st.robot_state = "normal"
    st.robot_pos = 1

    st.crystal_pos = rand{Int}(en.crystal_candidate[1]:en.crystal_candidate[2])
    obs = make_obs(en, st)
    
    return obs
end

function step(en::CorridorEnv ,st::stat, act)

    if(st.c.done)
        obs = reset(en,st)
        return nothing, nothing, obs
    end

    if(act==0)
        if(st.robot_pos==st.crystal_pos)
            rwd = en.rwd_crystal
            done = true
            st.robot_state = "success"
        else
            rwd = en.rwd_fail
            done = true
            st.robot_state = "fail"
        end
    else
        next_pos = st.robot_pos + 1
        if(next_pos>en.field_length)
            rwd = en.rwd_fail
            done = true
            st.robot_state = "fail"
        else
            st.robot_pos = next_pos
            rwd = en.rwd_move
            done = false
            st.robot_state = "normal"
        end
    end

    st.c.done = done
    obs = make_obs(en,st)

    return rwd, done, obs
end

function render(en::CorridorEnv)
    width = en.unit * en.field_length
    height = en.unit

    img = zeros{RGB{N0f8},(3, width, height)}
    for i in 1:en.field_length
        
    end
end



