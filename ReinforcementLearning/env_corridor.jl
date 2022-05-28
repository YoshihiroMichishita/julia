using "coreEnv.jl"
#using .core

using Images

PATH_BLANK = "image/blank.png"
PATH_ROBOT = "image/robot_right.png"
PATH_CRYSTAL = "image/crystal_small.png"

struct CorridorEnv
    c::coreEnv
    ID_blank::Int
    ID_robot::Int
    ID_crystal::Int
    field_length::Int
    crystal_candidate::Tuple{Int, Int}
    rwd_fail::Float64
    rwd_move::Float64
    rwd_crystal::Float64
    robot_pos
    crystal_pos
    robot_state
    img_robot::Array{RGB{N0f8},2}
    img_crystal::Array{RGB{N0f8},2}
    img_blank::Array{RGB{N0f8},2}
    unit
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
    robot_pos = nothing
    crystal_pos = nothing
    robot_state = nothing
    img_robot = load(PATH_ROBOT)
    img_crystal = load(PATH_CRYSTAL)
    img_blank = load(PATH_BLANK)
    unit = size(img_robot)

    return coreEnv_init(), ID_blank, ID_robot, ID_crystal, field_length, crystal_candidate, rwd_fail, rwd_move, rwd_crystal, robot_pos, crystal_pos, robot_state, img_robot, img_crystal, img_blank, unit
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
    robot_pos = nothing
    crystal_pos = nothing
    robot_state = nothing
    img_robot = load(PATH_ROBOT)
    img_crystal = load(PATH_CRYSTAL)
    img_blank = load(PATH_BLANK)
    unit = size(img_robot)

    return coreEnv_init(), ID_blank, ID_robot, ID_crystal, field_length, crystal_candidate, rwd_fail, rwd_move, rwd_crystal, robot_pos, crystal_pos, robot_state, img_robot, img_crystal, img_blank, unit
end



