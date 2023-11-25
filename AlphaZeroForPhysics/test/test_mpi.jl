using LinearAlgebra
using MPI

struct Game
    state::Vector{Int}
    action::Int
    reward::Float32
    done::Bool
    child_pi::Vector{Vector{Float32}}
end

function init_game()
    return Game([], -1, 0.0, false, [])
end
function init_game(n::Int)
    st::Vector{Int} = []
    ch::Vector{Vector{Float32}} = []
    for it in 1:n
        push!(st, it)
        push!(ch, 0.1f0*st)
    end
    return Game(st, -1, 0.0, false, ch)
end

function main()
    println("Start!")
    MPI.Init()
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    root::Integer = 0
    MPI.Barrier(comm)
    
    mygame = init_game(myrank+1)
    MPI.Barrier(comm)

    if(myrank == root)
        games = MPI.gather(mygame, comm, root=root)
    else
        MPI.gather(mygame, comm, root=root)
    end
    MPI.Barrier(comm)
    
    if(myrank == root)
        println("$(games)")
    end
    MPI.Finalize()
    println("Finish!")
end

main()