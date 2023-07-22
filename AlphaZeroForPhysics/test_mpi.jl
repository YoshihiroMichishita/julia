using LinearAlgebra
using MPI

function main()
    println("Start!")
    MPI.Init()
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    root = 0

    if myrank == root
        all_game = []
        ag_buf = MPI.Buffer(all_game)
        game_vec = []
        game_buf = MPI.Buffer(game_vec)
        println("Buffer Set!")
    else
        ag_buf = MPI.UBuffer(nothing)
        game_buf = MPI.UBuffer(nothing)
    end
    MPI.Barrier(comm)
    local_game = MPI.Scatterv!(game_buf, root, comm)
    
    for it in 1:myrank+1
        push!(local_game, it)
    end
    MPI.Barrier(comm)
    MPI.Gatherv!(local_game, output_vbuf, root, comm)
    MPI.Barrier(comm)
    
    if(myrank == root)
        println("$(all_game)")
    end
    MPI.Finalize()
    println("Finish!")
end

main()