using LinearAlgebra
using MPI

function main()
    println("Start!")
    MPI.Init()
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    root = 0

    if myrank == root
        all_game = Int[]
        ag_buf = MPI.Buffer(all_game)
        game_vec = Int[]
        game_buf = MPI.Buffer(game_vec)
        println("Buffer Set!")
    else
        ag_buf = MPI.Buffer(nothing)
        game_buf = MPI.Buffer(nothing)
    end
    MPI.Barrier(comm)
    println("setting up!")
    local_game = MPI.Scatterv!(game_buf, Int[], root, comm)
    println("scatter!")
    for it in 1:myrank+1
        push!(local_game, it)
    end
    MPI.Barrier(comm)
    output_vbuf = MPI.Buffer([])
    MPI.Gatherv!(local_game, output_vbuf, [length(game_vec)], [length(game_vec)], root, comm)
    MPI.Barrier(comm)
    
    if myrank == root
        all_game = MPI.Buffer(output_vbuf)
        println("$(all_game)")
    end
    MPI.Finalize()
    println("Finish!")
end

main()

