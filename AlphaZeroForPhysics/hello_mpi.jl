using MPI

function main()
    # initialize
    MPI.Init()

    # establish the MPI communicator and obtain rank
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # wait not to be mixed
    sleep(rank*0.1)
    println("Hello! I am rank $(rank).")

    # finalize
    MPI.Finalize()
end

main()
