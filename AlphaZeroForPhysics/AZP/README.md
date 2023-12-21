# Alpha Zero For Physics

The codes with "_CPU" work without GPU(CUDA) and can run even in Apple Silicon PC. (I have checked in my M1 Macbook Air.)

The codes with "_single" are not parallelized in multi-process, while they finishes their calculation in about 20min per single AZfP trial (in my M1 Macbook Air).

The codes with "_woNN" use small memory and do not use the neural networks. For simple problems, this treatment is often enough and its calculation is very fast.

In ``AZP_env.jl``, we define the environment for the reinforcement learning problems, those are the definition of the reward(score) function and the rules of games. You can apply Alpha Zero for Physics by rewritting this code for the problems you want to solve.

In ``AZP.agt.jl``, we define how to make the batches for learning and the memory for recent experienced scores.

In ``AZP_mcts~.jl``, we write the core part of Alpha Zero For Physics, which are P+UCT serch.

``AZP_DNN~.jl`` is the main code to run in terminal. 


If you try ``julia AZP_DNN_single.jl 14 100 64 12 2000 500 1 512 0.3 0.25 100 4 10.0 0.4 1.0 0.7 0.5 100 1.25 0.000001 0.1 1.0``, you can get the results similar to the figure.4 in my paper.
