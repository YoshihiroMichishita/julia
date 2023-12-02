# Alpha Zero For Physics
In this repository, I uploaded the codes using in [arXiv:2311.12713](https://arxiv.org/abs/2311.12713).

All source codes are wirtten in [Julia](https://julialang.org/) language and I have checked that they works well for the latest version(Julia-ver1.9.4).

The directory "AZP" includes the codes for the Alpha Zero for Physics and the derivation for the high-frequency expansion in periodically-driven systems.

The directory "PPO" includes the codes for the Actor-Critic + PPO methods and the derivation for the high-frequency expansion in periodically-driven systems.

The other directory includes codes which is not directly related to my paper.

e-greedy.jl can simulate the game of deriving the high-frequency expansion with $\epsilon$-greedy algorithm.

In the source code, we use the libraries of LinearAlgebras, Random, Distributions, StatsBase, [Flux](https://fluxml.ai/Flux.jl/stable/) (for using neural networks and its trining), [SymPy@1.2.1](https://github.com/JuliaPy/SymPy.jl) (for symbolic calculation), [Plots](https://docs.juliaplots.org/stable/) (for creating graphs), [CSV](https://csv.juliadata.org/stable/), [DataFrames](https://dataframes.juliadata.org/stable/), and [CUDA](https://cuda.juliagpu.org/stable/) (If you use GPU). Note that my code does not run with the latest version of SymPy-2.0.1.