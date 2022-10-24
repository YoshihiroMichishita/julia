#!/bin/sh
#PBS -l nodes=1:ppn=1
#PBS -q workq
#PBS -m e
#PBS -M q5s5n6r5e8g6x2p4@condensedmatt-fgt4462.slack.com
#PBS -N RNN_xi1.0_W10.0_batch3_d256_t100

#orig_dir=/home/michishita/Data/TwoSpin_WD_KpK_d256_W10.0_xi1.0_t100_batch3_1004
#root_dir=$PBS_O_WORKDIR/michishita/TwoSpin_WD_KpK_d256_W10.0_xi1.0_t100_batch3_1004 #work

#mkdir -p $root_dir

#cp ${orig_dir}/TwoSpin_agt.jl ${root_dir}/
#cd ${root_dir}

cd ~/Data/TwoSpin_WD_KpK_d256_W10.0_xi1.0_t100_batch3_1004

#If you input the parameters
t_size=100
W=10.0
xi=1.0
Jz=1.0
Jx=0.7
hz=0.5
n_dense=256
g=0.95
ep=5.0
it=4000

module load Julia/1.8.0
#if you use julia
julia ./TwoSpin_agt.jl ${t_size} ${W} ${xi} ${Jz} ${Jx} ${hz} ${n_dense} ${g} ${ep} ${it} init
#./../TwoSpin_g0.95_batch3_0828_res_d64_L3000_t100/mymodel.bson ./../TwoSpin_g0.95_batch3_0828_res_d64_L3000_t100/K_TL3000.csv

#Move the results(output files) to the Data directory
#mv ${root_dir}/* ${orig_dir}
