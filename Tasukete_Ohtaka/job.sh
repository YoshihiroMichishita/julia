#!/bin/sh
#SBATCH -p F1cpu
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --ntasks-per-socket=24
#SBATCH -J julia_test_20230612
#SBATCH --mail-type=ALL
#SBATCH --mail-user=000000000.slack.com

orig_dir=/home/0000/0000/Data/julia_test/julia_test_20230612
root_dir=/work/0000/0000/julia_test/julia_test_20230612 #work

mkdir -p $root_dir

cp ${orig_dir}/test.jl ${root_dir}/
cd ${root_dir}

srun julia test.jl 

#Move the results(output files) to the Data directory
mv ${root_dir}/* ${orig_dir}