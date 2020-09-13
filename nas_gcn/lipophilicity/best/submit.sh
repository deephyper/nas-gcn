#!/bin/sh
​
#SBATCH --job-name=regevo30
#SBATCH --account=PERFOPT
#SBATCH --partition=bdw
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=1
#SBATCH --output=regevo30.out
#SBATCH --error=regevo30.error
#SBATCH --time=03:00:00
​
​
# export MPICH_GNI_FORK_MODE=FULLCOPY # otherwise, fork() causes segfaults above 1024 nodes
export PMI_NO_FORK=1 # otherwise, mpi4py-enabled Python apps with custom signal handlers do not respond to sigterm
export KMP_AFFINITY=disabled # this can affect on-node scaling (test this)
​
# Required for Click_ to work: https://click.palletsprojects.com/en/7.x/python3/
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8
​
# Activate good python environment
source /blues/gpfs/home/[your_directory]/dh_env/bin/activate
​
echo $PATH
​
export PATH=/blues/gpfs/home/[your_directory]/dh-env/bin:/usr/local/bin:$PATH
​
# deactivate core dump (comment for debug)
ulimit -c 0
​
# Start cluster wit 30 nodes with regularized evolution
srun -n 30 python -m tuster.system.bebop.run 'python -m deephyper.search.nas.regevo --evaluator ray --redis-address {redis_address} --problem nas_gcn.lipophilicity.problem.Problem'

# Start cluster wit 30 nodes with random search
# srun -n 30 python -m tuster.system.bebop.run 'python -m deephyper.search.nas.full_random --evaluator ray --redis-address {redis_address} --problem nas_gcn.lipophilicity.problem.Problem'