#!/bin/bash
#SBATCH -J strainkit
#SBATCH -p qM
#SBATCH -n 128
#SBATCH -c 1
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -t 24:00:00
source /etc/profile.d/modules.sh
module purge
module load inteloneapi/2023.2.0 intel-mkl/2023.2 intel-mpi/2021.10
export LD_LIBRARY_PATH=$HOME/etc/hdf5_intel_cl/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
set -o pipefail
RUN_DFT_CALCULATION
