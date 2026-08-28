#!/bin/sh
#PBS -l nodes=1:ppn=16
#PBS -q default
# Job-script template: the line RUN_DFT_CALCULATION is replaced by a loop over
# the calculation directories which runs the lines of DFT_command.sh in each.
NUMPRO=16
cd $PBS_O_WORKDIR

RUN_DFT_CALCULATION
