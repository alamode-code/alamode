srun ~/src/vasp/vasp.6.5.1/bin/vasp_std > vasp.out || echo "FAILED $PWD" >> $SLURM_SUBMIT_DIR/failed_runs.txt
