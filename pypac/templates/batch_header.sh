{% if machine == 'debug' %}

{% elif machine == 'chinook' %}

# Running on chinook

#SBATCH  --job-name={{ prefix }}
#SBATCH  --ntasks={{ cores }}
#SBATCH  --tasks-per-node={{ ppn}}
#SBATCH  --output={{ scriptDir }}/{{ prefix }}.o%j
#SBATCH  --time={{ walltime }}
#SBATCH  --partition={{ partition }}

module list

umask 007

cd $SLURM_SUBMIT_DIR

# Generate a list of compute node hostnames reserved for this job,
# this ./nodes file is necessary for slurm to spawn mpi processes
# across multiple compute nodes
srun -l /bin/hostname | sort -n | awk '{{print $2}}' > ./nodes_$SLURM_JOBID

ulimit -l unlimited
ulimit -s unlimited
ulimit

{% elif machine == 'stampede2' %}

# Running on stampede2

#SBATCH  --job-name={{ prefix }}
#SBATCH  --account={{ account }}
#SBATCH  --nodes={{ nodes }}
#SBATCH  --output={{ scriptDir }}/{{ prefix }}.o%j
#SBATCH  --exclusive
#SBATCH  --time={{ walltime }}
#SBATCH  --partition={{ partition }}

{% endif %}
