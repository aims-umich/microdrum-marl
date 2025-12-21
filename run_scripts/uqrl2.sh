#!/bin/bash
#SBATCH --job-name=uqrl2-0.8-1.1
#SBATCH --output=uqrl2-0.8-1.1%j
#SBATCH --error=uqrl2-0.8-1.1%j
#SBATCH --wckey=edu_res
#SBATCH --partition=general
#SBATCH --export=NONE
#SBATCH --time=6:00:00
###################
##### This submission script requests full nodes using all CPUs including hyperthreads
##### Change the number of requested --nodes as necessary:
#SBATCH --nodes=1
##### leave the parameters below unchanged:
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --exclusive
###################

cd ~/projects/microdrum-marl
uv run wiggle.py -u 0.8,0.9,1.0,1.1

