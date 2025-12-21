#!/bin/bash
#SBATCH --job-name=uqrl-0-0.3
#SBATCH --output=uqrl-0-0.5%j
#SBATCH --error=uqrl-0-0.5%j
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
uv run wiggle.py -u 0.0,0.1,0.2,0.3

