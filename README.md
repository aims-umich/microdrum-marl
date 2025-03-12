# Control of the Holos-Quad Nuclear Microreactor Through Reinforcement Learning

This repo is intended to document and reproduce the methods and results of its corresponding paper. It is also intended to be an easy starting point and example to build from for others to perform their own reinforcement learning research with this microreactor model.

## Package setup
This uses python 3.11. Necessary packages may be found in pyproject.toml. The easiest way to get things running is to install the [uv](https://astral.sh/uv/) package manager with:

```curl -LsSf https://astral.sh/uv/install.sh | sh```

You will likely need to restart your terminal after installation.

## Regenerating results
First clone this repository:

```git clone https://github.com/lionturkey/microdrum-marl.git```  
```cd microdrum-marl```

With uv installed, you can run:

```uv run main.py```

Note that updating the training curve graphs to reflect new training requires manually downloading training logs, which is described below. Plots will be saved to a new folder named *graphs*.

## Project structure and retraining
The three key python files are:
- envs.py: Contains the reactor model and RL environments
- microutils.py: Contains utility functions for training, evaluation, tuning, and more
- main.py: Reproduces the results from the paper

To save training time, pretrained models and training outputs are included in the *runs* folder. To train these on your own, simply delete the folder you would like to retrain.
- single-rl: a single action agent trained for 2 million timesteps (estimated 1 hour)
- multi-rl: a multi action agent trained for 5 million timesteps (estimated 3 hours)
- symmetric-rl: a multi action agent trained for 5 million timesteps (estimated 3 hours)
- marl: a MARL agent trained for 40 million timesteps (5 million simulation timesteps, estimated 5 hours)

To retrain a given model, delete the *models* folder within the corresponding named directory in the *runs* folder. By default, this will be trained for 2 million simulation timesteps (which corresponds to 16 million timesteps in the marl case, since there are eight separate agent actions per simulation timestep). To change this, add the '-t' flag followed by the number of timesteps you would like to train for. For example, to train for 5 million timesteps, use:

```uv run main.py -t 5000000```

Note that in line with recommendations from [*stable-baselines3*](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), this is a CPU-only training, since the model doesn't involve a CNN.

## Testing models
By default, models are tested on the 200s "test" profile. To test on a different profile, add the '-p' flag followed by the name of the profile you would like to test on. You may disable a number of control drums at random with the '-d' flag. For example, to disable 1 drum on the 300 minute "longtest" profile, use:

```uv run main.py -p longtest -d 1```

Available profiles are "train", "test", "lowpower", and "longtest".

## Tracking training progress
A TensorBoard server can be started with:

```uv run tensorboard --logdir runs```

Opening the *localhost* URL in your browser will display various graphs showing training progress in real time.

Note that in order to update the training graphs generated in *main.py*, you will need to manually download and rename the csvs for episode length and reward through the browser from Tensorboard.
While this process is clunky, the python API to achieve the same results has been removed from the codebase after some time as an experimental feature.

## PID controller tuning
To retune the PID controller, run:

```uv run microutils.py```

## Repeating noise study
To rerun the noise study (estimated run time 12 minutes), delete the cached noise study files named "noise-metrics.csv" in the pid, single-rl, and marl folders.