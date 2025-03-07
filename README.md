# Control of the Holos-Quad Nuclear Microreactor Through Reinforcement Learning

This repo is intended to document and reproduce the methods and results of its corresponding paper. It is also intended to be an easy starting point and example to build from for others to perform their own reinforcement learning research with this microreactor model.

## Package setup
This uses python 3.11. Necessary packages may be found in pyproject.toml. The easiest way to get things running is to install the [uv](https://astral.sh/uv/) package manager with: 
```curl -LsSf https://astral.sh/uv/install.sh | sh```

You will likely need to restart your terminal after installation.

## Regenerating results
With uv installed, run:
```uv run main.py```

Note that updating the training curve graphs to reflect new training requires manually downloading training logs, which is described below.

## Retraining models
The following pretrained models are included in the *runs* folder:
- single_action_rl (trained for 2 million timesteps)
- multi_action_rl (trained for 5 million timesteps)
- multi_action_rl_symmetric (trained for 5 million timesteps)
- marl (trained for 5 million timesteps)

To retrain a given model, delete the *models* folder within the corresponding named directory in the *runs* folder. By default, this will be trained for 2 million simulation timesteps (which corresponds to 16 million timesteps in the marl case, since there are eight separate agent actions per simulation timestep). To change this, add the '-t' flag followed by the number of timesteps you would like to train for. For example, to train for 5 million timesteps, use: 
```uv run main.py -t 5000000```

Note that in line with recommendations from [*stable-baselines3*](https://astral.sh/uv/),this is a CPU-only training, since the model doesn't involve a CNN.

## Tracking training progress
A TensorBoard server can be started with: 
```uv run tensorboard --logdir runs```

Opening the *localhost* URL in your browser will display various graphs showing training progress in real time.

Note that in order to update the training graphs generated in *main.py*, you will need to manually download and rename the csvs for episode length and reward through the browser from Tensorboard.
While this process is clunky, the python API to achieve the same results has been removed from the codebase after some time as an experimental feature.

## PID controller tuning
To retune the PID controller, run:
```uv run microutils.py```
