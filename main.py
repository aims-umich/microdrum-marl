import stable_baselines3 as sb3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import envs
import microutils
from scipy.interpolate import interp1d
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor


def main():
    # create interpolated power profiles to match the Holos benchmark
    plotting = False
    training_profile = interp1d([  0,  20, 30, 35, 60, 100, 120, 125, 140, 160, 180, 200], # times (s)
                                [100, 100, 90, 90, 55,  55,  65,  65,  80,  80,  95,  95]) # power (SPU)
    testing_profile = interp1d([  0,  10, 70, 100, 115, 125, 150, 180, 200], # times (s)
                               [100, 100, 45, 45,   65,  65,  50,  80,  80]) # power (SPU)

    ##################
    # PID Controller #
    ##################
    # start with the PID benchmark, creating a run folder
    run_folder = Path.cwd() / 'runs' / 'pid_train'
    run_folder.mkdir(exist_ok=True, parents=True)
    training_kwargs = {'profile': training_profile,
                      'episode_length': 200,
                      'run_path': run_folder,
                      'train_mode': True}
    testing_kwargs = {'profile': testing_profile,
                     'episode_length': 200,
                     'run_path': run_folder,
                     'train_mode': False}

    # run the PID loop
    env = envs.HolosSingle(**testing_kwargs)
    microutils.pid_loop(env)
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    pid_test_history = microutils.load_history(history_path)
    mae, iae, control_effort = microutils.calc_metrics(pid_test_history)
    print(f'PID test - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')
    if plotting:
        microutils.plot_history(pid_test_history)

    ####################
    # Single Action RL #
    ####################
    # create a run folder
    run_folder = Path.cwd() / 'runs' / 'single_action_rl'
    run_folder.mkdir(exist_ok=True, parents=True)
    training_kwargs['run_path'] = run_folder
    testing_kwargs['run_path'] = run_folder
    model_folder = run_folder / 'models/'
    # if a model has already been trained, don't re-train
    if not model_folder.exists():
        microutils.train_rl(envs.HolosSingle, training_kwargs)
    # test trained model
    single_action_test_history = microutils.test_trained_rl(envs.HolosSingle, testing_kwargs)
    if plotting:
        microutils.plot_history(single_action_test_history)

    #################
    # Multi Drum RL #
    #################
    # create a run folder
    run_folder = Path.cwd() / 'runs' / 'multi_action_rl'
    run_folder.mkdir(exist_ok=True, parents=True)
    training_kwargs['run_path'] = run_folder
    testing_kwargs['run_path'] = run_folder
    model_folder = run_folder / 'models/'
    # if a model has already been trained, don't re-train
    if not model_folder.exists():
        microutils.train_rl(envs.HolosMulti, training_kwargs)
    multi_drum_test_history = microutils.test_trained_rl(envs.HolosMulti, testing_kwargs)
    if plotting:
        microutils.plot_history(multi_drum_test_history)

    #############################
    # Multi Drum RL (symmetric) #
    #############################
    # create a run folder
    run_folder = Path.cwd() / 'runs' / 'multi_action_rl_symmetric'
    run_folder.mkdir(exist_ok=True, parents=True)
    training_kwargs['run_path'] = run_folder
    testing_kwargs['run_path'] = run_folder
    model_folder = run_folder / 'models/'
    # if a model has already been trained, don't re-train
    if not model_folder.exists():
        microutils.train_rl(envs.HolosMulti,
                            {**training_kwargs,
                             'symmetry_reward': True},
                            total_timesteps=5_000_000)
    multi_symmetric_test_history = microutils.test_trained_rl(envs.HolosMulti,
                                                              {**testing_kwargs,
                                                              'symmetry_reward': True})
    if plotting:
        microutils.plot_history(multi_symmetric_test_history)

    ########
    # MARL #
    ########
    # create a run folder
    run_folder = Path.cwd() / 'runs' / 'marl'
    run_folder.mkdir(exist_ok=True, parents=True)
    training_kwargs['run_path'] = run_folder
    training_kwargs['valid_maskings'] = (0,1,2,3)
    testing_kwargs['run_path'] = run_folder
    testing_kwargs['valid_maskings'] = (5,)
    model_folder = run_folder / 'models/'
    # if a model has already been trained, don't re-train
    if not model_folder.exists():
        microutils.train_marl(envs.HolosMARL, training_kwargs)
    marl_test_history = microutils.test_trained_marl(envs.HolosMARL, testing_kwargs)
    if plotting:
        microutils.plot_history(marl_test_history)


if __name__ == '__main__':
    main()
