import stable_baselines3 as sb3
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib
import matplotlib.pyplot as plt
import envs
import microutils
from scipy.interpolate import interp1d


def main(args):
    # create interpolated power profiles to match the Holos benchmark
    args.plotting
    args.timesteps
    
    training_profile = interp1d([  0,  15, 30, 70, 100, 140, 160, 195, 200], # times (s)
                                [100, 100, 80, 55,  55,  70,  70,  80,  80]) # power (SPU)
    lowpower_profile = interp1d([  0,   5, 100, 200], # times (s)
                                [100, 100,  40,  90]) # power (SPU)
    longtest_profile = interp1d([  0,  2000, 3000, 3500, 6000, 10000, 12000, 12500, 14000, 16000, 18000, 20000], # times (s)
                                [100,   100,   90,   90,   45,    45,    65,    65,    80,  80,  95,  95]) # power (SPU)
    testing_profile = interp1d([  0,  10, 70, 100, 115, 125, 150, 180, 200], # times (s)
                               [100, 100, 45, 45,   65,  65,  50,  80,  80]) # power (SPU)

    match args.test_profile:
        case 'longtest':
            test_profile = longtest_profile
        case 'lowpower':
            test_profile = lowpower_profile
        case 'training':
            test_profile = training_profile
        case _:
            test_profile = testing_profile

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
    testing_kwargs = {'profile': test_profile,
                     'episode_length': 200,
                     'run_path': run_folder,
                     'valid_maskings': (args.disabled_drums,),
                     'train_mode': False}

    # run the PID loop
    env = envs.HolosSingle(**testing_kwargs)
    microutils.pid_loop(env)
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    pid_test_history = microutils.load_history(history_path)
    mae, iae, control_effort = microutils.calc_metrics(pid_test_history)
    print(f'PID test - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')
    if args.plotting:
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
        microutils.train_rl(envs.HolosSingle, training_kwargs,
                            total_timesteps=args.timesteps, n_envs=args.n_envs)
    # test trained model
    single_action_test_history = microutils.test_trained_rl(envs.HolosSingle, testing_kwargs)
    if args.plotting:
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
        microutils.train_rl(envs.HolosMulti, training_kwargs,
                            total_timesteps=args.timesteps, n_envs=args.n_envs)
    multi_drum_test_history = microutils.test_trained_rl(envs.HolosMulti, testing_kwargs)
    if args.plotting:
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
                            total_timesteps=args.timesteps, n_envs=args.n_envs)
    multi_symmetric_test_history = microutils.test_trained_rl(envs.HolosMulti,
                                                              {**testing_kwargs,
                                                              'symmetry_reward': True})
    if args.plotting:
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
    model_folder = run_folder / 'models/'
    # if a model has already been trained, don't re-train
    if not model_folder.exists():
        microutils.train_marl(envs.HolosMARL, training_kwargs,
                              total_timesteps=(args.timesteps * 8), n_envs=args.n_envs)
    marl_test_history = microutils.test_trained_marl(envs.HolosMARL, testing_kwargs)
    if args.plotting:
        microutils.plot_history(marl_test_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--plotting', action='store_true',
                        help='Plot interactive, intermediate results')
    parser.add_argument('-p', '--test_profile', type=str, default='test',
                        help='Profile to use for testing (test, train, longtest, lowpower)')
    parser.add_argument('-t', '--timesteps', type=int, default=2_000_000,
                        help='Number of timesteps to train for')
    parser.add_argument('-d', '--disabled_drums', type=int, default=0,
                        help='Number of drums to disable during testing')
    parser.add_argument('-n', '--n_envs', type=int, default=10,
                        help='Number of environments to use for training')
    args = parser.parse_args()
    main(args)
