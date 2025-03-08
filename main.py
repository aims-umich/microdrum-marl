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
    # create interpolated power profiles
    training_profile = interp1d([  0,  15, 30, 70, 100, 140, 160, 195, 200], # times (s)
                                [100, 100, 80, 55,  55,  70,  70,  80,  80]) # power (SPU)
    testing_profile = interp1d([  0,  10, 70, 100, 115, 125, 150, 180, 200], # times (s)
                               [100, 100, 50,  50,   65,  65,  50,  80,  80]) # power (SPU)
    lowpower_profile = interp1d([  0,   5, 100, 200], # times (s)
                                [100, 100,  30,  90]) # power (SPU)
    longtest_profile = interp1d([  0,  2000, 3000, 3500, 6000, 10000, 10020, 12500, 14000, 16000, 16010, 20000], # times (s)
                                [100,   100,   90,   90,   45,    45,    65,    65,    80,    80,  95,  95]) # power (SPU)

    match args.test_profile:
        case 'longtest':
            test_profile = longtest_profile
            episode_length = 20000
        case 'lowpower':
            test_profile = lowpower_profile
            episode_length = 200
        case 'train':
            test_profile = training_profile
            episode_length = 200
        case _:
            test_profile = testing_profile
            episode_length = 200

    training_kwargs = {'profile': training_profile,
                      'episode_length': 200,
                      'train_mode': True}
    testing_kwargs = {'profile': test_profile,
                     'episode_length': episode_length,
                     'valid_maskings': (args.disabled_drums,),
                     'train_mode': False}


    #############################
    # Single Action RL Training #
    #############################
    single_folder = Path.cwd() / 'runs' / 'single-rl'
    single_folder.mkdir(exist_ok=True, parents=True)
    model_folder = single_folder / 'models/'
    if not model_folder.exists():  # if a model has already been trained, don't re-train
        print('Training Single Action RL...')
        training_kwargs['run_path'] = single_folder
        microutils.train_rl(envs.HolosSingle, training_kwargs,
                            total_timesteps=args.timesteps, n_envs=args.n_envs)

    #########################################
    # Single Action Innoculated RL Training #
    #########################################
    noise_folder = Path.cwd() / 'runs' / 'noise-rl'
    noise_folder.mkdir(exist_ok=True, parents=True)
    model_folder = noise_folder / 'models/'
    if not model_folder.exists():  # if a model has already been trained, don't re-train
        print('Training Single Action Innoculated RL...')
        training_kwargs['run_path'] = noise_folder
        training_kwargs['noise'] = 0.01  # 1 SPU standard deviation of measurement noise
        microutils.train_rl(envs.HolosSingle, training_kwargs,
                            total_timesteps=args.timesteps, n_envs=args.n_envs)

    noise01_folder = Path.cwd() / 'runs' / 'noise-rl01'
    noise01_folder.mkdir(exist_ok=True, parents=True)
    model_folder = noise01_folder / 'models/'
    if not model_folder.exists():  # if a model has already been trained, don't re-train
        print('Training Single Action Innoculated RL...')
        training_kwargs['run_path'] = noise01_folder
        training_kwargs['noise'] = 0.001  # .1 SPU standard deviation of measurement noise
        microutils.train_rl(envs.HolosSingle, training_kwargs,
                            total_timesteps=args.timesteps, n_envs=args.n_envs)

    noise2_folder = Path.cwd() / 'runs' / 'noise-rl2'
    noise2_folder.mkdir(exist_ok=True, parents=True)
    model_folder = noise2_folder / 'models/'
    if not model_folder.exists():  # if a model has already been trained, don't re-train
        print('Training Single Action Innoculated RL...')
        training_kwargs['run_path'] = noise2_folder
        training_kwargs['noise'] = 0.02  # 2 SPU standard deviation of measurement noise
        microutils.train_rl(envs.HolosSingle, training_kwargs,
                            total_timesteps=args.timesteps, n_envs=args.n_envs)

    ##########################
    # Multi Drum RL Training #
    ##########################
    multi_folder = Path.cwd() / 'runs' / 'multi-rl'
    multi_folder.mkdir(exist_ok=True, parents=True)
    model_folder = multi_folder / 'models/'
    if not model_folder.exists():  # if a model has already been trained, don't re-train
        print('Training Multi Action RL...')
        training_kwargs['run_path'] = multi_folder
        # training_kwargs['valid_maskings'] = (0,1,2,3)  # disable up to three drums at random
        microutils.train_rl(envs.HolosMulti, training_kwargs,
                            total_timesteps=args.timesteps, n_envs=args.n_envs)

    ######################################
    # Multi Drum RL (symmetric) Training #
    ######################################
    symmetric_folder = Path.cwd() / 'runs' / 'symmetric-rl'
    symmetric_folder.mkdir(exist_ok=True, parents=True)
    model_folder = symmetric_folder / 'models/'
    if not model_folder.exists():  # if a model has already been trained, don't re-train
        print('Training Multi Action RL (Symmetric)...')
        training_kwargs['run_path'] = symmetric_folder
        # training_kwargs['valid_maskings'] = (0,1,2,3)  # disable up to three drums at random
        microutils.train_rl(envs.HolosMulti,
                            {**training_kwargs,
                             'symmetry_reward': True},
                            total_timesteps=args.timesteps, n_envs=args.n_envs)

    #################
    # MARL Training #
    #################
    marl_folder = Path.cwd() / 'runs' / 'marl'
    marl_folder.mkdir(exist_ok=True, parents=True)
    model_folder = marl_folder / 'models/'
    if not model_folder.exists():
        print('Training Multi Action RL (MARL)...')
        training_kwargs['run_path'] = marl_folder
        # training_kwargs['valid_maskings'] = (0,1,2,3)  # disable up to three drums at random
        microutils.train_marl(envs.HolosMARL, training_kwargs,
                              total_timesteps=(args.timesteps * 8), n_envs=args.n_envs)


    ####################
    # Plotting Figures #
    ####################
    graph_path = Path.cwd() / 'graphs'
    graph_path.mkdir(exist_ok=True, parents=True)

    # start with the PID benchmark, creating its own run folder
    pid_folder = Path.cwd() / 'runs' / 'pid'
    pid_folder.mkdir(exist_ok=True, parents=True)
    training_kwargs['run_path'] = pid_folder
    pid_train_history = microutils.test_pid(envs.HolosSingle, training_kwargs)

    # Example profiles to validate environment and show train profile with PID
    plot_path = graph_path / '1a_PID-train-power.png'
    data_list = [(pid_train_history, 'desired_power', 'desired power'),
                 (pid_train_history, 'actual_power', 'actual power')]
    microutils.plot_history(plot_path, data_list, 'Power (SPU)')

    plot_path = graph_path / '1b_PID-train-temp.png'
    data_list = [(pid_train_history, 'Tf', 'fuel temp'),
                 (pid_train_history, 'Tm', 'moderator temp'),
                 (pid_train_history, 'Tc', 'coolant temp')]
    microutils.plot_history(plot_path, data_list, 'Temperature (K)')

    # TODO redo from scratch
    plot_path = graph_path / '1c_PID-train-diff.png'
    data_list = [(pid_train_history, 'diff', 'power difference')]
    microutils.plot_history(plot_path, data_list, 'Power Difference (SPU)')

    # TODO redo from scratch
    plot_path = graph_path / '1d_PID-train-angle.png'
    data_list = [(pid_train_history, 'drum_1', 'all drums')]
    microutils.plot_history(plot_path, data_list, 'Control Drum Position (°)')


    # gather test histories
    #######################
    print(f'testing with {args.test_profile}:')
    pid_test_history = microutils.test_pid(envs.HolosSingle, {**testing_kwargs,
                                                              'run_path': pid_folder})
    single_test_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
                                                                        'run_path': single_folder,})
    multi_test_history = microutils.test_trained_rl(envs.HolosMulti, {**testing_kwargs,
                                                                      'run_path': multi_folder})
    symmetric_test_history = microutils.test_trained_rl(envs.HolosMulti,
                                                        {**testing_kwargs,
                                                        'run_path': symmetric_folder,
                                                        'symmetry_reward': True,
                                                        'train_mode': True})  # necessary to cutoff runaway power
    marl_test_history = microutils.test_trained_marl(envs.HolosMARL, {**testing_kwargs,
                                                                      'run_path': marl_folder})
    single_marl_test_history = microutils.test_trained_marl(envs.HolosMARL, {**testing_kwargs,
                                                                            'run_path': single_folder})
                                                
    # plot comparisons pid vs single agent
    ######################################
    plot_path = graph_path / f'2_{args.test_profile}-power.png'
    data_list = [(pid_test_history, 'desired_power', 'desired power'),
                 (pid_test_history, 'actual_power', 'pid'),
                 (single_test_history, 'actual_power', 'rl')]
    microutils.plot_history(plot_path, data_list, 'Power (SPU)')

    plot_path = graph_path / f'2_{args.test_profile}-diff.png'
    data_list = [(pid_test_history, 'diff', 'pid'),
                 (single_test_history, 'diff', 'rl')]
    microutils.plot_history(plot_path, data_list, 'Power Difference (SPU)')

    # plot comparisons multi-action vs symmetric vs marl
    ####################################################
    plot_path = graph_path / f'3_{args.test_profile}-power.png'
    data_list = [(pid_test_history, 'desired_power', 'desired power'),
                 (multi_test_history, 'actual_power', 'multi-action'),
                 (symmetric_test_history, 'actual_power', 'symmetric'),
                 (marl_test_history, 'actual_power', 'marl'),
                 (single_marl_test_history, 'actual_power', 'singly trained marl')]
    microutils.plot_history(plot_path, data_list, 'Power (SPU)')

    plot_path = graph_path / f'3_{args.test_profile}-diff.png'
    data_list = [(multi_test_history, 'diff', 'multi-action'),
                 (symmetric_test_history, 'diff', 'symmetric'),
                 (marl_test_history, 'diff', 'marl'),
                 (single_marl_test_history, 'diff', 'singly trained marl')]
    microutils.plot_history(plot_path, data_list, 'Power Difference (SPU)')

    plot_path = graph_path / f'3_{args.test_profile}-multi-angle.png'
    data_list = [(multi_test_history, 'drum_1', 'drum 1'),
                 (multi_test_history, 'drum_2', 'drum 2'),
                 (multi_test_history, 'drum_3', 'drum 3'),
                 (multi_test_history, 'drum_4', 'drum 4'),
                 (multi_test_history, 'drum_5', 'drum 5'),
                 (multi_test_history, 'drum_6', 'drum 6'),
                 (multi_test_history, 'drum_7', 'drum 7'),
                 (multi_test_history, 'drum_8', 'drum 8'),]
    microutils.plot_history(plot_path, data_list, 'Control Drum Position (°)')

    plot_path = graph_path / f'3_{args.test_profile}-marl-angle.png'
    data_list = [(marl_test_history, 'drum_1', 'drum 1'),
                 (marl_test_history, 'drum_2', 'drum 2'),
                 (marl_test_history, 'drum_3', 'drum 3'),
                 (marl_test_history, 'drum_4', 'drum 4'),
                 (marl_test_history, 'drum_5', 'drum 5'),
                 (marl_test_history, 'drum_6', 'drum 6'),
                 (marl_test_history, 'drum_7', 'drum 7'),
                 (marl_test_history, 'drum_8', 'drum 8'),]
    microutils.plot_history(plot_path, data_list, 'Control Drum Position (°)')

    plot_path = graph_path / f'3_{args.test_profile}-single-marl-angle.png'
    data_list = [(single_marl_test_history, 'drum_1', 'drum 1'),
                 (single_marl_test_history, 'drum_2', 'drum 2'),
                 (single_marl_test_history, 'drum_3', 'drum 3'),
                 (single_marl_test_history, 'drum_4', 'drum 4'),
                 (single_marl_test_history, 'drum_5', 'drum 5'),
                 (single_marl_test_history, 'drum_6', 'drum 6'),
                 (single_marl_test_history, 'drum_7', 'drum 7'),
                 (single_marl_test_history, 'drum_8', 'drum 8'),]
    microutils.plot_history(plot_path, data_list, 'Control Drum Position (°)')

    # plot training curves multi-action vs symmetric vs marl
    ########################################################
    multi_logs_path = multi_folder / 'logs/PPO_1'
    symmetric_logs_path = symmetric_folder / 'logs/PPO_1'
    marl_logs_path = marl_folder / 'logs/PPO_1'

    multi_ep_len = pd.read_csv(multi_logs_path / 'ep_len_mean.csv')
    symmetric_ep_len = pd.read_csv(symmetric_logs_path / 'ep_len_mean.csv')
    marl_ep_len = pd.read_csv(marl_logs_path / 'ep_len_mean.csv')
    marl_ep_len['Step'] = marl_ep_len['Step'] / 8  # normalize by point kinetics simulations run
    plt.clf()
    plt.plot(multi_ep_len['Step'], multi_ep_len['Value'], label='multi-action')
    plt.plot(symmetric_ep_len['Step'], symmetric_ep_len['Value'], label='symmetric')
    plt.plot(marl_ep_len['Step'], marl_ep_len['Value'], label='marl')
    plt.xlabel('Environment timesteps')
    plt.ylabel('Episode length (s)')
    plt.legend()
    plt.savefig(graph_path / f'4_training-curve-ep-len.png')

    multi_ep_rew = pd.read_csv(multi_logs_path / 'ep_rew_mean.csv')
    symmetric_ep_rew = pd.read_csv(symmetric_logs_path / 'ep_rew_mean.csv')
    marl_ep_rew = pd.read_csv(marl_logs_path / 'ep_rew_mean.csv')
    marl_ep_rew['Step'] = marl_ep_rew['Step'] / 8  # normalize by point kinetics simulations run
    plt.clf()
    plt.plot(multi_ep_rew['Step'], multi_ep_rew['Value'], label='multi-action')
    plt.plot(symmetric_ep_rew['Step'], symmetric_ep_rew['Value'], label='symmetric')
    plt.plot(marl_ep_rew['Step'], marl_ep_rew['Value'], label='marl')
    plt.xlabel('Environment timesteps')
    plt.ylabel('Episode reward')
    plt.legend()
    plt.savefig(graph_path / f'4_training-curve-ep-rew.png')

    # plot noise graphs
    ####################
    single_1noise_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
                                                                          'run_path': single_folder,
                                                                          'noise': 0.01})
    single_2noise_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
                                                                          'run_path': single_folder,
                                                                          'noise': 0.02})
    single_3noise_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
                                                                          'run_path': single_folder,
                                                                          'noise': 0.03})
    # innoculated2_2noise_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
    #                                                                             'run_path': innoculated2_folder,
    #                                                                             'noise': 0.03})
    noise_path = graph_path / '5_noise-power-2SPU.png'
    data_list = [(single_1noise_history, 'desired_power', 'desired power'),
                 (single_1noise_history, 'actual_power', '1 SPU noise'),
                 (single_2noise_history, 'actual_power', '2 SPU noise'),
                 (single_3noise_history, 'actual_power', '3 SPU noise')]
    microutils.plot_history(noise_path, data_list, 'Power (SPU)')

    noise_path = graph_path / '5_noise-diff-2SPU.png'   
    data_list = [(single_1noise_history, 'diff', '1 SPU noise'),
                 (single_2noise_history, 'diff', '2 SPU noise'),
                 (single_3noise_history, 'diff', '3 SPU noise')]
    microutils.plot_history(noise_path, data_list, 'Power Difference (SPU)')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--plotting', action='store_true',  # TODO remove, unused
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
