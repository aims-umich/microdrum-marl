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

plt.style.use('tableau-colorblind10')


def main(args):
    # create interpolated power profiles
    training_profile = interp1d([  0,  10, 20, 30, 50, 70, 120, 140, 160, 195, 200], # times (s)
                                [100, 100, 98, 99, 80, 60,  60,  70,  70,  80,  80]) # power (SPU)
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

        # marl_test_history = microutils.test_trained_marl(envs.HolosMARL, {**testing_kwargs,
        #                                                               'run_path': marl_folder})
    model_list = list(model_folder.glob('*.zip'))
    if len(model_list) > 1:
        print('Multiple MARL models found, running all to determine best model')
        low_cae = float('inf')
        for model in model_list:
            model.touch(exist_ok=True)
            history = microutils.test_trained_marl(envs.HolosMARL, {**training_kwargs,
                                                                  'run_path': marl_folder})
            _, cae, _, _ = microutils.calc_metrics(history)
            if cae < low_cae:
                print(f'New best model found: {model.name} - CAE: {cae}')
                low_cae = cae
                model.rename(model_folder / 'best_model.zip')
        # clean up poor models
        for model in model_list:
            if model.name != 'best_model.zip':
                model.unlink(missing_ok=True)


    ####################
    # Plotting Figures #
    ####################
    graph_path = Path.cwd() / 'graphs'
    graph_path.mkdir(exist_ok=True, parents=True)

    # gather test histories
    #######################
    print(f'testing with {args.test_profile}:')
    pid_folder = Path.cwd() / 'runs' / 'pid'
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
    # single_marl_test_history = microutils.test_trained_marl(envs.HolosMARL, {**testing_kwargs,
    #                                                                         'run_path': single_folder})

    # Graph 1: PID vs single-RL on test profile with temperature included
    # ###################################################################
    graph_1_path = graph_path / f'1-singletemp-{args.test_profile}.png'
    plt.clf()
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 12)) # power, error, temps, drum speed, drum position
    axs[0].plot(pid_test_history['time'], pid_test_history['desired_power'], label='Desired power', color='black', linestyle='-')
    axs[0].plot(pid_test_history['time'], pid_test_history['actual_power'], label='PID power', linestyle=':')
    axs[0].plot(single_test_history['time'], single_test_history['actual_power'], label='Single-RL power', linestyle='--')
    axs[0].legend()
    axs[0].set_ylabel('Power (SPU)')
    axs[1].plot(pid_test_history['time'], pid_test_history['diff'], label='PID error', linestyle=':')
    axs[1].plot(single_test_history['time'], single_test_history['diff'], label='Single-RL error', linestyle='-')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].legend()
    axs[1].set_ylabel('Error (SPU)')
    axs[2].plot(pid_test_history['time'], pid_test_history['Tf'], label='Fuel temperature')
    axs[2].plot(pid_test_history['time'], pid_test_history['Tm'], label='Moderator temperature')
    axs[2].plot(pid_test_history['time'], pid_test_history['Tc'], label='Coolant temperature')
    axs[2].legend()
    axs[2].set_ylabel('Temperature (deg C)')
    axs[3].plot(pid_test_history['time'], np.insert(np.diff(pid_test_history['drum_1']), 0, 0), label='PID drum speed', linestyle=':')
    axs[3].plot(single_test_history['time'], np.insert(np.diff(single_test_history['drum_1']), 0, 0), label='Single-RL drum speed', linestyle='-')
    axs[3].legend()
    axs[3].set_ylabel('Drum speed (degrees per second)')
    axs[4].plot(pid_test_history['time'], pid_test_history['drum_1'], label='PID drum position', linestyle=':')
    axs[4].plot(single_test_history['time'], single_test_history['drum_1'], label='Single-RL drum position', linestyle='-')
    axs[4].legend()
    axs[4].set_xlabel('Time (s)')
    axs[4].set_ylabel('Drum position (degrees)')
    fig.tight_layout()
    fig.savefig(graph_1_path)

    # Graph 2: PID vs single-RL on test profile without temperature or drum position
    # ##############################################################################
    graph_2_path = graph_path / f'2-singlespeed-{args.test_profile}.png'
    plt.clf()
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 7)) # power, error, drum speed
    axs[0].plot(pid_test_history['time'], pid_test_history['desired_power'], label='Desired power', color='black', linestyle=':')
    axs[0].plot(pid_test_history['time'], pid_test_history['actual_power'], label='PID power', linestyle='-.')
    axs[0].plot(single_test_history['time'], single_test_history['actual_power'], label='Single-RL power', linestyle='-')
    axs[0].legend()
    axs[0].set_ylabel('Power (SPU)')
    axs[1].plot(pid_test_history['time'], pid_test_history['diff'], label='PID error', linestyle=':')
    axs[1].plot(single_test_history['time'], single_test_history['diff'], label='Single-RL error', linestyle='-')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].legend()
    axs[1].set_ylabel('Error (SPU)')
    axs[2].plot(pid_test_history['time'], np.insert(np.diff(pid_test_history['drum_1']), 0, 0), label='PID drum speed', linestyle=':')
    axs[2].plot(single_test_history['time'], np.insert(np.diff(single_test_history['drum_1']), 0, 0), label='Single-RL drum speed', linestyle='-')
    axs[2].legend()
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Drum speed (degrees per second)')
    fig.tight_layout()
    fig.savefig(graph_2_path)

    # Graph 3: PID vs single-RL on test profile with drum position
    # ############################################################
    graph_3_path = graph_path / f'3-singleposition-{args.test_profile}.png'
    plt.clf()
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 7)) # power, error, drum position
    axs[0].plot(pid_test_history['time'], pid_test_history['desired_power'], label='Desired power', color='black', linestyle=':')
    axs[0].plot(pid_test_history['time'], pid_test_history['actual_power'], label='PID power', linestyle='-.')
    axs[0].plot(single_test_history['time'], single_test_history['actual_power'], label='Single-RL power', linestyle='--')
    axs[0].legend()
    axs[0].set_ylabel('Power (SPU)')
    axs[1].plot(pid_test_history['time'], pid_test_history['diff'], label='PID error', linestyle=':')
    axs[1].plot(single_test_history['time'], single_test_history['diff'], label='Single-RL error', linestyle='-')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].legend()
    axs[1].set_ylabel('Error (SPU)')
    axs[2].plot(pid_test_history['time'], pid_test_history['drum_1'], label='PID drum position', linestyle=':')
    axs[2].plot(single_test_history['time'], single_test_history['drum_1'], label='Single-RL drum position', linestyle='-')
    axs[2].legend()
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Drum position (degrees)')
    fig.tight_layout()
    fig.savefig(graph_3_path)

    # Graph 3.5: just marl
    # ###################
    graph_3_5_path = graph_path / f'3.5_marl-{args.test_profile}.png'
    plt.clf()
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 5)) # power, error
    axs[0].plot(marl_test_history['time'], marl_test_history['desired_power'], label='Desired power', color='black', linestyle='-')
    axs[0].plot(marl_test_history['time'], marl_test_history['actual_power'], label='MARL power', linestyle='-')
    axs[0].legend()
    axs[0].set_ylabel('Power (SPU)')
    axs[1].plot(marl_test_history['time'], marl_test_history['diff'], label='MARL error', linestyle='-')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].legend()
    axs[1].set_ylabel('Error (SPU)')
    fig.tight_layout()
    fig.savefig(graph_3_5_path)

    # Graph 4: multi-action vs symmetric vs marl
    # ##########################################
    graph_4_path = graph_path / f'4-multicompare-{args.test_profile}.png'
    plt.clf()
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 5)) # power, error
    axs[0].plot(multi_test_history['time'], multi_test_history['desired_power'], label='Desired power', color='black', linestyle='-')
    axs[0].plot(multi_test_history['time'], multi_test_history['actual_power'], label='Multi-RL power', linestyle='-.')
    axs[0].plot(symmetric_test_history['time'], symmetric_test_history['actual_power'], label='Symmetric-RL power', linestyle=':')
    axs[0].plot(marl_test_history['time'], marl_test_history['actual_power'], label='MARL power', linestyle='-')
    axs[0].legend()
    axs[0].set_ylabel('Power (SPU)')
    axs[1].plot(multi_test_history['time'], multi_test_history['diff'], label='Multi-RL error', linestyle='-.')
    axs[1].plot(symmetric_test_history['time'], symmetric_test_history['diff'], label='Symmetric-RL error', linestyle=':')
    axs[1].plot(marl_test_history['time'], marl_test_history['diff'], label='MARL error', linestyle='-')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].legend()
    axs[1].set_ylabel('Error (SPU)')
    fig.tight_layout()
    fig.savefig(graph_4_path)

    # ######################################################
    # plot training curves multi-action vs symmetric vs marl
    # ######################################################
    multi_logs_path = multi_folder / 'logs/PPO_1'
    symmetric_logs_path = symmetric_folder / 'logs/PPO_1'
    marl_logs_path = marl_folder / 'logs/PPO_1'

    # Graph 5: training curve (ep len) multi-action vs symmetric vs marl
    # ##################################################################
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
    plt.savefig(graph_path / f'5_training-curve-ep-len.png')

    # Graph 6: training curve (ep rew) multi-action vs symmetric vs marl
    # ##################################################################
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
    plt.savefig(graph_path / f'6_training-curve-ep-rew.png')



    # # plot noise graphs
    # ####################
    # single_1noise_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
    #                                                                       'run_path': single_folder,
    #                                                                       'noise': 0.01})
    # single_2noise_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
    #                                                                       'run_path': single_folder,
    #                                                                       'noise': 0.02})
    # single_3noise_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
    #                                                                       'run_path': single_folder,
    #                                                                       'noise': 0.03})
    # # innoculated2_2noise_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
    # #                                                                             'run_path': innoculated2_folder,
    # #                                                                             'noise': 0.03})
    # noise_path = graph_path / '5_noise-power-2SPU.png'
    # data_list = [(single_1noise_history, 'desired_power', 'desired power'),
    #              (single_1noise_history, 'actual_power', '1 SPU noise'),
    #              (single_2noise_history, 'actual_power', '2 SPU noise'),
    #              (single_3noise_history, 'actual_power', '3 SPU noise')]
    # microutils.plot_history(noise_path, data_list, 'Power (SPU)')

    # noise_path = graph_path / '5_noise-diff-2SPU.png'   
    # data_list = [(single_1noise_history, 'diff', '1 SPU noise'),
    #              (single_2noise_history, 'diff', '2 SPU noise'),
    #              (single_3noise_history, 'diff', '3 SPU noise')]
    # microutils.plot_history(noise_path, data_list, 'Power Difference (SPU)')



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
