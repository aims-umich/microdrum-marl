import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import envs
import microutils
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

    ##########################
    # Multi Drum RL Training #
    ##########################
    multi_folder = Path.cwd() / 'runs' / 'multi-rl'
    multi_folder.mkdir(exist_ok=True, parents=True)
    model_folder = multi_folder / 'models/'
    if not model_folder.exists():  # if a model has already been trained, don't re-train
        print('Training Multi Action RL...')
        training_kwargs['run_path'] = multi_folder
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
        microutils.train_marl(envs.HolosMARL, training_kwargs,
                              total_timesteps=(args.timesteps * 8), n_envs=args.n_envs)
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

    # Clean up run_history
    run_historys = Path.cwd().glob('runs/*/run_history*csv')
    for run_history in run_historys:
        run_history.unlink(missing_ok=True)

    # gather test histories
    #######################
    print(f'testing with {args.test_profile}:')
    pid_folder = Path.cwd() / 'runs' / 'pid'
    pid_folder.mkdir(exist_ok=True, parents=True)
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

    # Graph 4: multi-action vs symmetric vs marl
    # ##########################################
    graph_4_path = graph_path / f'4-multicompare-{args.test_profile}.png'
    plt.clf()
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10)) # power, error
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
    axs[2].plot(multi_test_history['time'], multi_test_history['drum_1'], label='Multi-RL drum 1')
    axs[2].plot(multi_test_history['time'], multi_test_history['drum_2'], label='Multi-RL drum 2')
    axs[2].plot(multi_test_history['time'], multi_test_history['drum_3'], label='Multi-RL drum 3')
    axs[2].plot(multi_test_history['time'], multi_test_history['drum_4'], label='Multi-RL drum 4')    
    axs[2].plot(multi_test_history['time'], multi_test_history['drum_5'], label='Multi-RL drum 5')
    axs[2].plot(multi_test_history['time'], multi_test_history['drum_6'], label='Multi-RL drum 6')
    axs[2].plot(multi_test_history['time'], multi_test_history['drum_7'], label='Multi-RL drum 7')
    axs[2].plot(multi_test_history['time'], multi_test_history['drum_8'], label='Multi-RL drum 8')
    axs[2].legend()
    axs[2].set_ylabel('Drum position (degrees)')
    axs[3].plot(marl_test_history['time'], marl_test_history['drum_1'], label='MARL drum 1')
    axs[3].plot(marl_test_history['time'], marl_test_history['drum_2'], label='MARL drum 2')
    axs[3].plot(marl_test_history['time'], marl_test_history['drum_3'], label='MARL drum 3')
    axs[3].plot(marl_test_history['time'], marl_test_history['drum_4'], label='MARL drum 4')    
    axs[3].plot(marl_test_history['time'], marl_test_history['drum_5'], label='MARL drum 5')
    axs[3].plot(marl_test_history['time'], marl_test_history['drum_6'], label='MARL drum 6')
    axs[3].plot(marl_test_history['time'], marl_test_history['drum_7'], label='MARL drum 7')
    axs[3].plot(marl_test_history['time'], marl_test_history['drum_8'], label='MARL drum 8')
    axs[3].legend()
    axs[3].set_ylabel('Drum position (degrees)')
    fig.tight_layout()
    fig.savefig(graph_4_path)

    # Graph 5: training curves (ep len and rew) multi-action vs symmetric vs marl
    # ##################################################################
    multi_logs_path = multi_folder / 'logs/PPO_1'
    symmetric_logs_path = symmetric_folder / 'logs/PPO_1'
    marl_logs_path = marl_folder / 'logs/PPO_1'
    multi_ep_len = pd.read_csv(multi_logs_path / 'ep_len_mean.csv')
    symmetric_ep_len = pd.read_csv(symmetric_logs_path / 'ep_len_mean.csv')
    marl_ep_len = pd.read_csv(marl_logs_path / 'ep_len_mean.csv')
    marl_ep_len['Step'] = marl_ep_len['Step'] / 8  # normalize by point kinetics simulations run
    multi_ep_rew = pd.read_csv(multi_logs_path / 'ep_rew_mean.csv')
    symmetric_ep_rew = pd.read_csv(symmetric_logs_path / 'ep_rew_mean.csv')
    marl_ep_rew = pd.read_csv(marl_logs_path / 'ep_rew_mean.csv')
    marl_ep_rew['Step'] = marl_ep_rew['Step'] / 8  # normalize by point kinetics simulations run

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10)) # power, error
    axs[0].plot(multi_ep_len['Step'], multi_ep_len['Value'], label='multi-action')
    axs[0].plot(symmetric_ep_len['Step'], symmetric_ep_len['Value'], label='symmetric')
    axs[0].plot(marl_ep_len['Step'], marl_ep_len['Value'], label='marl')
    axs[0].set_ylabel('Episode length (s)')
    axs[0].legend()
    axs[1].plot(multi_ep_rew['Step'], multi_ep_rew['Value'], label='multi-action')
    axs[1].plot(symmetric_ep_rew['Step'], symmetric_ep_rew['Value'], label='symmetric')
    axs[1].plot(marl_ep_rew['Step'], marl_ep_rew['Value'], label='marl')
    axs[1].set_xlabel('Environment timesteps')
    axs[1].set_ylabel('Episode reward')
    axs[1].legend()
    plt.savefig(graph_path / f'5_training-curves.png')

    # Graph 6: run histories for pid, single-rl, and marl at 0.015 noise
    # ##################################################################
    pid_noise_history = microutils.test_pid(envs.HolosSingle, {**testing_kwargs,
                                                               'run_path': pid_folder,
                                                               'noise': 0.02})
    single_noise_history = microutils.test_trained_rl(envs.HolosSingle, {**testing_kwargs,
                                                                         'run_path': single_folder,
                                                                         'noise': 0.02})
    marl_noise_history = microutils.test_trained_marl(envs.HolosMARL, {**testing_kwargs,
                                                                       'run_path': marl_folder,
                                                                       'noise': 0.02})

    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(10, 5)) # power, error
    axs.plot(pid_noise_history['time'], pid_noise_history['desired_power'], label='Desired power', color='k')
    axs.plot(pid_noise_history['time'], pid_noise_history['actual_power'], label='PID')
    axs.plot(single_noise_history['time'], single_noise_history['actual_power'], label='Single-RL')
    axs.plot(marl_noise_history['time'], marl_noise_history['actual_power'], label='MARL')
    axs.set_ylabel('Power (SPU)')
    axs.legend()
    # axs[1].plot(pid_noise_history['time'], np.insert(np.diff(pid_noise_history['drum_1']), 0, 0), label='PID')
    # axs[1].plot(single_noise_history['time'], np.insert(np.diff(single_noise_history['drum_1']), 0, 0), label='Single-RL')
    # axs[1].plot(marl_noise_history['time'], np.insert(np.diff(marl_noise_history['drum_1']), 0, 0), label='MARL')
    # axs[1].axhline(y=0, color='k', linestyle='--')
    # axs[1].set_xlabel('Time (s)')
    # axs[1].set_ylabel('Drum speed (degrees per second)')
    plt.legend()
    plt.savefig(graph_path / f'6_noise-run-histories.png')

    # Graph 7: cae and ce vs noise level for pid, single-rl, and marl
    # ###############################################################
    noise_levels = [0, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03]
    single_noise_path = single_folder / 'noise-metrics.csv'
    if not single_noise_path.exists():
        overall_metrics = microutils.noise_loop(envs.HolosSingle,
                                                {**testing_kwargs,
                                                'run_path': single_folder},
                                                type='rl',
                                                noise_levels=noise_levels)
        overall_metrics.to_csv(single_noise_path, index=True)
    single_noise_metrics = pd.read_csv(single_noise_path, index_col=0)

    pid_noise_path = pid_folder / 'noise-metrics.csv'
    if not pid_noise_path.exists():
        overall_metrics = microutils.noise_loop(envs.HolosSingle,
                                               {**testing_kwargs,
                                                'run_path': pid_folder},
                                               type='pid',
                                               noise_levels=noise_levels)
        overall_metrics.to_csv(pid_noise_path, index=True)
    pid_noise_metrics = pd.read_csv(pid_noise_path, index_col=0)

    marl_noise_path = marl_folder / 'noise-metrics.csv'
    if not marl_noise_path.exists():
        overall_metrics = microutils.noise_loop(envs.HolosMARL,
                                               {**testing_kwargs,
                                                'run_path': marl_folder},
                                               type='marl',
                                               noise_levels=noise_levels)
        overall_metrics.to_csv(marl_noise_path, index=True)
    marl_noise_metrics = pd.read_csv(marl_noise_path, index_col=0)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10)) # power, error
    axs[0].errorbar((single_noise_metrics.index * 100), single_noise_metrics['cae_mean'], yerr=single_noise_metrics['cae_std'], label='Single-RL')
    axs[0].errorbar((pid_noise_metrics.index * 100), pid_noise_metrics['cae_mean'], yerr=pid_noise_metrics['cae_std'], label='PID')
    axs[0].errorbar((marl_noise_metrics.index * 100), marl_noise_metrics['cae_mean'], yerr=marl_noise_metrics['cae_std'], label='MARL')
    axs[0].set_ylabel('CAE (SPU)')
    axs[0].legend()
    axs[1].errorbar((single_noise_metrics.index * 100), single_noise_metrics['ce_mean'], yerr=single_noise_metrics['ce_std'], label='Single-RL')
    axs[1].errorbar((pid_noise_metrics.index * 100), pid_noise_metrics['ce_mean'], yerr=pid_noise_metrics['ce_std'], label='PID')
    axs[1].errorbar((marl_noise_metrics.index * 100), marl_noise_metrics['ce_mean'], yerr=marl_noise_metrics['ce_std'], label='MARL')
    axs[1].set_xlabel('Noise standard deviation (SPU)')
    axs[1].set_ylabel('Control Effort (degrees)')
    plt.legend()
    plt.savefig(graph_path / f'7_noise-metrics.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
