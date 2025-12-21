import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import envs
import microutils

plt.style.use("tableau-colorblind10")


def main(args):
    # create interpolated power profiles
    training_profile = interp1d(
        [0, 10, 20, 30, 50, 70, 120, 140, 160, 195, 200],  # times (s)
        [100, 100, 98, 99, 80, 60, 60, 70, 70, 80, 80],
    )  # power (SPU)
    testing_profile = interp1d(
        [0, 10, 70, 100, 115, 125, 150, 180, 200],  # times (s)
        [100, 100, 50, 50, 65, 65, 50, 80, 80],
    )  # power (SPU)
    lowpower_profile = interp1d(
        [0, 5, 100, 200],  # times (s)
        [100, 100, 30, 90],
    )  # power (SPU)
    longtest_profile = interp1d(
        [
            0,
            2000,
            3000,
            5500,
            6000,
            10000,
            10020,
            12500,
            14000,
            16000,
            16010,
            20000,
        ],  # times (s)
        [100, 100, 90, 90, 45, 45, 65, 65, 80, 80, 95, 95],
    )  # power (SPU)

    match args.test_profile:
        case "longtest":
            test_profile = longtest_profile
            episode_length = 20000
        case "lowpower":
            test_profile = lowpower_profile
            episode_length = 200
        case "train":
            test_profile = training_profile
            episode_length = 200
        case _:
            test_profile = testing_profile
            episode_length = 200

    training_kwargs = {
        "profile": training_profile,
        "episode_length": 200,
        "train_mode": True,
    }
    testing_kwargs = {
        "profile": test_profile,
        "episode_length": episode_length,
        "valid_maskings": (args.disabled_drums,),
        "train_mode": False,
    }

    for uncertainty in args.uncertainties:
        training_kwargs["uncertainty"] = uncertainty

        ###########################################
        # Single Action Uncertain Rho RL Training #
        ###########################################
        run_folder = Path.cwd() / "runs" / f"single-rl-{uncertainty}"
        run_folder.mkdir(exist_ok=True, parents=True)
        model_folder = run_folder / "models/"
        if not model_folder.exists():  # if a model has already been trained, don't re-train
            print(f"Training Single Action Uncertain Rho = {uncertainty} RL...")
            training_kwargs["run_path"] = run_folder
            microutils.train_rl(
                envs.HolosSingle,
                training_kwargs,
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
            )

        #######################
        # gather test history #
        #######################
        pid_test_history = microutils.test_pid(
            envs.HolosSingle, {**testing_kwargs, "run_path": run_folder}
        )
        run_test_history = microutils.test_trained_rl(
            envs.HolosSingle,
            {
                **testing_kwargs,
                "run_path": run_folder,
            },
        )

        graph_path = run_folder / "graphs"
        graph_path.mkdir(exist_ok=True, parents=True)

        # Graph 1: PID vs single-RL on test profile with temperature included
        # ###################################################################
        graph_1_path = graph_path / f"1-singletemp-{args.test_profile}.png"
        plt.clf()
        fig, axs = plt.subplots(
            5, 1, sharex=True, figsize=(10, 12)
        )  # power, error, temps, drum speed, drum position
        axs[0].plot(
            pid_test_history["time"],
            pid_test_history["desired_power"],
            label="Desired power",
            color="black",
            linestyle="-",
        )
        axs[0].plot(
            pid_test_history["time"],
            pid_test_history["actual_power"],
            label="PID power",
            linestyle=":",
        )
        axs[0].plot(
            run_test_history["time"],
            run_test_history["actual_power"],
            label="Single-RL power",
            linestyle="--",
        )
        axs[0].legend()
        axs[0].set_ylabel("Power (SPU)")
        axs[1].plot(
            pid_test_history["time"],
            pid_test_history["diff"],
            label="PID error",
            linestyle=":",
        )
        axs[1].plot(
            run_test_history["time"],
            run_test_history["diff"],
            label="Single-RL error",
            linestyle="-",
        )
        axs[1].axhline(y=0, color="black", linestyle="--")
        axs[1].legend()
        axs[1].set_ylabel("Error (SPU)")
        axs[2].plot(
            pid_test_history["time"], pid_test_history["Tf"], label="Fuel temperature"
        )
        axs[2].plot(
            pid_test_history["time"], pid_test_history["Tm"], label="Moderator temperature"
        )
        axs[2].plot(
            pid_test_history["time"], pid_test_history["Tc"], label="Coolant temperature"
        )
        axs[2].legend()
        axs[2].set_ylabel("Temperature (deg C)")
        axs[3].plot(
            pid_test_history["time"],
            np.insert(np.diff(pid_test_history["drum_1"]), 0, 0),
            label="PID drum speed",
            linestyle=":",
        )
        axs[3].plot(
            run_test_history["time"],
            np.insert(np.diff(run_test_history["drum_1"]), 0, 0),
            label="Single-RL drum speed",
            linestyle="-",
        )
        axs[3].legend()
        axs[3].set_ylabel("Drum speed (degrees per second)")
        axs[4].plot(
            pid_test_history["time"],
            pid_test_history["drum_1"],
            label="PID drum position",
            linestyle=":",
        )
        axs[4].plot(
            run_test_history["time"],
            run_test_history["drum_1"],
            label="Single-RL drum position",
            linestyle="-",
        )
        axs[4].legend()
        axs[4].set_xlabel("Time (s)")
        axs[4].set_ylabel("Drum position (degrees)")
        fig.tight_layout()
        fig.savefig(graph_1_path, dpi=300)

        # Graph 2: PID vs single-RL on test profile without temperature or drum position
        # ##############################################################################
        graph_2_path = graph_path / f"2-singlespeed-{args.test_profile}.png"
        plt.clf()
        fig, axs = plt.subplots(
            3, 1, sharex=True, figsize=(10, 7)
        )  # power, error, drum speed
        axs[0].plot(
            pid_test_history["time"],
            pid_test_history["desired_power"],
            label="Desired power",
            color="black",
            linestyle="-",
        )
        axs[0].plot(
            pid_test_history["time"],
            pid_test_history["actual_power"],
            label="PID power",
            linestyle=":",
        )
        axs[0].plot(
            run_test_history["time"],
            run_test_history["actual_power"],
            label="Single-RL power",
            linestyle="--",
        )
        axs[0].legend()
        axs[0].set_ylabel("Power (SPU)")
        axs[1].plot(
            pid_test_history["time"],
            pid_test_history["diff"],
            label="PID error",
            linestyle=":",
        )
        axs[1].plot(
            run_test_history["time"],
            run_test_history["diff"],
            label="Single-RL error",
            linestyle="-",
        )
        axs[1].axhline(y=0, color="black", linestyle="--")
        axs[1].legend()
        axs[1].set_ylabel("Error (SPU)")
        axs[2].plot(
            pid_test_history["time"],
            np.insert(np.diff(pid_test_history["drum_1"]), 0, 0),
            label="PID drum speed",
            linestyle=":",
        )
        axs[2].plot(
            run_test_history["time"],
            np.insert(np.diff(run_test_history["drum_1"]), 0, 0),
            label="Single-RL drum speed",
            linestyle="-",
        )
        axs[2].legend()
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Drum speed (degrees per second)")
        fig.tight_layout()
        fig.savefig(graph_2_path, dpi=300)

        # Graph 3: PID vs single-RL on test profile with drum position
        # ############################################################
        graph_3_path = graph_path / f"3-singleposition-{args.test_profile}.png"
        plt.clf()
        fig, axs = plt.subplots(
            4, 1, sharex=True, figsize=(10, 9)
        )  # power, error, xenon + iodine, drum position
        axs[0].plot(
            pid_test_history["time"],
            pid_test_history["desired_power"],
            label="Desired power",
            color="black",
            linestyle="-",
        )
        axs[0].plot(
            pid_test_history["time"],
            pid_test_history["actual_power"],
            label="PID power",
            linestyle=":",
        )
        axs[0].plot(
            run_test_history["time"],
            run_test_history["actual_power"],
            label="Single-RL power",
            linestyle="--",
        )
        axs[0].legend()
        axs[0].set_ylabel("Power (SPU)")
        axs[1].plot(
            pid_test_history["time"],
            pid_test_history["diff"],
            label="PID error",
            linestyle=":",
        )
        axs[1].plot(
            run_test_history["time"],
            run_test_history["diff"],
            label="Single-RL error",
            linestyle="-",
        )
        axs[1].axhline(y=0, color="black", linestyle="--")
        axs[1].legend()
        axs[1].set_ylabel("Error (SPU)")
        axs[2].plot(
            pid_test_history["time"],
            pid_test_history["Xe"],
            label="PID Xenon",
            linestyle=":",
        )
        axs[2].plot(
            run_test_history["time"],
            run_test_history["Xe"],
            label="Single-RL Xenon",
            linestyle="-",
        )
        axs[2].plot(
            pid_test_history["time"],
            pid_test_history["I"],
            label="PID Iodine",
            linestyle=":",
        )
        axs[2].plot(
            run_test_history["time"],
            run_test_history["I"],
            label="Single-RL Iodine",
            linestyle="-",
        )
        axs[2].legend()
        axs[2].set_ylabel("Concentration (m^-3)")
        axs[3].plot(
            pid_test_history["time"],
            pid_test_history["drum_1"],
            label="PID drum position",
            linestyle=":",
        )
        axs[3].plot(
            run_test_history["time"],
            run_test_history["drum_1"],
            label="Single-RL drum position",
            linestyle="-",
        )
        axs[3].legend()
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Drum position (degrees)")
        fig.tight_layout()
        fig.savefig(graph_3_path, dpi=300)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--test_profile",
        type=str,
        default="longtest",
        help="Profile to use for testing (test, train, longtest, lowpower)",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=5_000_000,
        help="Number of timesteps to train for",
    )
    parser.add_argument(
        "-d",
        "--disabled_drums",
        type=int,
        default=0,
        help="Number of drums to disable during testing",
    )
    parser.add_argument(
        "-n",
        "--n_envs",
        type=int,
        default=10,
        help="Number of environments to use for training",
    )
    parser.add_argument(
        "-u",
        "--uncertainties",
        type=list,
        default=[
            0,
            # 0, 0.2, 0.4, 0.6, 0.8, 1, 2, 5,
            ],
        help="List of uncertainties to use for study",
    )
    args = parser.parse_args()
    main(args)
