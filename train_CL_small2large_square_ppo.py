"""Curriculum learning: small -> large square shapes (PPO)."""

import argparse

from constants import (
    SCALE_MIN, SCALE_MAX, ROTATION_MIN, ROTATION_MAX, Y_POS_MIN, Y_POS_MAX,
)
from train_cl_runner import run_cl_experiment


CURRICULUM_LEVELS = [
    {
        "name": "L0_small",
        "scale_min": 0.02, "scale_max": 0.04,
        "rotation_min": -10.0, "rotation_max": 10.0,
        "y_pos_min": -0.24, "y_pos_max": -0.16,
        "advance_threshold": 0.70,
    },
    {
        "name": "L1_small_medium",
        "scale_min": 0.02, "scale_max": 0.06,
        "rotation_min": -25.0, "rotation_max": 25.0,
        "y_pos_min": -0.24, "y_pos_max": -0.12,
        "advance_threshold": 0.60,
    },
    {
        "name": "L2_small_medium_large",
        "scale_min": SCALE_MIN, "scale_max": SCALE_MAX,
        "rotation_min": ROTATION_MIN, "rotation_max": ROTATION_MAX,
        "y_pos_min": Y_POS_MIN, "y_pos_max": Y_POS_MAX,
        "advance_threshold": None,
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    log_dir = args.log_dir or f"./experiments/PPO_square_CL_small2large_seed{args.seed}"

    run_cl_experiment(
        strategy_name="CL_small2large",
        curriculum_levels=CURRICULUM_LEVELS,
        log_dir=log_dir,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
