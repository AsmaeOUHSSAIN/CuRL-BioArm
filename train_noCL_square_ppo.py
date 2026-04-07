"""Baseline: no curriculum, fully random square parameters (PPO).

"""

import os
import json
import argparse
import numpy as np
import myosuite  
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from gymnasium import Wrapper

from constants import (
    CENTER_X, CENTER_Z, MAX_EPISODE_STEPS, TOLERANCE,
    EVAL_SCALE, EVAL_ROTATION, EVAL_Y_POS,
    SCALE_MIN, SCALE_MAX, ROTATION_MIN, ROTATION_MAX, Y_POS_MIN, Y_POS_MAX,
    FINAL_EVAL_LEVELS,
)
from shape_tracing_myoarm import ShapeTracingWrapper
from utils import (
    MetricsTracker,
    TrainingMetricsCallback,
    EvalWithMetricsCallback,
    run_final_evaluation,
)


SHAPE = "square"
STRATEGY = "noCL"
ALGO = "PPO"


class RandomSquareWrapper(Wrapper):
    """Samples square parameters uniformly across the full ranges."""

    def __init__(self, env, scale_min, scale_max, rotation_min, rotation_max,
                 y_pos_min, y_pos_max, tolerance):
        super().__init__(env)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rotation_min = rotation_min
        self.rotation_max = rotation_max
        self.y_pos_min = y_pos_min
        self.y_pos_max = y_pos_max
        self.tolerance = tolerance

    def reset(self, **kwargs):
        scale = np.random.uniform(self.scale_min, self.scale_max)
        rotation = np.random.uniform(self.rotation_min, self.rotation_max)
        y_pos = np.random.uniform(self.y_pos_min, self.y_pos_max)

        self.env.set_scale(scale)
        self.env.set_rotation_y(rotation)
        self.env.set_center_y(y_pos)
        self.env.set_tolerance(self.tolerance)

        obs, info = self.env.reset(**kwargs)
        info["strategy"] = STRATEGY
        info["curriculum_level"] = -1
        info["rotation_y"] = rotation
        info["y_position"] = y_pos
        return obs, info

    def step(self, action):
        return self.env.step(action)


def make_train_env(rank, seed):
    def _init():
        env = gym.make("myoArmReachRandom-v0", max_episode_steps=MAX_EPISODE_STEPS)
        env.reset(seed=seed + rank)
        env = ShapeTracingWrapper(env, shape=SHAPE, tolerance=TOLERANCE)
        env = RandomSquareWrapper(
            env,
            scale_min=SCALE_MIN, scale_max=SCALE_MAX,
            rotation_min=ROTATION_MIN, rotation_max=ROTATION_MAX,
            y_pos_min=Y_POS_MIN, y_pos_max=Y_POS_MAX,
            tolerance=TOLERANCE,
        )
        return env
    return _init


def make_eval_env(seed):
    env = gym.make("myoArmReachRandom-v0", max_episode_steps=MAX_EPISODE_STEPS)
    env.reset(seed=seed + 1000)
    return ShapeTracingWrapper(
        env, shape=SHAPE, scale=EVAL_SCALE,
        tolerance=TOLERANCE, rotation_y=EVAL_ROTATION,
        center=np.array([CENTER_X, EVAL_Y_POS, CENTER_Z]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    seed = args.seed
    log_dir = args.log_dir or f"./experiments/PPO_square_{STRATEGY}_seed{seed}"
    os.makedirs(log_dir, exist_ok=True)

    experiment_name = f"{ALGO}_{SHAPE}_{STRATEGY}_seed{seed}"
    tracker = MetricsTracker(log_dir=log_dir, experiment_name=experiment_name)

    config = {
        "shape": SHAPE, "strategy": STRATEGY, "algo": ALGO,
        "total_timesteps": args.total_timesteps, "n_envs": args.n_envs,
        "eval_freq": args.eval_freq, "n_eval_episodes": args.n_eval_episodes,
        "max_episode_steps": MAX_EPISODE_STEPS, "tolerance": TOLERANCE,
        "seed": seed,
        "scale_range": [SCALE_MIN, SCALE_MAX],
        "rotation_range": [ROTATION_MIN, ROTATION_MAX],
        "y_pos_range": [Y_POS_MIN, Y_POS_MAX],
        "curriculum_levels": "N/A (random)",
        "ppo_params": {
            "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
            "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_range": 0.2, "ent_coef": 0.01, "net_arch": [256, 256],
        },
        "final_eval_levels": FINAL_EVAL_LEVELS,
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING: {experiment_name}")
    print(f"  Strategy: No Curriculum (random all params)")
    print(f"  Scale: [{SCALE_MIN}, {SCALE_MAX}]")
    print(f"  Rotation: [{ROTATION_MIN}, {ROTATION_MAX}] deg")
    print(f"  Y pos: [{Y_POS_MIN}, {Y_POS_MAX}]")
    print(f"  Tolerance: {TOLERANCE*100:.1f} cm fixed")
    print(f"  Timesteps: {args.total_timesteps:,} | Envs: {args.n_envs}")
    print(f"  Seed: {seed} | Log: {log_dir}")
    print(f"{'='*60}\n")

    train_env = DummyVecEnv([make_train_env(i, seed) for i in range(args.n_envs)])
    train_env = VecMonitor(train_env)
    eval_env = make_eval_env(seed)

    training_cb = TrainingMetricsCallback(tracker=tracker)
    eval_cb = EvalWithMetricsCallback(
        eval_env=eval_env, tracker=tracker,
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        deterministic=True, verbose=1,
    )

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=0.5, verbose=1, seed=seed,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[training_cb, eval_cb],
        progress_bar=True,
    )
    model.save(os.path.join(log_dir, "final_model"))

    print(f"\n  Training complete. Running final evaluation...")
    run_final_evaluation(
        model=model, shape=SHAPE, tracker=tracker,
        difficulty_levels=FINAL_EVAL_LEVELS, n_episodes=50,
    )

    tracker.close()
    train_env.close()
    eval_env.close()

    print(f"\n{'='*60}")
    print(f"  ALL DONE: {experiment_name}")
    print(f"  Final model:  {log_dir}/final_model.zip")
    print(f"  Best model:   {log_dir}/best_model/best_model.zip")
    print(f"{'='*60}")
