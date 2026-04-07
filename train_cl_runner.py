import os
import json
import numpy as np
import myosuite  
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from constants import (
    CENTER_X, CENTER_Z, MAX_EPISODE_STEPS, TOLERANCE,
    EVAL_SCALE, EVAL_ROTATION, EVAL_Y_POS,
    SCALE_MIN, SCALE_MAX, ROTATION_MIN, ROTATION_MAX, Y_POS_MIN, Y_POS_MAX,
    FINAL_EVAL_LEVELS,
)
from shape_tracing_myoarm import ShapeTracingWrapper
from curriculum import CurriculumState, CurriculumSquareWrapper, CurriculumEvalCallback
from utils import (
    MetricsTracker,
    TrainingMetricsCallback,
    EvalWithMetricsCallback,
    run_final_evaluation,
)


SHAPE = "square"
ALGO = "PPO"


def _make_train_env(rank, curriculum_state, strategy_name, seed):
    def _init():
        env = gym.make("myoArmReachRandom-v0", max_episode_steps=MAX_EPISODE_STEPS)
        env.reset(seed=seed + rank)
        env = ShapeTracingWrapper(env, shape=SHAPE, tolerance=TOLERANCE)
        env = CurriculumSquareWrapper(env, curriculum_state, strategy_name, tolerance=TOLERANCE)
        return env
    return _init


def _make_eval_env(seed):
    env = gym.make("myoArmReachRandom-v0", max_episode_steps=MAX_EPISODE_STEPS)
    env.reset(seed=seed + 1000)
    return ShapeTracingWrapper(
        env, shape=SHAPE, scale=EVAL_SCALE,
        tolerance=TOLERANCE, rotation_y=EVAL_ROTATION,
        center=np.array([CENTER_X, EVAL_Y_POS, CENTER_Z]),
    )


def _make_curriculum_eval_env(seed):
    env = gym.make("myoArmReachRandom-v0", max_episode_steps=MAX_EPISODE_STEPS)
    env.reset(seed=seed + 2000)
    return ShapeTracingWrapper(
        env, shape=SHAPE, scale=0.05,
        tolerance=TOLERANCE, rotation_y=0.0,
        center=np.array([CENTER_X, -0.20, CENTER_Z]),
    )


def run_cl_experiment(
    strategy_name,
    curriculum_levels,
    log_dir,
    total_timesteps=10_000_000,
    n_envs=8,
    eval_freq=50_000,
    n_eval_episodes=10,
    n_curriculum_eval_episodes=10,
    seed=0,
):
    os.makedirs(log_dir, exist_ok=True)
    experiment_name = f"{ALGO}_{SHAPE}_{strategy_name}_seed{seed}"

    curriculum_state = CurriculumState(curriculum_levels)
    tracker = MetricsTracker(log_dir=log_dir, experiment_name=experiment_name)

    config = {
        "shape": SHAPE, "strategy": strategy_name, "algo": ALGO,
        "total_timesteps": total_timesteps, "n_envs": n_envs,
        "tolerance": TOLERANCE, "seed": seed,
        "scale_range": [SCALE_MIN, SCALE_MAX],
        "rotation_range": [ROTATION_MIN, ROTATION_MAX],
        "y_pos_range": [Y_POS_MIN, Y_POS_MAX],
        "curriculum_levels": curriculum_levels,
        "advancement_criterion": "eval success_rate >= threshold on current level tasks",
        "n_curriculum_eval_episodes": n_curriculum_eval_episodes,
        "final_eval_levels": FINAL_EVAL_LEVELS,
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING: {experiment_name}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Levels:")
    for i, lvl in enumerate(curriculum_levels):
        thresh = f"{lvl['advance_threshold']*100:.0f}%" if lvl["advance_threshold"] else "terminal"
        print(f"    L{i} scale=[{lvl['scale_min']}, {lvl['scale_max']}]  "
              f"rot=[{lvl['rotation_min']:+.0f}, {lvl['rotation_max']:+.0f}]  "
              f"y=[{lvl['y_pos_min']}, {lvl['y_pos_max']}]  thresh={thresh}")
    print(f"  Tolerance: {TOLERANCE*100:.1f} cm | Timesteps: {total_timesteps:,}")
    print(f"  Seed: {seed} | Log: {log_dir}")
    print(f"{'='*60}\n")

    train_env = DummyVecEnv([
        _make_train_env(i, curriculum_state, strategy_name, seed) for i in range(n_envs)
    ])
    train_env = VecMonitor(train_env)
    eval_env = _make_eval_env(seed)
    curriculum_eval_env = _make_curriculum_eval_env(seed)

    training_cb = TrainingMetricsCallback(tracker=tracker)
    eval_cb = EvalWithMetricsCallback(
        eval_env=eval_env, tracker=tracker,
        eval_freq=eval_freq // n_envs, n_eval_episodes=n_eval_episodes,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        deterministic=True, verbose=1,
    )
    curriculum_eval_cb = CurriculumEvalCallback(
        eval_env=curriculum_eval_env,
        curriculum_state=curriculum_state,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=n_curriculum_eval_episodes,
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
        total_timesteps=total_timesteps,
        callback=[training_cb, eval_cb, curriculum_eval_cb],
        progress_bar=True,
    )
    model.save(os.path.join(log_dir, "final_model"))

    cl_summary = {
        "final_level": curriculum_state.current_level,
        "final_level_name": curriculum_state.cfg["name"],
        "total_advances": curriculum_state.total_advances,
        "advance_history": curriculum_state.advance_history,
        "curriculum_eval_history": curriculum_eval_cb.eval_history,
    }
    with open(os.path.join(log_dir, "curriculum_history.json"), "w") as f:
        json.dump(cl_summary, f, indent=2)

    print(f"\n  Training complete. Running final evaluation...")
    run_final_evaluation(
        model=model, shape=SHAPE, tracker=tracker,
        difficulty_levels=FINAL_EVAL_LEVELS, n_episodes=50,
    )

    tracker.close()
    train_env.close()
    eval_env.close()
    curriculum_eval_env.close()

    print(f"\n{'='*60}")
    print(f"  ALL DONE: {experiment_name}")
    print(f"  Final level: {cl_summary['final_level']} ({cl_summary['final_level_name']})")
    print(f"  Advances: {cl_summary['total_advances']}")
    print(f"  CL history: {log_dir}/curriculum_history.json")
    print(f"{'='*60}")
