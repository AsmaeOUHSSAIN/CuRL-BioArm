"""Metrics tracking + SB3 callbacks + final-evaluation helper."""

import os
import csv
import json
import time
import numpy as np
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

from constants import CENTER_X, CENTER_Z


MONITOR_COLUMNS = [
    "episode", "global_step",
    "total_reward", "mean_reward", "n_steps",
    "waypoints_reached", "total_waypoints", "waypoint_progress",
    "success", "mean_distance", "final_distance",
    "mean_tracking_error", "distance_traveled", "scale",
]

EVAL_MONITOR_COLUMNS = [
    "global_step", "level", "scale", "tolerance",
    "episode", "total_reward", "success", "waypoint_progress",
    "waypoints_reached", "total_waypoints",
    "mean_tracking_error", "mean_distance", "distance_traveled",
]


class MetricsTracker:

    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.train_dir = os.path.join(log_dir, "train_metrics")
        self.eval_dir = os.path.join(log_dir, "eval_metrics")
        self.tb_dir = os.path.join(log_dir, "tensorboard")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        self.train_csv_path = os.path.join(self.train_dir, "monitor.csv")
        self._train_csv_file = open(self.train_csv_path, "w", newline="")
        self._train_csv_writer = csv.DictWriter(self._train_csv_file, fieldnames=MONITOR_COLUMNS)
        self._train_csv_writer.writeheader()

        self.eval_csv_path = os.path.join(self.eval_dir, "monitor.csv")
        self._eval_csv_file = open(self.eval_csv_path, "w", newline="")
        self._eval_csv_writer = csv.DictWriter(self._eval_csv_file, fieldnames=EVAL_MONITOR_COLUMNS)
        self._eval_csv_writer.writeheader()

        self.train_log = {
            "experiment_name": experiment_name,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "episodes": [],
            "checkpoints": [],
        }
        self.eval_log = {"experiment_name": experiment_name, "evaluations": []}

        self._episode_buffer = defaultdict(list)
        self._current_episode = {"rewards": [], "distances": []}
        self._episode_count = 0
        self._completed_count = 0

    def log_step(self, info, reward, global_step):
        self._current_episode["rewards"].append(float(reward))
        if "current_distance" in info:
            self._current_episode["distances"].append(float(info["current_distance"]))

    def log_episode_end(self, info, global_step):
        self._episode_count += 1
        completed = info.get("shape_completed", False)
        if completed:
            self._completed_count += 1

        episode_data = {
            "episode": self._episode_count,
            "global_step": int(global_step),
            "total_reward": float(np.sum(self._current_episode["rewards"])),
            "mean_reward": float(np.mean(self._current_episode["rewards"])),
            "n_steps": len(self._current_episode["rewards"]),
            "waypoints_reached": int(info.get("waypoints_reached", 0)),
            "total_waypoints": int(info.get("n_waypoints", 0)),
            "waypoint_progress": float(info.get("waypoint_progress", 0)),
            "success": bool(completed),
            "mean_distance": float(np.mean(self._current_episode["distances"])) if self._current_episode["distances"] else 0,
            "final_distance": float(info.get("current_distance", 0)),
            "mean_tracking_error": float(info.get("mean_tracking_error", 0)),
            "distance_traveled": float(info.get("distance_traveled", 0)),
            "scale": float(info.get("scale", 0)),
            "difficulty": float(info.get("difficulty", 0)),
        }

        self.train_log["episodes"].append(episode_data)

        self.tb_writer.add_scalar("episode/total_reward", episode_data["total_reward"], self._episode_count)
        self.tb_writer.add_scalar("episode/waypoint_progress", episode_data["waypoint_progress"], self._episode_count)
        self.tb_writer.add_scalar("episode/waypoints_reached", episode_data["waypoints_reached"], self._episode_count)
        self.tb_writer.add_scalar("episode/success", float(completed), self._episode_count)
        self.tb_writer.add_scalar("episode/mean_distance", episode_data["mean_distance"], self._episode_count)
        self.tb_writer.add_scalar("episode/mean_tracking_error", episode_data["mean_tracking_error"], self._episode_count)
        self.tb_writer.add_scalar("episode/distance_traveled", episode_data["distance_traveled"], self._episode_count)
        self.tb_writer.add_scalar("episode/scale", episode_data["scale"], self._episode_count)
        self.tb_writer.add_scalar(
            "episode/success_rate",
            self._completed_count / self._episode_count,
            self._episode_count,
        )

        self._current_episode = {"rewards": [], "distances": []}

        csv_row = {k: episode_data.get(k, "") for k in MONITOR_COLUMNS}
        self._train_csv_writer.writerow(csv_row)
        self._train_csv_file.flush()
        return episode_data

    def log_checkpoint(self, global_step, eval_reward, eval_success_rate):
        checkpoint = {
            "global_step": int(global_step),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eval_mean_reward": float(eval_reward),
            "eval_success_rate": float(eval_success_rate),
            "total_episodes": self._episode_count,
            "total_completions": self._completed_count,
            "overall_success_rate": self._completed_count / max(self._episode_count, 1),
        }
        self.train_log["checkpoints"].append(checkpoint)
        self.tb_writer.add_scalar("checkpoint/eval_mean_reward", eval_reward, global_step)
        self.tb_writer.add_scalar("checkpoint/eval_success_rate", eval_success_rate, global_step)

    def log_evaluation(self, global_step, level_name, scale, tolerance, episodes_data):
        rewards = [e["total_reward"] for e in episodes_data]
        successes = [e["success"] for e in episodes_data]
        progresses = [e["waypoint_progress"] for e in episodes_data]
        track_errors = [e["mean_tracking_error"] for e in episodes_data]

        eval_entry = {
            "global_step": int(global_step),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": level_name,
            "scale": float(scale),
            "tolerance": float(tolerance),
            "n_episodes": len(episodes_data),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_waypoint_progress": float(np.mean(progresses)),
            "std_waypoint_progress": float(np.std(progresses)),
            "mean_tracking_error": float(np.mean(track_errors)),
            "std_tracking_error": float(np.std(track_errors)),
            "episodes": episodes_data,
        }

        self.eval_log["evaluations"].append(eval_entry)

        prefix = f"eval_{level_name}"
        self.tb_writer.add_scalar(f"{prefix}/mean_reward", eval_entry["mean_reward"], global_step)
        self.tb_writer.add_scalar(f"{prefix}/success_rate", eval_entry["success_rate"], global_step)
        self.tb_writer.add_scalar(f"{prefix}/mean_wp_progress", eval_entry["mean_waypoint_progress"], global_step)
        self.tb_writer.add_scalar(f"{prefix}/mean_tracking_error", eval_entry["mean_tracking_error"], global_step)

        for ep in episodes_data:
            csv_row = {
                "global_step": int(global_step),
                "level": level_name,
                "scale": float(scale),
                "tolerance": float(tolerance),
                "episode": ep.get("episode", ""),
                "total_reward": ep.get("total_reward", ""),
                "success": ep.get("success", ""),
                "waypoint_progress": ep.get("waypoint_progress", ""),
                "waypoints_reached": ep.get("waypoints_reached", ""),
                "total_waypoints": ep.get("total_waypoints", ""),
                "mean_tracking_error": ep.get("mean_tracking_error", ""),
                "mean_distance": ep.get("mean_distance", ""),
                "distance_traveled": ep.get("distance_traveled", ""),
            }
            self._eval_csv_writer.writerow(csv_row)
        self._eval_csv_file.flush()
        return eval_entry

    def save(self):
        self.train_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.train_log["total_episodes"] = self._episode_count
        self.train_log["total_completions"] = self._completed_count
        self.train_log["final_success_rate"] = self._completed_count / max(self._episode_count, 1)

        train_path = os.path.join(self.train_dir, "train_metrics.json")
        with open(train_path, "w") as f:
            json.dump(self.train_log, f, indent=2)

        eval_path = os.path.join(self.eval_dir, "eval_metrics.json")
        with open(eval_path, "w") as f:
            json.dump(self.eval_log, f, indent=2)

        self.tb_writer.flush()
        print(f"  Metrics saved to {train_path}")
        print(f"  Metrics saved to {eval_path}")

    def close(self):
        self.save()
        self._train_csv_file.close()
        self._eval_csv_file.close()
        self.tb_writer.close()


class TrainingMetricsCallback(BaseCallback):

    def __init__(self, tracker, verbose=0):
        super().__init__(verbose)
        self.tracker = tracker

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for info, done, reward in zip(infos, dones, rewards):
            self.tracker.log_step(info, reward, self.num_timesteps)
            if done:
                self.tracker.log_episode_end(info, self.num_timesteps)
        return True

    def _on_training_end(self):
        self.tracker.save()


class EvalWithMetricsCallback(BaseCallback):

    def __init__(self, eval_env, tracker, eval_freq, n_eval_episodes=10,
                 best_model_save_path=None, deterministic=True, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tracker = tracker
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf

        if best_model_save_path:
            os.makedirs(best_model_save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True

    def _run_evaluation(self):
        episodes_data = []
        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            total_reward = 0
            distances = []
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                if "current_distance" in info:
                    distances.append(float(info["current_distance"]))
                done = terminated or truncated

            episodes_data.append({
                "episode": ep,
                "total_reward": float(total_reward),
                "success": bool(info.get("shape_completed", False)),
                "waypoint_progress": float(info.get("waypoint_progress", 0)),
                "waypoints_reached": int(info.get("waypoints_reached", 0)),
                "total_waypoints": int(info.get("n_waypoints", 0)),
                "mean_tracking_error": float(info.get("mean_tracking_error", 0)),
                "mean_distance": float(np.mean(distances)) if distances else 0,
                "distance_traveled": float(info.get("distance_traveled", 0)),
            })

        eval_entry = self.tracker.log_evaluation(
            global_step=self.num_timesteps,
            level_name="training_eval",
            scale=getattr(self.eval_env, "scale", 0),
            tolerance=getattr(self.eval_env, "tolerance", 0),
            episodes_data=episodes_data,
        )

        mean_reward = eval_entry["mean_reward"]
        success_rate = eval_entry["success_rate"]
        self.tracker.log_checkpoint(self.num_timesteps, mean_reward, success_rate)

        if self.verbose:
            print(f"  Eval @ {self.num_timesteps:,} steps | "
                  f"Reward={mean_reward:.1f} | "
                  f"Success={success_rate*100:.1f}% | "
                  f"Progress={eval_entry['mean_waypoint_progress']*100:.1f}%")

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.best_model_save_path:
                path = os.path.join(self.best_model_save_path, "best_model")
                self.model.save(path)
                if self.verbose:
                    print(f"  New best model saved! (reward={mean_reward:.1f})")

        if self.best_model_save_path:
            checkpoint_dir = os.path.join(os.path.dirname(self.best_model_save_path), "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"model_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                print(f"  Checkpoint saved: {path}")


def run_final_evaluation(model, shape, tracker, difficulty_levels, n_episodes=50):
    from shape_tracing_myoarm import ShapeTracingWrapper
    import myosuite 
    from myosuite.utils import gym

    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION")
    print(f"{'='*60}\n")

    all_results = {}

    for level_name, cfg in difficulty_levels.items():
        env = gym.make("myoArmReachRandom-v0", max_episode_steps=1000)

        rotation_y = cfg.get("rotation", 0.0)
        y_pos = cfg.get("y_pos", -0.25)
        center = np.array([CENTER_X, y_pos, CENTER_Z]) if "y_pos" in cfg else None

        env = ShapeTracingWrapper(
            env, shape=shape,
            scale=cfg["scale"], tolerance=cfg["tolerance"],
            rotation_y=rotation_y, center=center,
        )

        episodes_data = []
        for ep in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0
            distances = []
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if "current_distance" in info:
                    distances.append(float(info["current_distance"]))
                done = terminated or truncated

            episodes_data.append({
                "episode": ep,
                "total_reward": float(total_reward),
                "success": bool(info.get("shape_completed", False)),
                "waypoint_progress": float(info.get("waypoint_progress", 0)),
                "waypoints_reached": int(info.get("waypoints_reached", 0)),
                "total_waypoints": int(info.get("n_waypoints", 0)),
                "mean_tracking_error": float(info.get("mean_tracking_error", 0)),
                "mean_distance": float(np.mean(distances)) if distances else 0,
                "distance_traveled": float(info.get("distance_traveled", 0)),
            })

        eval_entry = tracker.log_evaluation(
            global_step=0,
            level_name=level_name,
            scale=cfg["scale"],
            tolerance=cfg["tolerance"],
            episodes_data=episodes_data,
        )

        all_results[level_name] = eval_entry
        env.close()

        print(f"  {level_name:8s} | Success={eval_entry['success_rate']*100:.1f}% | "
              f"Progress={eval_entry['mean_waypoint_progress']*100:.1f}% | "
              f"TrackErr={eval_entry['mean_tracking_error']*100:.2f}cm | "
              f"Reward={eval_entry['mean_reward']:.1f}")

    tracker.save()
    return all_results
