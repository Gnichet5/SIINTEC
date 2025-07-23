import os
import random
import numpy as np
from stable_baselines3 import PPO
from custom_env.temp_control_env import TempControlEnv
from utils.metrics import start_timer, end_timer, get_cpu_usage, get_gpu_usage
import csv

NUM_RUNS = 3
TOTAL_TIMESTEPS = 10_000
EVAL_EPISODES = 10
SEED_BASE = 42

os.makedirs("logs", exist_ok=True)

csv_header = ["config", "run", "reward_mean", "reward_std", "time", "cpu_mean", "gpu_mean"]

with open("logs/base.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

    for run_idx in range(NUM_RUNS):
        seed = SEED_BASE + run_idx
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n--- Base Run {run_idx + 1}/{NUM_RUNS} (Seed: {seed}) ---")

        env = TempControlEnv()
        obs, info = env.reset(seed=seed)

        model = PPO("MlpPolicy", env, verbose=0, seed=seed)

        cpu_usages = []
        gpu_usages = []

        timer = start_timer()
        def callback(_locals, _globals):
            cpu_usages.append(get_cpu_usage())
            gpu_usages.append(get_gpu_usage())
            return True

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        duration = end_timer(timer)

        rewards = []
        for ep in range(EVAL_EPISODES):
            obs, info = env.reset(seed=seed + ep)
            done = False
            truncated = False
            total_reward = 0.0
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_cpu = np.mean(cpu_usages) if cpu_usages else 0
        mean_gpu = np.mean(gpu_usages) if gpu_usages else 0

        writer.writerow(["base", run_idx + 1, mean_reward, std_reward, duration, mean_cpu, mean_gpu])

        env.close()

print("\nBase experiments completed and logged.")
