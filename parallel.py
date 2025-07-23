import os
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv 
from stable_baselines3.common.callbacks import CallbackList
from custom_env.temp_control_env import TempControlEnv
from utils.metrics import start_timer, end_timer
from utils.convergence_callback import ConvergenceLogger
from utils.cpu_gpu_callback import CPU_GPU_Logger
import csv
import time

# --- Configurações do Experimento ---
NUM_RUNS = 5
NUM_ENVS = 4
TOTAL_TIMESTEPS = 100_000
EVAL_EPISODES = 10
EVAL_FREQ_CONVERGENCE = 10_000

os.makedirs("logs", exist_ok=True)
csv_header = ["config", "run", "reward_mean", "reward_std", "time", "cpu_mean", "gpu_mean"]

def make_env_fn(seed):
    def _init():
        env = TempControlEnv()
        env.reset(seed=seed)
        return env
    return _init

def run_parallel_experiment():
    with open("logs/debug_dummy.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

        for run_idx in range(NUM_RUNS):
            seed = 42 + run_idx
            np.random.seed(seed)
            random.seed(seed)

            print(f"\n[RUN {run_idx + 1}] Seed: {seed} - Inicializando ambientes...")
            t_env_start = time.perf_counter()
            env_fns = [make_env_fn(seed + i) for i in range(NUM_ENVS)]
            vec_env = DummyVecEnv(env_fns)
            t_env_end = time.perf_counter()
            print(f"[RUN {run_idx + 1}] Ambientes criados em {t_env_end - t_env_start:.2f}s")

            model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed)

            cpu_gpu_callback = CPU_GPU_Logger()
            convergence_log_dir = os.path.join("logs", f"debug_run_{run_idx+1}")
            convergence_callback = ConvergenceLogger(
                log_dir=convergence_log_dir,
                eval_freq=EVAL_FREQ_CONVERGENCE,
                n_eval_episodes=EVAL_EPISODES
            )
            callback = CallbackList([cpu_gpu_callback, convergence_callback])

            print(f"[RUN {run_idx + 1}] Iniciando treinamento...")
            t_train_start = time.perf_counter()
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
            t_train_end = time.perf_counter()
            train_duration = t_train_end - t_train_start
            print(f"[RUN {run_idx + 1}] Treinamento concluído em {train_duration:.2f}s")

            # Avaliação
            print(f"[RUN {run_idx + 1}] Iniciando avaliação...")
            t_eval_start = time.perf_counter()
            episode_rewards_final = []

            for ep in range(EVAL_EPISODES):
                obs = vec_env.reset()
                episode_reward = 0
                dones = [False] * NUM_ENVS
                steps = 0
                ep_start = time.perf_counter()

                while not all(dones):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, dones, _ = vec_env.step(action)
                    episode_reward += reward.mean()
                    steps += 1
                    if steps % 50 == 0:
                        print(f"[RUN {run_idx + 1}] Avaliação EP {ep+1}: {steps} passos...")

                ep_end = time.perf_counter()
                print(f"[RUN {run_idx + 1}] Episódio {ep+1} recompensa média: {episode_reward:.2f} (tempo: {ep_end - ep_start:.2f}s)")
                episode_rewards_final.append(episode_reward)

            t_eval_end = time.perf_counter()
            eval_duration = t_eval_end - t_eval_start

            mean_reward_final = np.mean(episode_rewards_final)
            std_reward_final = np.std(episode_rewards_final)
            mean_cpu = cpu_gpu_callback.get_mean_cpu_usage()
            mean_gpu = cpu_gpu_callback.get_mean_gpu_usage()

            writer.writerow(["debug_dummy", run_idx + 1, mean_reward_final, std_reward_final, train_duration, mean_cpu, mean_gpu])
            vec_env.close()

            print(f"[RUN {run_idx + 1}] Finalizado. Treino: {train_duration:.2f}s | Avaliação: {eval_duration:.2f}s | Média recompensa: {mean_reward_final:.2f} | CPU: {mean_cpu:.1f}%, GPU: {mean_gpu:.1f}%")

    print("\nTodos os experimentos foram concluídos.")


if __name__ == "__main__":
    run_parallel_experiment()
