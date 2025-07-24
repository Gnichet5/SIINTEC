from stable_baselines3 import PPO
from custom_env.temp_control_env import TempControlEnv
from utils.metrics import *
import csv
import numpy as np
import random
import os

# Configuração para replicar experimentos
NUM_RUNS = 5
TOTAL_TIMESTEPS = 50000
EVAL_EPISODES = 20


# Garante que o diretório de logs exista
os.makedirs("logs", exist_ok=True)

# Cabeçalho do CSV
csv_header = ["config", "run", "reward_mean", "reward_std", "time", "cpu_mean", "gpu_mean"]

with open("logs/base.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

    for run_idx in range(NUM_RUNS):
        # Define uma seed para o experimento (para reprodutibilidade de cada run)
        seed = 42 + run_idx
        np.random.seed(seed)
        random.seed(seed)
        # Para SB3, a seed do ambiente precisa ser passada na criação ou reset
        # PPO também tem um parâmetro 'seed'
        
        print(f"\n--- Base Run {run_idx + 1}/{NUM_RUNS} (Seed: {seed}) ---")

        env = TempControlEnv()
        # Definir a seed para o ambiente se necessário (alguns envs têm um método .seed())
        # No caso do gymnasium/gym, a seed é passada no reset para determinismo
        obs, info = env.reset(seed=seed)
        
        model = PPO("MlpPolicy", env, verbose=0, seed=seed) # Definir seed para o modelo PPO

        cpu_usages_during_train = []
        gpu_usages_during_train = []

        # Monitoramento durante o treinamento (pode ser mais granular se necessário)
        train_start_time = start_timer()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=lambda _locals, _globals: cpu_usages_during_train.append(get_cpu_usage()) or gpu_usages_during_train.append(get_gpu_usage()) or True)
        duration = end_timer(train_start_time)

        # Avaliação do agente
        episode_rewards = []
        for eval_episode in range(EVAL_EPISODES):
            obs, info = env.reset(seed=seed + eval_episode) # Nova seed para avaliação
            episode_reward = 0
            done = False
            truncated = False
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True) # Usar modo determinístico para avaliação
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        mean_cpu = np.mean(cpu_usages_during_train) if cpu_usages_during_train else 0
        mean_gpu = np.mean(gpu_usages_during_train) if gpu_usages_during_train else 0

        writer.writerow(["base", run_idx + 1, mean_reward, std_reward, duration, mean_cpu, mean_gpu])
        env.close()

print("\nBase experiments completed and logged.")