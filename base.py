import os
import random
import numpy as np
import gymnasium as gym # Usar gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList # Para múltiplos callbacks
from custom_env.temp_control_env import TempControlEnv
from utils.metrics import start_timer, end_timer, get_cpu_usage, get_gpu_usage
from utils.convergence_callback import ConvergenceLogger # Importa o novo callback
import csv

# --- Configurações do Experimento ---
NUM_RUNS = 3 # Rodar cada configuração 3 vezes para robustez estatística
TOTAL_TIMESTEPS = 10000 # Total de timesteps para o treinamento
EVAL_EPISODES = 10 # Número de episódios para avaliação final após o treinamento
EVAL_FREQ_CONVERGENCE = 100 # Frequência de avaliação para o gráfico de convergência

# Garante que o diretório de logs exista
os.makedirs("logs", exist_ok=True)

# --- Configuração do CSV Principal ---
csv_header = ["config", "run", "reward_mean", "reward_std", "time", "cpu_mean", "gpu_mean"]

with open("logs/base.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

    for run_idx in range(NUM_RUNS):
        # --- Gerenciamento de Seeds para Reprodutibilidade ---
        seed = 42 + run_idx
        np.random.seed(seed)
        random.seed(seed)
        # Stable Baselines3 lida com a seed do ambiente se ela for passada para o modelo

        print(f"\n--- Base Run {run_idx + 1}/{NUM_RUNS} (Seed: {seed}) ---")

        # --- Instanciação do Ambiente e Modelo ---
        env = TempControlEnv()
        # O reset do ambiente é feito implicitamente por SB3, mas garantimos a seed no modelo.
        # env.reset(seed=seed) # Não é necessário chamar aqui se a seed for passada para o modelo PPO

        model = PPO("MlpPolicy", env, verbose=0, seed=seed) # Definir seed para o modelo PPO

        # --- Callbacks para Coleta de Métricas ---
        cpu_usages_during_train = []
        gpu_usages_during_train = []

        # Callback para coletar uso de CPU/GPU
        # Este callback é um lambda simples que é chamado a cada passo de treinamento
        # Ele não pausa o treinamento, apenas coleta um snapshot do uso
        def cpu_gpu_monitor_callback(_locals, _globals):
            cpu_usages_during_train.append(get_cpu_usage())
            gpu_usages_during_train.append(get_gpu_usage())
            return True # Retorna True para continuar o treinamento

        # Callback para registrar a convergência do treinamento
        convergence_log_dir = os.path.join("logs", f"base_run_{run_idx+1}")
        convergence_callback = ConvergenceLogger(
            log_dir=convergence_log_dir, 
            eval_freq=EVAL_FREQ_CONVERGENCE, 
            n_eval_episodes=EVAL_EPISODES # Usar o mesmo número de episódios de avaliação para consistência
        )

        # Junta os callbacks em uma lista
        callback = CallbackList([cpu_gpu_monitor_callback, convergence_callback])

        # --- Treinamento do Agente ---
        train_start_time = start_timer()
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=callback # Passar a lista de callbacks
        )
        duration = end_timer(train_start_time)

        # --- Avaliação Final do Agente ---
        episode_rewards_final = []
        for eval_episode in range(EVAL_EPISODES):
            # Reinicia o ambiente para cada episódio de avaliação com uma seed diferente
            obs, info = env.reset(seed=seed + eval_episode + 1000) 
            episode_reward = 0
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=True) # Usar modo determinístico para avaliação
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
            episode_rewards_final.append(episode_reward)
        
        mean_reward_final = np.mean(episode_rewards_final)
        std_reward_final = np.std(episode_rewards_final)

        # Calcula a média do uso de CPU/GPU durante o treinamento
        mean_cpu = np.mean(cpu_usages_during_train) if cpu_usages_during_train else 0
        mean_gpu = np.mean(gpu_usages_during_train) if gpu_usages_during_train else 0

        # --- Escreve os Resultados no CSV ---
        writer.writerow(["base", run_idx + 1, mean_reward_final, std_reward_final, duration, mean_cpu, mean_gpu])
        env.close()

print("\nBase experiments completed and logged.")