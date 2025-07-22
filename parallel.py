import os
import random
import numpy as np
import gymnasium as gym # Usar gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList # Para múltiplos callbacks
from custom_env.temp_control_env import TempControlEnv
from utils.metrics import start_timer, end_timer, get_cpu_usage, get_gpu_usage
from utils.convergence_callback import ConvergenceLogger # Importa o novo callback
import csv

# --- Configurações do Experimento ---
NUM_RUNS = 3 # Rodar cada configuração 3 vezes para robustez estatística
NUM_ENVS = 4 # Número de ambientes paralelos
TOTAL_TIMESTEPS = 10000 # Total de timesteps para o treinamento
EVAL_EPISODES = 10 # Número de episódios para avaliação final após o treinamento
EVAL_FREQ_CONVERGENCE = 100 # Frequência de avaliação para o gráfico de convergência

# Garante que o diretório de logs exista
os.makedirs("logs", exist_ok=True)

# --- Configuração do CSV Principal ---
csv_header = ["config", "run", "reward_mean", "reward_std", "time", "cpu_mean", "gpu_mean"]

with open("logs/parallel.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

    for run_idx in range(NUM_RUNS):
        # --- Gerenciamento de Seeds para Reprodutibilidade ---
        seed = 42 + run_idx
        np.random.seed(seed)
        random.seed(seed)

        print(f"\n--- Parallel Run {run_idx + 1}/{NUM_RUNS} (Seed: {seed}) ---")

        # --- Função para criar o ambiente (necessário para VecEnv) ---
        def make_env_fn(env_seed): # Renomeado para evitar conflitos
            def _init():
                env = TempControlEnv()
                # A seed para o ambiente individual é definida no reset.
                # Para VecEnv, a seed é passada para o modelo SB3 que a distribui.
                return env
            return _init

        # --- Instanciação do Ambiente Paralelo e Modelo ---
        # Cria uma lista de funções que instanciam o ambiente
        env_fns = [make_env_fn(seed + i) for i in range(NUM_ENVS)]
        vec_env = SubprocVecEnv(env_fns, start_method='fork') # 'fork' pode ser mais eficiente no Linux

        model = PPO("MlpPolicy", vec_env, verbose=0, seed=seed) # Definir seed para o modelo PPO

        # --- Callbacks para Coleta de Métricas ---
        cpu_usages_during_train = []
        gpu_usages_during_train = []

        def cpu_gpu_monitor_callback(_locals, _globals):
            cpu_usages_during_train.append(get_cpu_usage())
            gpu_usages_during_train.append(get_gpu_usage())
            return True

        # Callback para registrar a convergência do treinamento
        convergence_log_dir = os.path.join("logs", f"parallel_run_{run_idx+1}")
        convergence_callback = ConvergenceLogger(
            log_dir=convergence_log_dir, 
            eval_freq=EVAL_FREQ_CONVERGENCE, 
            n_eval_episodes=EVAL_EPISODES
        )

        callback = CallbackList([cpu_gpu_monitor_callback, convergence_callback])

        # --- Treinamento do Agente ---
        train_start_time = start_timer()
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=callback
        )
        duration = end_timer(train_start_time)

        # --- Avaliação Final do Agente ---
        episode_rewards_final = []
        for eval_episode in range(EVAL_EPISODES):
            # Reset da VecEnv. A seed para os ambientes individuais dentro da VecEnv
            # é gerenciada pelo modelo SB3 se ele tiver sido inicializado com uma seed.
            obs, info = vec_env.reset() 
            episode_reward = 0
            terminated = [False] * NUM_ENVS
            truncated = [False] * NUM_ENVS
            # Loop de avaliação enquanto houver ambientes ativos
            while not all(terminated) and not all(truncated): 
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = vec_env.step(action)
                episode_reward += reward.mean() # Soma a média das recompensas de todos os ambientes
            episode_rewards_final.append(episode_reward)
        
        mean_reward_final = np.mean(episode_rewards_final)
        std_reward_final = np.std(episode_rewards_final)

        # Calcula a média do uso de CPU/GPU durante o treinamento
        mean_cpu = np.mean(cpu_usages_during_train) if cpu_usages_during_train else 0
        mean_gpu = np.mean(gpu_usages_during_train) if gpu_usages_during_train else 0

        # --- Escreve os Resultados no CSV ---
        writer.writerow(["parallel", run_idx + 1, mean_reward_final, std_reward_final, duration, mean_cpu, mean_gpu])
        vec_env.close()

print("\nParallel experiments completed and logged.")