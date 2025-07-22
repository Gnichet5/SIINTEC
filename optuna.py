import optuna
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
NUM_TRIALS_OPTUNA = 50 # Número de trials para a otimização Optuna (aumentado para rigor)
NUM_ENVS = 4 # Número de ambientes paralelos para os trials e treinamento final
TOTAL_TIMESTEPS_OPTUNA_TRIAL = 2000 # Menos timesteps para cada trial Optuna (para agilizar a otimização)
TOTAL_TIMESTEPS_FINAL_TRAIN = 10000 # Timesteps para o treinamento final (consistente com base/parallel)
EVAL_EPISODES = 10 # Número de episódios para avaliação final após o treinamento
EVAL_FREQ_CONVERGENCE = 100 # Frequência de avaliação para o gráfico de convergência

# Garante que o diretório de logs exista
os.makedirs("logs", exist_ok=True)

# --- Configuração do CSV Principal ---
csv_header = ["config", "run", "reward_mean", "reward_std", "time", "cpu_mean", "gpu_mean"]

# Abre o arquivo em modo 'w' (write) para sobrescrever ou criar e 'a' (append) para adicionar
with open("logs/optuna.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_header) # Escreve o cabeçalho apenas uma vez

for run_idx in range(NUM_RUNS):
    # --- Gerenciamento de Seeds para Reprodutibilidade ---
    seed = 42 + run_idx
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n--- Optuna Run {run_idx + 1}/{NUM_RUNS} (Seed: {seed}) ---")

    # --- Função para criar o ambiente (necessário para VecEnv) ---
    def make_env_fn(env_seed):
        def _init():
            env = TempControlEnv()
            return env
        return _init

    # --- Função Objetivo para Optuna ---
    def objective(trial):
        # Sugere hiperparâmetros
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
        n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)
        
        # Cria ambientes paralelos para o trial (com seeds baseadas no trial para unicidade)
        env_fns_trial = [make_env_fn(seed + i + trial.number * 100) for i in range(NUM_ENVS)]
        vec_env_trial = SubprocVecEnv(env_fns_trial, start_method='fork')
        
        # Instancia o modelo PPO para o trial
        model_trial = PPO("MlpPolicy", vec_env_trial, learning_rate=lr, n_steps=n_steps, verbose=0, seed=seed)
        
        # Treina o modelo para um número reduzido de timesteps para agilizar a otimização
        model_trial.learn(total_timesteps=TOTAL_TIMESTEPS_OPTUNA_TRIAL)

        # --- Avaliação rápida para o trial (para o valor objetivo) ---
        obs, info = vec_env_trial.reset() # Reset da VecEnv
        reward_sum_eval = 0
        terminated = [False] * NUM_ENVS
        truncated = [False] * NUM_ENVS
        # Avalia por um número fixo de passos ou até que todos os envs terminem uma vez
        for step_eval in range(100): # Avaliação rápida por 100 passos
            action, _ = model_trial.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vec_env_trial.step(action)
            reward_sum_eval += reward.mean() # Soma a média das recompensas de todos os ambientes
            
            if all(terminated) or all(truncated): # Se todos os ambientes terminaram, saia
                break
        
        vec_env_trial.close() # Fecha a VecEnv do trial
        
        # Reportar a recompensa acumulada do trial para o Optuna para o pruning
        trial.report(reward_sum_eval, step=TOTAL_TIMESTEPS_OPTUNA_TRIAL)
        
        # Lógica de pruning: interrompe trials sem perspectivas
        if trial.should_prune():
            print(f"Trial {trial.number} podado no timestep {TOTAL_TIMESTEPS_OPTUNA_TRIAL}. Recompensa: {reward_sum_eval:.2f}")
            raise optuna.exceptions.TrialPruned()

        return reward_sum_eval

    # --- Criação e Otimização do Estudo Optuna ---
    # Usa MedianPruner para poda de trials
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=NUM_TRIALS_OPTUNA)

    print(f"\nMelhores parâmetros encontrados para Run {run_idx + 1}: {study.best_params}")

    # --- Treinamento Final com os Melhores Parâmetros ---
    params = study.best_params
    
    # Cria uma nova set de seeds para o treinamento final com os melhores parâmetros
    env_fns_final = [make_env_fn(seed + i + 2000) for i in range(NUM_ENVS)] # seeds distintas
    vec_env_final = SubprocVecEnv(env_fns_final, start_method='fork')
    
    model_final = PPO("MlpPolicy", vec_env_final, learning_rate=params["lr"], n_steps=params["n_steps"], verbose=0, seed=seed)

    # --- Callbacks para Coleta de Métricas do Treinamento Final ---
    cpu_usages_during_train = []
    gpu_usages_during_train = []

    def cpu_gpu_monitor_callback_final(_locals, _globals):
        cpu_usages_during_train.append(get_cpu_usage())
        gpu_usages_during_train.append(get_gpu_usage())
        return True

    # Callback para registrar a convergência do treinamento final
    convergence_log_dir_final = os.path.join("logs", f"optuna_run_{run_idx+1}")
    convergence_callback_final = ConvergenceLogger(
        log_dir=convergence_log_dir_final, 
        eval_freq=EVAL_FREQ_CONVERGENCE, 
        n_eval_episodes=EVAL_EPISODES
    )

    callback_final = CallbackList([cpu_gpu_monitor_callback_final, convergence_callback_final])

    start_time = start_timer()
    model_final.learn(
        total_timesteps=TOTAL_TIMESTEPS_FINAL_TRAIN, 
        callback=callback_final
    )
    duration = end_timer(start_time)

    # --- Avaliação Final do Agente Otimizado ---
    episode_rewards_final = []
    for eval_episode in range(EVAL_EPISODES):
        obs, info = vec_env_final.reset()
        episode_reward = 0
        terminated = [False] * NUM_ENVS
        truncated = [False] * NUM_ENVS
        while not all(terminated) and not all(truncated):
            action, _ = model_final.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vec_env_final.step(action)
            episode_reward += reward.mean()
        episode_rewards_final.append(episode_reward)

    mean_reward_final = np.mean(episode_rewards_final)
    std_reward_final = np.std(episode_rewards_final)

    # Calcula a média do uso de CPU/GPU durante o treinamento final
    mean_cpu = np.mean(cpu_usages_during_train) if cpu_usages_during_train else 0
    mean_gpu = np.mean(gpu_usages_during_train) if gpu_usages_during_train else 0

    # --- Escreve os Resultados no CSV (em modo append) ---
    with open("logs/optuna.csv", "a", newline="") as file: 
        writer = csv.writer(file)
        writer.writerow(["optuna", run_idx + 1, mean_reward_final, std_reward_final, duration, mean_cpu, mean_gpu])
    
    vec_env_final.close()

print("\nOptuna experiments completed and logged.")