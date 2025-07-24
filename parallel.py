import os
import random
import numpy as np
import time
import csv
import json # Importamos JSON para salvar os hiperparâmetros

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList

# Assumindo que os arquivos estão na mesma pasta
from custom_env.temp_control_env import TempControlEnv
# Mock para os Callbacks se os arquivos utils não estiverem disponíveis
try:
    from utils.convergence_callback import ConvergenceLogger
    from utils.cpu_gpu_callback import CPU_GPU_Logger
except ImportError:
    print("Aviso: Módulos 'utils' não encontrados. Usando Mocks para Callbacks.")
    class MockCallback:
        def __init__(self, *args, **kwargs): pass
        def get_mean_cpu_usage(self): return 0.0
        def get_mean_gpu_usage(self): return 0.0
    class MockCallbackList:
        def __init__(self, callbacks): pass
        def on_step(self): return True

    ConvergenceLogger = CPU_GPU_Logger = MockCallback
    CallbackList = MockCallbackList


# --- Configurações do Experimento ---
NUM_RUNS = 5
NUM_ENVS = 4  # Número de ambientes em paralelo
TOTAL_TIMESTEPS = 100_000
EVAL_EPISODES = 10
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def make_env_fn(seed):
    def _init():
        env = TempControlEnv()
        env.reset(seed=seed)
        return env
    return _init

def run_parallel_experiment():
    # NOME DO ARQUIVO ATUALIZADO PARA PADRONIZAÇÃO
    output_path = os.path.join(LOG_DIR, "parallel_evaluation.csv")

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        
        # CABEÇALHO PADRONIZADO (incluindo as métricas extras úteis)
        csv_header = [
            "experiment_type", 
            "run", 
            "reward_mean", 
            "reward_std", 
            "training_time", 
            "cpu_mean", 
            "gpu_mean",
            "hyperparameters"
        ]
        writer.writerow(csv_header)

        for run_idx in range(NUM_RUNS):
            seed = 42 + run_idx
            np.random.seed(seed)
            random.seed(seed)

            print(f"\n[RUN {run_idx + 1}] Seed: {seed} - Inicializando {NUM_ENVS} ambientes em paralelo...")
            
            env_fns = [make_env_fn(seed + i) for i in range(NUM_ENVS)]
            vec_env = DummyVecEnv(env_fns)

            # Hiperparâmetros padrão do PPO para referência
            # Isso é útil para registrar quais parâmetros foram usados neste experimento
            model_params = {
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'gamma': 0.99,
                'ent_coef': 0.0
            }

            model = PPO("MlpPolicy", vec_env, verbose=0, seed=seed, **model_params)

            cpu_gpu_callback = CPU_GPU_Logger()
            callback = CallbackList([cpu_gpu_callback])

            print(f"[RUN {run_idx + 1}] Iniciando treinamento...")
            t_train_start = time.time()
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
            train_duration = time.time() - t_train_start
            
            print(f"[RUN {run_idx + 1}] Treinamento concluído. Iniciando avaliação...")
            
            mean_reward, std_reward = evaluate_policy(
                model, 
                model.get_env(), 
                n_eval_episodes=EVAL_EPISODES
            )

            mean_cpu = cpu_gpu_callback.get_mean_cpu_usage()
            mean_gpu = cpu_gpu_callback.get_mean_gpu_usage()

            # LINHA ATUALIZADA COM AS NOVAS COLUNAS
            writer.writerow([
                "parallel_ppo", # Tipo de experimento
                run_idx + 1,
                mean_reward,
                std_reward,
                train_duration,
                mean_cpu,
                mean_gpu,
                json.dumps(model_params) # Salvando os hiperparâmetros usados
            ])
            vec_env.close()

            print(f"[RUN {run_idx + 1}] Finalizado. Média recompensa: {mean_reward:.2f} (±{std_reward:.2f})")

    print(f"\nTodos os experimentos foram concluídos. Resultados salvos em '{output_path}'")


if __name__ == "__main__":
    run_parallel_experiment()