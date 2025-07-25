import os
import time
import pandas as pd
import json
import psutil
import pynvml
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_env.temp_control_env import TempControlEnv

def monitor_resources(q, process_id):
    p = psutil.Process(process_id)
    gpu_available, handle = False, None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_available = True
    except pynvml.NVMLError: pass
    cpu_usage, gpu_usage = [], []
    while True:
        try:
            cpu_usage.append(p.cpu_percent(interval=1.0))
            if gpu_available: gpu_usage.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            if not q.empty() and q.get() == 'stop': break
        except (psutil.NoSuchProcess, pynvml.NVMLError): break
    if gpu_available: pynvml.nvmlShutdown()
    cpu_mean = sum(cpu_usage)/len(cpu_usage) if cpu_usage else 0
    gpu_mean = sum(gpu_usage)/len(gpu_usage) if gpu_usage else 0
    q.put({'cpu_mean': cpu_mean, 'gpu_mean': gpu_mean})

def make_env(env_id, rank, seed=0, config={}):
    """Função para criar um ambiente para o SubprocVecEnv."""
    def _init():
        env = Monitor(TempControlEnv(config=config))
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    optuna_results_file = 'optuna_final_evaluation.csv'
    try:
        df_optuna = pd.read_csv(optuna_results_file)
        hyperparams_str = df_optuna.iloc[0]['hyperparameters']
        HYPERPARAMS = json.loads(hyperparams_str)
        print(f"Hiperparâmetros otimizados carregados: {HYPERPARAMS}")
    except FileNotFoundError:
        print(f"AVISO: '{optuna_results_file}' não encontrado. Usando hiperparâmetros padrão.")
        HYPERPARAMS = {}

    N_RUNS = 5
    NUM_CPU = 4
        "enable_matrix_calcs": True,
        "enable_high_precision": True
    }
    
    results = []
    output_file = 'parallel_optuna_evaluation.csv'

    for i in range(N_RUNS):
        run_seed = 42 + i
        print(f"--- Iniciando Execução PARALELA OTIMIZADA N° {i+1}/{N_RUNS} ---")
        
        main_process = psutil.Process(os.getpid())
        q = Queue()
        monitor_proc = Process(target=monitor_resources, args=(q, main_process.pid))
        monitor_proc.start()
        
        start_time = time.time()
        
        vec_env = SubprocVecEnv([make_env('temp-control', j, seed=run_seed, config=CONFIG) for j in range(NUM_CPU)])
        
        model = PPO("MlpPolicy", vec_env, verbose=0, seed=run_seed, **HYPERPARAMS)
        
        model.learn(total_timesteps=20000)
        
        training_time = time.time() - start_time
        
        eval_env = Monitor(TempControlEnv(config=CONFIG))
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        
        q.put('stop')
        resource_usage = q.get()
        monitor_proc.join()
        
        vec_env.close()
        eval_env.close()

        run_data = {
            'config': 'parallel_otimizada',
            'run': i + 1,
            'reward_mean': mean_reward,
            'reward_std': std_reward,
            'time': training_time,
            'cpu_mean': resource_usage['cpu_mean'],
            'gpu_mean': resource_usage['gpu_mean'],
            'hyperparameters': json.dumps(HYPERPARAMS)
        }
        results.append(run_data)
        print(f"Resultado da execução {i+1}: {run_data}")

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Resultados da avaliação paralela otimizada salvos em {output_file}")