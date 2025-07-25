import os
import time
import pandas as pd
import psutil
import pynvml
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from custom_env.temp_control_env import TempControlEnv

def monitor_resources(q, process_id):
    p = psutil.Process(process_id)
    gpu_available = False
    handle = None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_available = True
    except pynvml.NVMLError:
        pass 

    cpu_usage, gpu_usage = [], []
    
    while True:
        try:
            cpu_usage.append(p.cpu_percent(interval=1.0))
            if gpu_available:
                gpu_usage.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            if not q.empty() and q.get() == 'stop':
                break
        except (psutil.NoSuchProcess, pynvml.NVMLError):
            break 

    if gpu_available:
        pynvml.nvmlShutdown()

    cpu_mean = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    gpu_mean = sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0
    q.put({'cpu_mean': cpu_mean, 'gpu_mean': gpu_mean})

if __name__ == "__main__":
    N_RUNS = 5
    
    CONFIG = {
        "enable_matrix_calcs": True,
        "enable_high_precision": True
    }
    
    results = []
    output_file = 'base_heavy_evaluation.csv' 

    for i in range(N_RUNS):
        print(f"--- Iniciando Execução BASE (PESADO) N° {i+1}/{N_RUNS} ---")
        
        main_process = psutil.Process(os.getpid())
        q = Queue()
        
        monitor_proc = Process(target=monitor_resources, args=(q, main_process.pid))
        monitor_proc.start()
        
        start_time = time.time()
        
        env = Monitor(TempControlEnv(config=CONFIG))
        
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=20000)
        
        training_time = time.time() - start_time
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        
        q.put('stop')
        resource_usage = q.get()
        monitor_proc.join()
        env.close()

        run_data = {
            'config': 'base_pesado',
            'run': i + 1,
            'reward_mean': mean_reward,
            'reward_std': std_reward,
            'time': training_time,
            'cpu_mean': resource_usage['cpu_mean'],
            'gpu_mean': resource_usage['gpu_mean']
        }
        results.append(run_data)
        print(f"Resultado da execução {i+1}: {run_data}")

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Resultados da avaliação base (pesado) salvos em {output_file}")