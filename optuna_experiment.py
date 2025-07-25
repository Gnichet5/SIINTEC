import os
import time
import json
import optuna
import numpy as np
import pandas as pd
import psutil
import pynvml
from typing import Dict, Any, Optional
from threading import Thread, Event
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

try:
    from custom_env.temp_control_env import TempControlEnv
except ImportError as e:
    raise ImportError("Erro: Arquivo 'temp_control_env.py' não encontrado na pasta 'custom_env'") from e

# ======================================================
# CONFIGURAÇÕES GLOBAIS (sem alterações)
# ======================================================
class ExperimentConfig:
    ENV_CONFIG = {"enable_matrix_calcs": True, "enable_high_precision": True}
    N_TRIALS = 50
    N_TIMESTEPS_TRIAL = 15_000
    N_TIMESTEPS_FINAL = 20_000
    N_EVAL_EPISODES = 10
    N_ENVS = 4
    LOG_DIR = "optuna_logs"
    STUDY_NAME = "temp_control_optimization"
    SQL_DB_URL = f"sqlite:///{LOG_DIR}/optuna_study.db"
    def __init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)

# ======================================================
# CALLBACK E FÁBRICA DE AMBIENTES 
# ======================================================
class CustomPruningCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, eval_interval: int = 1000):
        super().__init__()
        self.trial, self.eval_interval, self.best_mean_reward = trial, eval_interval, -np.inf
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_interval == 0 and len(self.model.ep_info_buffer) > 0:
            current_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            self.best_mean_reward = max(self.best_mean_reward, current_reward)
            self.trial.report(self.best_mean_reward, self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return True

def make_env(seed: int, config: Dict[str, Any]) -> callable:
    def _init():
        env = Monitor(TempControlEnv(config=config))
        env.reset(seed=seed)
        return env
    return _init

def objective(trial: optuna.Trial, config: ExperimentConfig) -> float:
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [256, 512, 1024, 2048]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 0.1, log=True),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'n_epochs': trial.suggest_int('n_epochs', 1, 10),
    }
    vec_env = DummyVecEnv([make_env(seed=int(time.time())+i, config=config.ENV_CONFIG) for i in range(config.N_ENVS)])
    model = PPO("MlpPolicy", vec_env, **hyperparams, verbose=0, tensorboard_log=os.path.join(config.LOG_DIR, "tensorboard"))
    callback = CustomPruningCallback(trial)
    try:
        model.learn(total_timesteps=config.N_TIMESTEPS_TRIAL, callback=callback, tb_log_name=f"trial_{trial.number}")
    except optuna.exceptions.TrialPruned:
        vec_env.close()
        raise
    mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=config.N_EVAL_EPISODES)
    vec_env.close()
    print(f"✅ Trial {trial.number} concluído | Recompensa: {mean_reward:.2f}")
    return float(mean_reward)

# ======================================================
# ---  LÓGICA DE MONITORAMENTO ATUALIZADA ---
# ======================================================
def monitor_worker(monitor, stop_event):
    """Função que roda em uma thread para coletar dados periodicamente."""
    while not stop_event.is_set():
        monitor.update()
        time.sleep(1.0) # Intervalo de coleta

def evaluate_final_model(study: optuna.Study, config: ExperimentConfig) -> Dict[str, Any]:
    print("\n" + "="*50 + "\nAvaliando melhor modelo...")
    
    eval_env = DummyVecEnv([make_env(seed=42+i, config=config.ENV_CONFIG) for i in range(config.N_ENVS)])
    monitor = ResourceMonitor(os.getpid())
    stop_event = Event()
    
    with monitor: 
        monitor_thread = Thread(target=monitor_worker, args=(monitor, stop_event))
        monitor_thread.start()

        start_time = time.time()
        model = PPO("MlpPolicy", eval_env, **study.best_params, verbose=1, tensorboard_log=os.path.join(config.LOG_DIR, "tensorboard"))
        model.learn(total_timesteps=config.N_TIMESTEPS_FINAL, tb_log_name="final_model")
        training_time = time.time() - start_time
        stop_event.set()
        monitor_thread.join()

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=config.N_EVAL_EPISODES * 2)
    eval_env.close()
    resource_stats = monitor.get_stats()

    results = {
        'config': 'optuna_heavy',
        'reward_mean': float(mean_reward), 'reward_std': float(std_reward),
        'time': training_time,
        'cpu_mean': resource_stats['cpu_mean'], 'gpu_mean': resource_stats['gpu_mean'],
        'hyperparameters': json.dumps(study.best_params),
    }
    
    print("\n" + "="*50 + "\nResultados Finais:")
    print(f"Recompensa: {results['reward_mean']:.2f} ± {results['reward_std']:.2f}")
    print(f"Tempo: {results['time']:.2f}s | CPU: {results['cpu_mean']:.1f}% | GPU: {results['gpu_mean']:.1f}%")
    print("="*50)
    
    return results

# ======================================================
# --- CLASSE ResourceMonitor  ---
# ======================================================
class ResourceMonitor:
    def __init__(self, process_id: int):
        self.process = psutil.Process(process_id)
        self.cpu_usage, self.gpu_usage = [], []
        self.gpu_available, self.handle = False, None
    def __enter__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
            print("Monitoramento de GPU ativado.")
        except pynvml.NVMLError:
            print("Monitoramento de GPU não disponível.")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu_available: pynvml.nvmlShutdown()
    def update(self):
        try:
            self.cpu_usage.append(self.process.cpu_percent())
            if self.gpu_available:
                self.gpu_usage.append(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
        except (psutil.NoSuchProcess, pynvml.NVMLError): pass
    def get_stats(self) -> Dict[str, float]:
        return {
            'cpu_mean': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'gpu_mean': np.mean(self.gpu_usage) if self.gpu_usage else 0,
        }

# ======================================================
# FUNÇÃO PRINCIPAL 
# ======================================================
def main():
    config = ExperimentConfig()
    set_random_seed(42)
    study = optuna.create_study(
        study_name=config.STUDY_NAME, storage=config.SQL_DB_URL,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        direction="maximize", load_if_exists=True
    )
    print(f"\n🚀 Iniciando otimização com {config.N_TRIALS} trials")
    study.optimize(lambda trial: objective(trial, config), n_trials=config.N_TRIALS, show_progress_bar=True)
    
    results = evaluate_final_model(study, config)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.LOG_DIR, f"results_{timestamp}.csv")
    pd.DataFrame([results]).to_csv(results_file, index=False)
    
    print(f"\n📊 Resultados salvos em: {results_file}")
    print(f"⭐ Melhor recompensa do estudo: {study.best_value:.2f}")

if __name__ == "__main__":
    main()