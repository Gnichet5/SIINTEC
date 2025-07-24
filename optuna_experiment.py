import os
import time
import json
import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Assumindo que seu arquivo de ambiente está na mesma pasta ou em um local acessível
try:
    from custom_env.temp_control_env import TempControlEnv
except ImportError:
    print("Erro: Verifique se o arquivo 'temp_control_env.py' está no mesmo diretório.")
    exit()

# --- 1. CONFIGURAÇÕES DO EXPERIMENTO ---

# Configurações para a busca do Optuna
N_TRIALS = 50  # Número de tentativas que o Optuna fará
N_TIMESTEPS_TRIAL = 10_000 # Passos de treino para cada tentativa (menor para ser mais rápido)

# Configurações para a avaliação final (deve ser igual ao seu script base.py)
N_TIMESTEPS_FINAL = 100_000 # Passos de treino para o modelo final
N_EVAL_EPISODES = 10       # Número de episódios para a avaliação final
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def make_env_fn(seed: int):
    """Função auxiliar para criar o ambiente com uma seed."""
    def _init():
        env = TempControlEnv()
        env.reset(seed=seed)
        return env
    return _init


# --- 2. ETAPA DE OTIMIZAÇÃO COM OPTUNA ---

def objective(trial: optuna.Trial) -> float:
    """
    Função objetivo que o Optuna tentará maximizar.
    Ela treina um modelo com um conjunto de hiperparâmetros e retorna sua performance.
    """
    print(f"\nIniciando Trial {trial.number}...")
    
    # Cria um ambiente vetorizado para este trial
    vec_env = DummyVecEnv([make_env_fn(seed=int(time.time())) for _ in range(4)])

    # Definição do espaço de busca de hiperparâmetros
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    # Criação do modelo PPO com os parâmetros do trial
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0  # Silencioso para não poluir o log do Optuna
    )

    # Treina o modelo por um número menor de passos
    model.learn(total_timesteps=N_TIMESTEPS_TRIAL)

    # Avalia o modelo e retorna a recompensa média
    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    
    vec_env.close()
    
    print(f"Trial {trial.number} finalizado com recompensa: {mean_reward:.2f}")
    
    return mean_reward


if __name__ == "__main__":
    # Cria e executa o estudo do Optuna
    # Usamos um banco de dados SQLite para salvar o progresso e poder resumir o estudo
    study = optuna.create_study(
        study_name="ppo_temp_control_optimization",
        storage=f"sqlite:///{LOG_DIR}/optuna_study.db",
        direction="maximize",
        load_if_exists=True # Permite continuar um estudo anterior
    )
    
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*50)
    print("Busca com Optuna concluída!")
    print(f"Melhor recompensa (valor): {study.best_value:.2f}")
    print(f"Melhores hiperparâmetros: {study.best_params}")
    print("="*50 + "\n")

    # --- 3. ETAPA DE AVALIAÇÃO FINAL DO MELHOR MODELO ---

    print("Iniciando treinamento e avaliação final do melhor modelo...")

    # Pega os melhores hiperparâmetros encontrados
    best_params = study.best_params

    # Cria um novo ambiente vetorizado para a avaliação final
    # Usamos uma seed fixa para garantir a reprodutibilidade da avaliação
    eval_vec_env = DummyVecEnv([make_env_fn(seed=42 + i) for i in range(4)])

    # Cria o modelo final com os melhores parâmetros
    final_model = PPO("MlpPolicy", eval_vec_env, verbose=1, **best_params)

    # Treina o modelo final com o número de passos completo (igual ao base.py)
    t_start = time.time()
    final_model.learn(total_timesteps=N_TIMESTEPS_FINAL)
    training_time = time.time() - t_start

    # Avalia o modelo final de forma robusta
    mean_reward, std_reward = evaluate_policy(
        final_model, 
        final_model.get_env(), 
        n_eval_episodes=N_EVAL_EPISODES
    )

    eval_vec_env.close()

    print("\n" + "-"*50)
    print("Avaliação Final Concluída:")
    print(f"  Recompensa Média: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Tempo de Treino: {training_time:.2f}s")
    print("-" * 50)

    # --- 4. SALVAMENTO DOS RESULTADOS COMPARATIVOS ---

    output_path = os.path.join(LOG_DIR, "optuna_final_evaluation.csv")
    
    with open(output_path, "w", newline="") as file:
        import csv
        writer = csv.writer(file)
        
        # Cabeçalho para comparação direta com os resultados do 'base.py'
        header = ["reward_mean", "reward_std", "training_time", "hyperparameters"]
        writer.writerow(header)
        
        # Salva os dados
        writer.writerow([mean_reward, std_reward, training_time, json.dumps(best_params)])

    print(f"\nResultados da avaliação final salvos em: '{output_path}'")