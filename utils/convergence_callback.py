import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv # Para type hinting e checagem

class ConvergenceLogger(BaseCallback):
    """
    Callback personalizado para logar a recompensa média durante o treinamento
    e salvar em um arquivo CSV para análise de convergência.
    """
    def __init__(self, log_dir: str, eval_freq: int, n_eval_episodes: int, verbose: int = 0):
        super(ConvergenceLogger, self).__init__(verbose)
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Garante que o diretório de log exista
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "convergence_data.csv")
        
        # Escreve o cabeçalho do CSV
        with open(self.log_path, "w", newline="") as f:
            f.write("timesteps,mean_reward,std_reward\n")

    def _on_step(self) -> bool:
        """
        Chamado a cada passo de treinamento.
        """
        # Avalia o agente a cada 'eval_freq' timesteps
        if self.n_calls % self.eval_freq == 0:
            if self.model is None or self.model.env is None:
                if self.verbose > 0:
                    print("Modelo ou ambiente não disponível para avaliação no callback.")
                return True

            # Lista para armazenar recompensas de avaliação de múltiplos episódios
            eval_rewards = []

            # Realiza n_eval_episodes para obter uma média mais robusta
            for _ in range(self.n_eval_episodes):
                episode_reward = 0
                
                # O reset da VecEnv retorna apenas observações
                # Para ambientes únicos, retorna (obs, info)
                if isinstance(self.model.env, VecEnv):
                    obs = self.model.env.reset()
                else:
                    obs, info = self.model.env.reset()
                
                # Para VecEnv, 'dones' é um array de booleanos. Para ambiente único, é um booleano.
                dones = [False] * self.model.env.num_envs if isinstance(self.model.env, VecEnv) else False
                steps = 0
                max_steps_eval = 200 # Limite de passos para a avaliação do callback para evitar loops infinitos

                # Loop para um único episódio de avaliação (ou simulação de um)
                while not (isinstance(self.model.env, VecEnv) and all(dones)) and \
                      not (not isinstance(self.model.env, VecEnv) and dones) and \
                      steps < max_steps_eval:
                    
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # VecEnv.step() retorna (obs, reward, dones, infos) - 4 valores
                    # Ambiente único .step() retorna (obs, reward, terminated, truncated, info) - 5 valores
                    if isinstance(self.model.env, VecEnv):
                        obs, reward, dones, info = self.model.env.step(action)
                    else:
                        obs, reward, terminated, truncated, info = self.model.env.step(action)
                        dones = terminated or truncated # Para consistência com a lógica de parada

                    # Acumula a recompensa: se for VecEnv, soma a média das recompensas de todos os ambientes
                    episode_reward += np.mean(reward) if isinstance(reward, np.ndarray) else reward
                    
                    steps += 1

                eval_rewards.append(episode_reward)

            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)

            # Loga os dados no CSV
            with open(self.log_path, "a", newline="") as f:
                f.write(f"{self.num_timesteps},{mean_reward},{std_reward}\n")

            if self.verbose > 0:
                print(f"Timesteps: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")

        return True # Retorna True para continuar o treinamento