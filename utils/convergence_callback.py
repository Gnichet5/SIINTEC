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
                # Reset do ambiente: para VecEnv, apenas um reset() global
                # Para ambiente único, env.reset() retorna (obs, info)
                obs, info = self.model.env.reset() if not isinstance(self.model.env, VecEnv) else self.model.env.reset()
                
                terminated = [False] * self.model.env.num_envs if isinstance(self.model.env, VecEnv) else False
                truncated = [False] * self.model.env.num_envs if isinstance(self.model.env, VecEnv) else False
                
                # Loop para um único episódio de avaliação
                while not (isinstance(self.model.env, VecEnv) and all(terminated) and all(truncated)) and \
                      not (not isinstance(self.model.env, VecEnv) and terminated and truncated): # Condição de parada flexível
                    
                    action, _ = self.model.predict(obs, deterministic=True)
                    next_obs, reward, next_terminated, next_truncated, info = self.model.env.step(action)
                    
                    # Acumula a recompensa: se for VecEnv, soma a média das recompensas de todos os ambientes
                    episode_reward += np.mean(reward) if isinstance(reward, np.ndarray) else reward
                    
                    obs = next_obs
                    terminated = next_terminated
                    truncated = next_truncated
                    
                    # Se um ambiente único terminou, saia do loop do episódio
                    if not isinstance(self.model.env, VecEnv) and (terminated or truncated):
                        break

                eval_rewards.append(episode_reward)

            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)

            # Loga os dados no CSV
            with open(self.log_path, "a", newline="") as f:
                f.write(f"{self.num_timesteps},{mean_reward},{std_reward}\n")

            if self.verbose > 0:
                print(f"Timesteps: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")

        return True # Retorna True para continuar o treinamento