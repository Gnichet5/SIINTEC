import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TempControlEnv(gym.Env):
    """
    Ambiente de controle de temperatura simulando um forno industrial com inércia térmica.
    Compatível com a API Gymnasium.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30} # Metadados para o Gym/Gymnasium

    def __init__(self):
        super(TempControlEnv, self).__init__()
        # Espaço de ação: Um valor contínuo para o controle de potência (-1.0 a 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Espaço de observação: [Temperatura Atual, Erro Instantâneo, Derivada do Erro]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.setpoint = 100.0 # Temperatura alvo
        self.temp = np.random.uniform(20, 30) # Temperatura inicial aleatória
        self.last_error = self.setpoint - self.temp # Erro inicial

        # Limite de passos por episódio para lidar com 'truncated'
        self.max_episode_steps = 200 
        self.current_step = 0 # Contador de passos dentro do episódio

    def step(self, action):
        """
        Executa um passo no ambiente.
        Retorna: observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        power = float(np.clip(action, -1, 1)) # Garante que a ação esteja no range
        noise = np.random.normal(0, 0.3) # Ruído estocástico no processo
        
        # Dinâmica do forno: delta de temperatura baseado em potência, erro e ruído
        delta = 0.1 * power - 0.05 * (self.temp - self.setpoint) + noise
        self.temp += delta

        error = self.setpoint - self.temp
        d_error = error - self.last_error
        self.last_error = error

        obs = np.array([self.temp, error, d_error], dtype=np.float32)
        reward = -abs(error)  # Recompensa: penaliza o erro absoluto

        # Condições de término do episódio
        terminated = abs(error) < 0.5 # True se o agente atingiu a temperatura alvo (erro pequeno)
        truncated = self.current_step >= self.max_episode_steps # True se o limite de passos foi atingido

        info = {} # Informações adicionais (pode ser vazio por enquanto)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente para um novo episódio.
        Retorna: observation, info
        """
        super().reset(seed=seed) # Lida com a seed para reprodutibilidade
        
        # Reinicia a temperatura e o erro
        self.temp = np.random.uniform(20, 30)
        self.last_error = self.setpoint - self.temp
        self.current_step = 0 # Reinicia o contador de passos

        obs = np.array([self.temp, self.last_error, 0.0], dtype=np.float32)
        info = {} # Informações adicionais na reinicialização (pode ser vazio)

        return obs, info

    def render(self):
        # Implementação opcional para visualização do ambiente
        pass

    def close(self):
        # Implementação opcional para fechar recursos
        pass