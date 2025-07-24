import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class TempControlEnv(gym.Env):
    """
    Ambiente de controle de temperatura simulando um forno industrial com inércia térmica.
    Versão estendida com mais complexidade para experimentos de otimização.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(TempControlEnv, self).__init__()
        # Espaço de ação: valor contínuo (-1 a 1), simula controle de potência
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Espaço de observação expandido:
        # [T1, T2, T3, erro, d_erro, energia, tempo_desde_pico]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.setpoint = 100.0
        self.temps = np.random.uniform(20, 30, size=3)  # Temperaturas em 3 sensores
        self.last_error = self.setpoint - np.mean(self.temps)
        self.energy = 100.0  # energia disponível no sistema
        self.steps_since_last_peak = 0

        self.max_episode_steps = 500
        self.current_step = 0

        # Atraso da ação (inércia térmica)
        self.action_delay = deque([0.0]*5, maxlen=5)

    def step(self, action):
        self.current_step += 1
        self.steps_since_last_peak += 1

        power = float(np.clip(action, -1, 1))
        self.action_delay.append(power)
        delayed_power = self.action_delay[0]

        noise = np.random.normal(0, 0.3, size=3)  # ruído individual em cada sensor
        deltas = 0.1 * delayed_power - 0.05 * (self.temps - self.setpoint) + noise
        self.temps += deltas

        # Perturbação aleatória
        if np.random.rand() < 0.01:
            self.temps -= np.random.uniform(3.0, 6.0)
            self.steps_since_last_peak = 0

        avg_temp = np.mean(self.temps)
        error = self.setpoint - avg_temp
        d_error = error - self.last_error
        self.last_error = error

        # Atualização de energia (decresce com uso)
        self.energy -= abs(delayed_power) * 0.5
        self.energy = max(self.energy, 0.0)

        obs = np.array([
            self.temps[0] + np.random.normal(0, 0.2),
            self.temps[1] + np.random.normal(0, 0.2),
            self.temps[2] + np.random.normal(0, 0.2),
            error,
            d_error,
            self.energy,
            float(self.steps_since_last_peak)
        ], dtype=np.float32)

        # Recompensa penaliza erro e bonifica estabilidade prolongada
        reward = -abs(error)
        if abs(error) < 1.0:
            reward += 1.0  # bônus por manter temperatura perto do alvo
        if self.steps_since_last_peak > 50:
            reward += 0.5  # bônus por estabilidade prolongada

        terminated = abs(error) < 0.3 and self.steps_since_last_peak > 30
        truncated = self.current_step >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temps = np.random.uniform(20, 30, size=3)
        self.last_error = self.setpoint - np.mean(self.temps)
        self.current_step = 0
        self.energy = 100.0
        self.steps_since_last_peak = 0
        self.action_delay = deque([0.0]*5, maxlen=5)

        obs = np.array([
            self.temps[0] + np.random.normal(0, 0.2),
            self.temps[1] + np.random.normal(0, 0.2),
            self.temps[2] + np.random.normal(0, 0.2),
            self.last_error,
            0.0,
            self.energy,
            0.0
        ], dtype=np.float32)
        return obs, {}

    def render(self):
        pass

    def close(self):
        pass
