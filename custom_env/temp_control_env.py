import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from collections import deque

class TempControlEnv(gym.Env):
    """
    Um ambiente customizado do Gymnasium para simular o controle de temperatura
    de um sistema com inércia térmica (como um forno industrial), com gargalos
    computacionais controlados para benchmarking.

    - Física térmica mais complexa
    - Histórico de observações
    - Múltiplos distúrbios
    - Cálculos matriciais opcionais
    - Simulação de alta precisão opcional
    - Gargalos computacionais controláveis
    """

    metadata = {'render_modes': ['human', 'benchmark'], 'render_fps': 30}

    def __init__(self, config=None):
        """
        Inicializa o ambiente de controle de temperatura com opções para gargalos computacionais.

        Args:
            config (dict, optional): Configurações adicionais:
                - "enable_history": (bool) Ativa histórico de observações (aumenta dimensão)
                - "enable_matrix_calcs": (bool) Ativa cálculos matriciais pesados
                - "enable_high_precision": (bool) Ativa simulação de alta precisão (mais lenta)
                - "enable_multiple_disturbances": (bool) Ativa múltiplos distúrbios
                - "compute_sleep": (float) Tempo de sleep artificial para simular carga
        """
        super().__init__()

        default_config = {
            "setpoint": 100.0,
            "initial_temp_range": (25.0, 40.0),
            "ambient_temp": 25.0,
            "max_temp": 150.0,
            "min_temp": 20.0,
            "max_steps": 500,
            "time_step": 1.0,
            "noise_std_dev": 0.1,
            "disturbance_magnitude": -20.0,
            "disturbance_step_range": [(200, 300)],
            "action_penalty_weight": 0.05,
            "precision_bonus_threshold": 1.0,
            "precision_bonus": 1.0,
            "enable_history": False,
            "history_length": 5,
            "enable_matrix_calcs": False,
            "matrix_dim": 100,
            "enable_high_precision": False,
            "enable_multiple_disturbances": False,
            "compute_sleep": 0.0,
            "thermal_mass": 10.0,
            "heat_transfer_coeff": 0.02,
            "heating_efficiency": 0.5,
            "nonlinearity_factor": 0.001
        }
        
        if config:
            default_config.update(config)
        self.config = default_config
        if self.config["enable_multiple_disturbances"]:
            self.config["disturbance_step_range"] = [
                (100, 150), (200, 250), (300, 350)
            ]

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        

        obs_dim = 2  
        if self.config["enable_history"]:
            obs_dim += 2 * self.config["history_length"]  
            
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)


        self.current_temp = 0
        self.previous_error = 0
        self.current_step = 0
        self.last_action = 0.0
        self._disturbance_steps = [] 
        self._history = deque(maxlen=self.config["history_length"])
        
        # Para cálculos matriciais
        self._matrix = None
        if self.config["enable_matrix_calcs"]:
            self._matrix = np.random.randn(self.config["matrix_dim"], self.config["matrix_dim"])

        self._compute_time = 0
        self._step_times = []

    def _complex_physics_update(self, action: float):
        """Atualiza a temperatura com física mais complexa e computacionalmente intensiva."""
        if self.config["compute_sleep"] > 0:
            time.sleep(self.config["compute_sleep"])
            
        start_time = time.time()
        
        heating_effect = action * self.config["heating_efficiency"]
        if self.config["enable_high_precision"]:
            heating_effect *= 1 + self.config["nonlinearity_factor"] * self.current_temp

        temp_diff = self.current_temp - self.config["ambient_temp"]
        cooling_effect = (self.config["heat_transfer_coeff"] * temp_diff + 
                          self.config["nonlinearity_factor"] * temp_diff**2)

        delta_temp = (heating_effect - cooling_effect) / self.config["thermal_mass"]
        
        if self.config["enable_high_precision"]:
            k1 = delta_temp
            temp_k1 = self.current_temp + 0.5 * self.config["time_step"] * k1
            k2 = (heating_effect - (self.config["heat_transfer_coeff"] * (temp_k1 - self.config["ambient_temp"]))) / self.config["thermal_mass"]
            temp_k2 = self.current_temp + 0.5 * self.config["time_step"] * k2
            k3 = (heating_effect - (self.config["heat_transfer_coeff"] * (temp_k2 - self.config["ambient_temp"]))) / self.config["thermal_mass"]
            temp_k3 = self.current_temp + self.config["time_step"] * k3
            k4 = (heating_effect - (self.config["heat_transfer_coeff"] * (temp_k3 - self.config["ambient_temp"]))) / self.config["thermal_mass"]
            
            self.current_temp += (self.config["time_step"] / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            self.current_temp += delta_temp * self.config["time_step"]

        noise = self.np_random.normal(0, self.config["noise_std_dev"])
        self.current_temp += noise

        self.current_temp = np.clip(
            self.current_temp, self.config["min_temp"], self.config["max_temp"]
        )

        if self.config["enable_matrix_calcs"]:
            for _ in range(3): 
                self._matrix = np.dot(self._matrix, self._matrix.T)
                self._matrix = np.linalg.svd(self._matrix)[0]
        
        self._compute_time += time.time() - start_time

    def step(self, action: np.ndarray):
        """Executa um passo no ambiente com gargalos computacionais controlados."""
        action_value = action[0]
        
        self._complex_physics_update(action_value)
        if self.current_step in self._disturbance_steps:
            self.current_temp += self.config["disturbance_magnitude"]
            if self.config["enable_high_precision"]:
                self.current_temp += 0.1 * self.np_random.normal(0, 1)

        error = self.config["setpoint"] - self.current_temp
        derivative_of_error = (error - self.previous_error) / self.config["time_step"]
        
        # Atualiza o histórico
        if self.config["enable_history"]:
            self._history.append((self.previous_error, derivative_of_error))
            
        # --- Lógica de Recompensa ---
        reward = -np.abs(error)  
        action_penalty = self.config["action_penalty_weight"] * (action_value - self.last_action)**2
        reward -= action_penalty

        if np.abs(error) < self.config["precision_bonus_threshold"]:
            reward += self.config["precision_bonus"]
            
        if self.config["enable_history"] and len(self._history) > 1:
            prev_errors = [h[0] for h in self._history]
            oscillation_penalty = 0.01 * np.std(prev_errors)
            reward -= oscillation_penalty


        self.previous_error = error
        self.last_action = action_value
        self.current_step += 1
        

        observation = [error, derivative_of_error]
        if self.config["enable_history"]:

            for h in self._history:
                observation.extend(h)

            while len(observation) < self.observation_space.shape[0]:
                observation.extend([0.0, 0.0])
                
        observation = np.array(observation, dtype=np.float32)
        

        terminated = False
        truncated = self.current_step >= self.config["max_steps"]
        
        self._step_times.append(time.time())
        if len(self._step_times) > 1:
            step_time = self._step_times[-1] - self._step_times[-2]
        else:
            step_time = 0
            
        info = {
            "step_time": step_time,
            "compute_time": self._compute_time,
            "current_temp": self.current_temp,
            "error": error
        }
        
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Reseta o ambiente para um novo episódio."""
        super().reset(seed=seed)
        

        self.current_temp = self.np_random.uniform(
            low=self.config["initial_temp_range"][0],
            high=self.config["initial_temp_range"][1]
        )

        self._disturbance_steps = []
        for step_range in self.config["disturbance_step_range"]:
            self._disturbance_steps.append(
                self.np_random.integers(low=step_range[0], high=step_range[1])
            )

        self.previous_error = self.config["setpoint"] - self.current_temp
        self.current_step = 0
        self.last_action = 0.0
        self._history.clear()
        self._compute_time = 0
        self._step_times = [time.time()]

        observation = [self.previous_error, 0.0]
        if self.config["enable_history"]:
            for _ in range(self.config["history_length"]):
                observation.extend([0.0, 0.0])
                
        return np.array(observation, dtype=np.float32), {}

    def render(self):
        """Renderiza o estado atual do ambiente."""
        if self.render_mode == 'human':
            print(
                f"Step: {self.current_step} | "
                f"Temp: {self.current_temp:.2f}°C | "
                f"Error: {self.previous_error:.2f} | "
                f"Setpoint: {self.config['setpoint']:.2f}°C"
            )
        elif self.render_mode == 'benchmark':
            if len(self._step_times) > 1:
                step_diffs = np.diff(self._step_times)
                avg_step_time = np.mean(step_diffs)
                std_step_time = np.std(step_diffs)
                total_elapsed_time = (time.time() - self._step_times[0]) + 1e-9
                compute_ratio = self._compute_time / total_elapsed_time
            else:
                avg_step_time = 0.0
                std_step_time = 0.0
                compute_ratio = 0.0

            print(
                f"Benchmark: Steps={self.current_step} | "
                f"Avg Step Time={avg_step_time:.4f}s (±{std_step_time:.4f}) | "
                f"Compute Ratio={compute_ratio:.1%}"
            )

    def close(self):
        """Fecha o ambiente."""
        pass