import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from utils.metrics import get_cpu_usage, get_gpu_usage # Importar as funções de métricas

class CPU_GPU_Logger(BaseCallback):
    """
    Callback personalizado para logar o uso de CPU e GPU durante o treinamento.
    """
    def __init__(self, verbose: int = 0):
        super(CPU_GPU_Logger, self).__init__(verbose)
        self.cpu_usages = []
        self.gpu_usages = []

    def _on_step(self) -> bool:
        """
        Chamado a cada passo de treinamento para coletar o uso de CPU/GPU.
        """
        self.cpu_usages.append(get_cpu_usage())
        self.gpu_usages.append(get_gpu_usage())
        return True # Retorna True para continuar o treinamento

    def _on_training_end(self) -> None:
        """
        Chamado no final do treinamento para processar os dados coletados.
        Você pode adicionar lógica para salvar ou imprimir médias aqui se quiser,
        mas já estamos fazendo isso nos scripts principais.
        """
        pass

    def get_mean_cpu_usage(self):
        """Retorna a média do uso de CPU coletado."""
        return np.mean(self.cpu_usages) if self.cpu_usages else 0

    def get_mean_gpu_usage(self):
        """Retorna a média do uso de GPU coletado."""
        return np.mean(self.gpu_usages) if self.gpu_usages else 0