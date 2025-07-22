import time
import psutil
import pynvml

# Variáveis globais para gerenciar o estado da pynvml
_gpu_handle = None
_nvml_initialized = False

try:
    pynvml.nvmlInit() # Tenta inicializar pynvml
    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Tenta obter o handle da GPU 0
    _nvml_initialized = True
except pynvml.NVMLError as error:
    print(f"ATENÇÃO: Erro ao inicializar pynvml: {error}. O monitoramento da GPU não estará disponível.")
    _nvml_initialized = False

def start_timer():
    """Inicia o temporizador."""
    return time.time()

def end_timer(start_time):
    """Calcula a duração desde o início do temporizador."""
    return time.time() - start_time

def get_cpu_usage():
    """
    Retorna o uso atual da CPU como uma porcentagem.
    Usa interval=None para não bloquear e obter a porcentagem desde a última chamada.
    """
    return psutil.cpu_percent(interval=None)

def get_gpu_usage():
    """
    Retorna o uso da GPU como uma porcentagem. Retorna 0 se a GPU não estiver disponível ou ocorrer erro.
    """
    if _nvml_initialized and _gpu_handle:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
            return util.gpu
        except pynvml.NVMLError as error:
            # Captura erros durante a coleta de uso (ex: GPU em estado de suspensão)
            # print(f"ATENÇÃO: Erro ao coletar uso da GPU: {error}. Retornando 0.") # Descomente para debug
            return 0
    return 0

# Não é estritamente necessário um shutdown explícito para scripts curtos,
# mas é boa prática para liberar recursos em aplicações de longa duração.
# pynvml.nvmlShutdown() seria chamado no final do script principal se fosse necessário.