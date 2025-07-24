import pandas as pd

# Carregar os dados
base_df = pd.read_csv("base.csv")
parallel_df = pd.read_csv("parallel_evaluation.csv")
optuna_df = pd.read_csv("optuna_final_evaluation.csv")

# Padronizar nomes de colunas
base_df.rename(columns={"time": "training_time"}, inplace=True)
parallel_df.rename(columns={"time": "training_time"}, inplace=True)
optuna_df.rename(columns={"training_time": "training_time"}, inplace=True)

# Adicionar coluna de método
base_df["method"] = "Base PPO"
parallel_df["method"] = "Parallel PPO"
optuna_df["method"] = "Optuna PPO"

# Corrigir colunas ausentes em optuna_df (caso não existam)
if "cpu_mean" not in optuna_df.columns:
    optuna_df["cpu_mean"] = None
if "gpu_mean" not in optuna_df.columns:
    optuna_df["gpu_mean"] = None

# Selecionar e combinar colunas de interesse
cols = ["method", "reward_mean", "reward_std", "training_time", "cpu_mean", "gpu_mean"]
combined_df = pd.concat([base_df[cols], parallel_df[cols], optuna_df[cols]])

# Converter colunas numéricas
for col in ["reward_mean", "reward_std", "training_time", "cpu_mean", "gpu_mean"]:
    combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

# Calcular estatísticas descritivas
summary = combined_df.groupby("method").agg({
    "reward_mean": ["mean", "std"],
    "reward_std": ["mean"],
    "training_time": ["mean"],
    "cpu_mean": ["mean"],
    "gpu_mean": ["mean"]
}).round(2)

# Melhorar visualização do resultado
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary.reset_index(inplace=True)

print("📊 Comparativo de Desempenho por Método:\n")
print(summary)
