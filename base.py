import streamlit as st
import pandas as pd
import os
import json

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Resultados de RL",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard de Análise de Experimentos de RL")
st.write("Análise comparativa dos modelos Base, Paralelo e Otimizado com Optuna.")

# --- Funções de Carregamento e Processamento ---

@st.cache_data # Cache para não recarregar os dados a cada interação
def carregar_dados(logs_dir="logs"):
    """
    Carrega e processa os dados dos arquivos CSV de resultados.
    """
    arquivos = {
        "base": os.path.join(logs_dir, "base.csv"),
        "paralelo": os.path.join(logs_dir, "parallel_evaluation.csv"),
        "otimizado": os.path.join(logs_dir, "optuna_final_evaluation.csv")
    }

    dados_processados = []
    erros = []

    # Carregar e processar Modelo Base
    try:
        df_base = pd.read_csv(arquivos["base"])
        # Renomeia a coluna 'time' se existir, para padronizar
        if 'time' in df_base.columns:
            df_base = df_base.rename(columns={'time': 'training_time'})
        
        dados_processados.append({
            "Experimento": "Base",
            "Recompensa Média": df_base['reward_mean'].mean(),
            "Desvio Padrão Recompensa": df_base['reward_std'].mean(),
            "Tempo de Treino (s)": df_base['training_time'].mean()
        })
    except FileNotFoundError:
        erros.append("Arquivo 'base.csv' não encontrado.")

    # Carregar e processar Modelo Paralelo
    try:
        df_parallel = pd.read_csv(arquivos["paralelo"])
        dados_processados.append({
            "Experimento": "Paralelo",
            "Recompensa Média": df_parallel['reward_mean'].mean(),
            "Desvio Padrão Recompensa": df_parallel['reward_std'].mean(),
            "Tempo de Treino (s)": df_parallel['training_time'].mean()
        })
    except FileNotFoundError:
        erros.append("Arquivo 'parallel_evaluation.csv' não encontrado.")

    # Carregar e processar Modelo Otimizado
    try:
        df_optuna = pd.read_csv(arquivos["otimizado"])
        dados_processados.append({
            "Experimento": "Otimizado (Optuna)",
            "Recompensa Média": df_optuna['reward_mean'].iloc[0],
            "Desvio Padrão Recompensa": df_optuna['reward_std'].iloc[0],
            "Tempo de Treino (s)": df_optuna['training_time'].iloc[0]
        })
    except FileNotFoundError:
        erros.append("Arquivo 'optuna_final_evaluation.csv' não encontrado.")
    
    if not dados_processados:
        return None, erros, None

    # Retorna o DataFrame final e os hiperparâmetros do melhor modelo
    df_final = pd.DataFrame(dados_processados).set_index("Experimento")
    
    best_params = {}
    try:
        df_optuna = pd.read_csv(arquivos["otimizado"])
        best_params = json.loads(df_optuna['hyperparameters'].iloc[0])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass # Ignora se não conseguir ler os hiperparâmetros

    return df_final, erros, best_params

# --- Interface do Dashboard ---

# Carrega os dados
df_resultados, erros_carga, melhores_params = carregar_dados()

if erros_carga:
    for erro in erros_carga:
        st.warning(f"Aviso: {erro} Verifique se o arquivo está na pasta 'logs'.")

if df_resultados is not None and not df_resultados.empty:
    
    # --- Métricas de Destaque ---
    st.header("Resumo dos Resultados")
    
    # Encontra o melhor modelo pela recompensa
    melhor_experimento = df_resultados['Recompensa Média'].idxmax()
    maior_recompensa = df_resultados['Recompensa Média'].max()
    
    # Encontra o modelo mais rápido
    experimento_mais_rapido = df_resultados['Tempo de Treino (s)'].idxmin()
    menor_tempo = df_resultados['Tempo de Treino (s)'].min()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="🏆 Melhor Performance (Recompensa)",
            value=f"{maior_recompensa:.2f}",
            help=f"O modelo '{melhor_experimento}' teve a maior recompensa média."
        )
    with col2:
        st.metric(
            label="⏱️ Treino Mais Rápido",
            value=f"{menor_tempo:.2f} s",
            help=f"O modelo '{experimento_mais_rapido}' foi o mais rápido para treinar."
        )

    # --- Abas para organizar a informação ---
    tab1, tab2, tab3 = st.tabs(["📊 Gráficos Comparativos", "📋 Tabela Detalhada", "⚙️ Melhores Hiperparâmetros"])

    with tab1:
        st.subheader("Comparativo de Performance (Recompensa)")
        st.bar_chart(df_resultados, y="Recompensa Média", height=400)

        st.subheader("Comparativo de Eficiência (Tempo de Treino)")
        st.bar_chart(df_resultados, y="Tempo de Treino (s)", height=400)

    with tab2:
        st.subheader("Tabela Comparativa de Métricas")
        # Formata o DataFrame para melhor visualização
        st.dataframe(df_resultados.style.format({
            "Recompensa Média": "{:.2f}",
            "Desvio Padrão Recompensa": "{:.2f}",
            "Tempo de Treino (s)": "{:.2f}"
        }).highlight_max(subset="Recompensa Média", color="lightgreen")
          .highlight_min(subset="Tempo de Treino (s)", color="lightblue"),
          use_container_width=True
        )

    with tab3:
        st.subheader("Hiperparâmetros do Modelo Vencedor (Optuna)")
        if melhores_params:
            st.json(melhores_params)
        else:
            st.info("Não foi possível carregar os hiperparâmetros do arquivo do Optuna.")

else:
    st.error("Nenhum dado de resultado foi carregado. Verifique se os arquivos CSV estão na pasta 'logs' e se não estão vazios.")