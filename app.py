import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")
st.title("Análise de Desempenho: DRL com Otimizações")

# Tentar carregar todos os arquivos CSV e concatenar
try:
    # Carrega os CSVs de resultados finais
    df_base = pd.read_csv("logs/base.csv")
    df_parallel = pd.read_csv("logs/parallel.csv")
    df_optuna = pd.read_csv("logs/optuna.csv")
    
    data = pd.concat([df_base, df_parallel, df_optuna])
    
    # Calcular médias e desvio padrão para cada configuração
    summary_data = data.groupby("config").agg(
        reward_mean=('reward_mean', 'mean'),
        reward_std=('reward_mean', 'std'),
        time_mean=('time', 'mean'),
        time_std=('time', 'std'),
        cpu_mean=('cpu_mean', 'mean'),
        cpu_std=('cpu_mean', 'std'),
        gpu_mean=('gpu_mean', 'mean'),
        gpu_std=('gpu_mean', 'std')
    ).reset_index()

    st.subheader("Resultados Médios por Configuração (N Runs)")
    st.dataframe(summary_data.round(2))

    col1, col2 = st.columns(2)
    with col1:
        # Gráfico de Recompensa Média com Barras de Erro
        fig1 = go.Figure(data=[
            go.Bar(
                name='Recompensa Média',
                x=summary_data['config'],
                y=summary_data['reward_mean'],
                error_y=dict(type='data', array=summary_data['reward_std'], visible=True)
            )
        ])
        fig1.update_layout(title="Recompensa Média por Configuração",
                           xaxis_title="Configuração",
                           yaxis_title="Recompensa Média")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Gráfico de Tempo de Treinamento com Barras de Erro
        fig2 = go.Figure(data=[
            go.Bar(
                name='Tempo de Treinamento',
                x=summary_data['config'],
                y=summary_data['time_mean'],
                error_y=dict(type='data', array=summary_data['time_std'], visible=True)
            )
        ])
        fig2.update_layout(title="Tempo de Treinamento (s) por Configuração",
                           xaxis_title="Configuração",
                           yaxis_title="Tempo (s)")
        st.plotly_chart(fig2, use_container_width=True)

    # Gráfico de Uso de CPU/GPU com Barras de Erro
    fig3 = go.Figure(data=[
        go.Bar(name='Uso de CPU', x=summary_data['config'], y=summary_data['cpu_mean'],
               error_y=dict(type='data', array=summary_data['cpu_std'], visible=True)),
        go.Bar(name='Uso de GPU', x=summary_data['config'], y=summary_data['gpu_mean'],
               error_y=dict(type='data', array=summary_data['gpu_std'], visible=True))
    ])
    fig3.update_layout(barmode='group', title="Uso Médio de CPU/GPU (%) por Configuração",
                       xaxis_title="Configuração",
                       yaxis_title="Uso (%)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Curvas de Convergência (Recompensa Média por Timesteps)")

    convergence_data_list = []
    configs = ["base", "parallel", "optuna"]
    num_runs_expected = 3 

    for config in configs:
        for run_idx in range(1, num_runs_expected + 1):
            log_dir = os.path.join("logs", f"{config}_run_{run_idx}")
            log_path = os.path.join(log_dir, "convergence_data.csv")
            if os.path.exists(log_path):
                df_conv = pd.read_csv(log_path)
                df_conv["config"] = config
                df_conv["run"] = run_idx
                convergence_data_list.append(df_conv)
    
    if convergence_data_list:
        all_convergence_data = pd.concat(convergence_data_list)
        
        # Calcular a média e o desvio padrão da recompensa por timesteps para cada configuração
        mean_convergence = all_convergence_data.groupby(['config', 'timesteps']).agg(
            mean_reward=('mean_reward', 'mean'),
            std_reward=('mean_reward', 'std')
        ).reset_index()

        fig_conv = go.Figure()

        colors = {
            "base": px.colors.qualitative.Plotly[0],
            "parallel": px.colors.qualitative.Plotly[1],
            "optuna": px.colors.qualitative.Plotly[2]
        }

        for config in configs:
            config_data = mean_convergence[mean_convergence['config'] == config]
            
            fig_conv.add_trace(go.Scatter(
                x=config_data['timesteps'], 
                y=config_data['mean_reward'], 
                mode='lines', 
                name=f'{config.capitalize()} (Média)',
                line=dict(width=2, color=colors[config])
            ))
            
            timesteps_fill = config_data['timesteps'].tolist() + config_data['timesteps'].iloc[::-1].tolist()
            upper_bound = (config_data['mean_reward'] + config_data['std_reward']).tolist()
            lower_bound_reversed = (config_data['mean_reward'] - config_data['std_reward']).iloc[::-1].tolist()

            fig_conv.add_trace(go.Scatter(
                x=timesteps_fill, 
                y=upper_bound + lower_bound_reversed, 
                fill='toself',
                fillcolor=colors[config] + '30', 
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))

        fig_conv.update_layout(
            title="Curvas de Convergência (Recompensa Média vs. Timesteps)",
            xaxis_title="Timesteps de Treinamento",
            yaxis_title="Recompensa Média",
            hovermode="x unified",
            legend_title="Configuração"
        )
        st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.info("Nenhum dado de convergência encontrado para plotar. Por favor, execute os scripts de treinamento primeiro.")


except FileNotFoundError:
    st.warning("Certifique-se de que os arquivos de log (base.csv, parallel.csv, optuna.csv, e os diretórios de convergência) foram gerados pelos scripts de treinamento.")
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar ou processar os dados: {e}")
    st.info("Verifique se os scripts de treinamento foram executados e os arquivos CSV estão no formato correto. Erro detalhado: " + str(e))