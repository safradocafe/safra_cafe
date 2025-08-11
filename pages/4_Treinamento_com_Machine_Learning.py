# pages/2_Treinamento.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import joblib
import io
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# Configuração da página e supressão de avisos
st.set_page_config(layout="wide")
warnings.filterwarnings('ignore')
np.random.seed(42)

st.markdown("<h3>Treinamento e Avaliação de Modelos</h3>", unsafe_allow_html=True)
st.markdown("<h4>Treinamento de modelos de regressão para predição de produtividade.</h4>", unsafe_allow_html=True)
st.markdown("---")

# 1. Carregar e preparar os dados (da sessão)
if 'gdf_resultado' not in st.session_state or st.session_state['gdf_resultado'] is None:
    st.warning("⚠️ Os dados processados não foram encontrados na sessão. Por favor, volte à página 'Processamento' e execute a análise primeiro.")
    st.stop()

# Converte o GeoDataFrame para um DataFrame padrão do Pandas
df = pd.DataFrame(st.session_state['gdf_resultado'].drop(columns=['geometry'], errors='ignore'))

# Verificação essencial
if 'maduro_kg' not in df.columns:
    st.error("❌ A coluna 'maduro_kg' não foi encontrada nos dados. O treinamento não pode continuar.")
    st.stop()

st.success("✅ Dados processados carregados com sucesso da sessão!")
with st.expander("Visualizar dados para treinamento"):
    st.dataframe(df.head())

X = df.drop(columns=['maduro_kg'])
y = df['maduro_kg']

# 2. Definir os modelos (sem mudanças)
modelos = {
    "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam',
                        max_iter=2000, early_stopping=True, random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Lasso": Lasso(alpha=0.1, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
}
modelos_escalonados = ["MLP", "SVR", "KNN", "Ridge", "Lasso", "ElasticNet"]

# 3. Interface de usuário para treinamento
st.sidebar.header("Parâmetros do Treinamento")
num_execucoes = st.sidebar.number_input("Número de execuções", min_value=1, value=20, help="Quantas vezes cada modelo será treinado para encontrar o melhor resultado.")
if st.sidebar.button("▶️ Iniciar Treinamento e Avaliação"):
    
    st.subheader("Análise dos Resultados")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    resultados_df = pd.DataFrame()
    melhores_modelos = {}

    for i in range(num_execucoes):
        status_text.text(f"Executando modelos... (Execução {i + 1}/{num_execucoes})")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for nome, modelo in modelos.items():
                X_tr, X_te = (X_train_scaled, X_test_scaled) if nome in modelos_escalonados else (X_train, X_test)
                
                try:
                    modelo.fit(X_tr, y_train)
                    y_pred = modelo.predict(X_te)
                    
                    r2_atual = r2_score(y_test, y_pred)
                    if nome not in melhores_modelos or r2_atual > melhores_modelos[nome]['r2']:
                         melhores_modelos[nome] = {
                            'modelo': modelo,
                            'r2': r2_atual,
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'execucao': i + 1,
                            'X_tr': X_tr,
                            'y_tr': y_train,
                            'X_te': X_te,
                            'y_te': y_test
                        }

                    resultados_df = pd.concat([resultados_df, pd.DataFrame([{
                        'execucao': i + 1,
                        'modelo': nome,
                        'r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                    }])], ignore_index=True)

                except Exception as e:
                    st.warning(f"Erro no modelo {nome} (execução {i+1}): {str(e)}")
            
            progress_bar.progress((i + 1) / num_execucoes)

        except Exception as e:
            st.error(f"Erro na execução {i + 1}: {str(e)}")
            continue

    status_text.success("✅ Treinamento concluído!")

    # Análise dos resultados e exibição
    st.subheader("Resultados de Todas as Execuções")
    with st.expander("Ver tabela de resultados"):
        st.dataframe(resultados_df)

    if not resultados_df.empty:
        df_melhores_execucoes = pd.DataFrame(
            [{"modelo": k, "r2": v['r2'], "rmse": v['rmse']} for k, v in melhores_modelos.items()]
        )
        
        melhor_modelo_nome = df_melhores_execucoes.loc[df_melhores_execucoes["r2"].idxmax()]["modelo"]
        melhor_modelo = melhores_modelos[melhor_modelo_nome]['modelo']
        melhor_execucao_params = melhores_modelos[melhor_modelo_nome]
        
        st.markdown("---")
        st.subheader("🏆 Melhor Modelo Global")
        col1, col2, col3 = st.columns(3)
        col1.metric("Modelo", melhor_modelo_nome)
        col2.metric("Melhor R²", f"{melhor_execucao_params['r2']:.4f}")
        col3.metric("Melhor RMSE", f"{melhor_execucao_params['rmse']:.4f}")
        
        # Download do melhor modelo
        buffer = io.BytesIO()
        joblib.dump(melhor_modelo, buffer)
        st.download_button(
            label=f"💾 Baixar o melhor modelo ({melhor_modelo_nome}.pkl)",
            data=buffer.getvalue(),
            file_name=f"melhor_modelo_{melhor_modelo_nome}.pkl",
            mime="application/octet-stream"
        )
        
        # Importância das features
        st.markdown("---")
        st.subheader("Importância das Features")

        try:
            X_for_importance = melhor_execucao_params['X_tr']
            y_for_importance = melhor_execucao_params['y_tr']

            result_importance = permutation_importance(melhor_modelo, X_for_importance, y_for_importance, n_repeats=10, random_state=42)
            importancia_indices = pd.DataFrame({
                "Índice": X.columns,
                "Importância": result_importance.importances_mean
            }).sort_values(by="Importância", ascending=False)
            
            top5_indices = importancia_indices.head(5)
            top5_indices["Porcentagem"] = (top5_indices["Importância"] / top5_indices["Importância"].sum()) * 100
            
            st.write(f"Os 5 índices espectrais mais importantes para o modelo {melhor_modelo_nome}:")
            st.dataframe(top5_indices.round(4))

            csv_top5 = top5_indices.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar top 5 índices como CSV",
                data=csv_top5,
                file_name="top5_indices.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.warning(f"Não foi possível calcular a importância das features: {str(e)}")

        # Avaliação estatística com TESTE
        st.markdown("---")
        st.subheader("Avaliação com Dados de TESTE")
        
        y_test_final = melhor_execucao_params['y_te']
        y_pred_final = melhor_modelo.predict(melhor_execucao_params['X_te'])

        def avaliacao_estatistica(y_real, y_pred):
            r2 = r2_score(y_real, y_pred)
            rmse = np.sqrt(mean_squared_error(y_real, y_pred))
            residuals = y_real - y_pred
            rmse_relativo = (rmse / np.mean(y_real)) * 100
            bias = np.mean(residuals)
            bias_relativo = (bias / np.mean(y_real)) * 100
            return {
                'R²': r2,
                'RMSE': rmse,
                'RMSE Relativo (%)': rmse_relativo,
                'Bias': bias,
                'Bias Relativo (%)': bias_relativo
            }

        metricas = avaliacao_estatistica(y_test_final, y_pred_final)

        col_metr_1, col_metr_2, col_metr_3, col_metr_4, col_metr_5 = st.columns(5)
        col_metr_1.metric("R²", f"{metricas['R²']:.4f}")
        col_metr_2.metric("RMSE", f"{metricas['RMSE']:.4f}")
        col_metr_3.metric("RMSE Relativo (%)", f"{metricas['RMSE Relativo (%)']:.2f}%")
        col_metr_4.metric("Bias", f"{metricas['Bias']:.4f}")
        col_metr_5.metric("Bias Relativo (%)", f"{metricas['Bias Relativo (%)']:.2f}%")
        
        # Tabela comparativa e visualização
        st.markdown("---")
        st.subheader("Comparativo de Valores Reais vs. Preditos")
        
        df_comparativo = pd.DataFrame({
            'Produtividade_Real': y_test_final,
            'Produtividade_Predita': y_pred_final,
            'Resíduo': y_test_final - y_pred_final
        }).reset_index(drop=True)
        df_comparativo['Erro_Relativo'] = (df_comparativo['Resíduo'] / df_comparativo['Produtividade_Real']) * 100

        st.dataframe(df_comparativo.sort_values('Produtividade_Real').head(10).round(4))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_comparativo['Produtividade_Real'], df_comparativo['Produtividade_Predita'], alpha=0.7)
        ax.plot([y_test_final.min(), y_test_final.max()], [y_test_final.min(), y_test_final.max()], 'r--', lw=2, label='Linha de 1:1')
        ax.set_title("Produtividade Predita vs. Real (Dados de Teste)")
        ax.set_xlabel("Produtividade Real (kg)")
        ax.set_ylabel("Produtividade Predita (kg)")
        ax.legend()
        st.pyplot(fig)
        
    else:
        st.warning("⚠️ Nenhum resultado de treinamento foi gerado.")

# Explicação didática das métricas
st.markdown("---")
with st.expander("📘 Interpretação das Métricas"):
    st.markdown("""
🔹 **R² (Coeficiente de Determinação):**
- Mede o quanto da variabilidade dos dados reais é explicada pelo modelo.
- Varia de **0** a **1**. Quanto mais próximo de **1**, melhor o desempenho.

🔹 **RMSE (Root Mean Squared Error):**
- Erro médio quadrático da predição, em unidades reais (ex: kg). É sensível a erros grandes.
- Quanto mais próximo de **zero**, melhor.

🔹 **RMSE Relativo (%):**
- RMSE em relação à média dos valores reais (em percentual).
- Permite comparar erros entre diferentes contextos ou culturas agrícolas.

🔹 **Bias (Viés)::**
- Indica se o modelo tende a superestimar (bias negativo) ou subestimar (bias positivo) os valores.
- Idealmente, deve ser próximo de **zero**.

🔹 **Bias Relativo (%):**
- Bias expresso em relação à média dos valores reais.

✅ **Recomendações:**
- Busque R² alto (≥ 0.75), e RMSE e Bias baixos.
- Sempre avalie RMSE e Bias relativos para entender o impacto percentual do erro.
    """)
