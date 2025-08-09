import pandas as pd
import numpy as np
import os
import json
import streamlit as st
from scipy.stats import shapiro, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(page_title="Análise de Correlação", layout="wide")
st.title("📊 Análise de Correlação entre Índices e Produtividade")

# 1. Função para carregar dados da nuvem
@st.cache_data
def carregar_dados_da_nuvem():
    """Carrega os dados salvos na nuvem pelo código anterior"""
    temp_dir = "/tmp/streamlit_dados"
    
    try:
        # Tentar carregar CSV primeiro
        csv_path = f"{temp_dir}/resultados_analise.csv"
        gpkg_path = f"{temp_dir}/resultados_analise.gpkg"
        
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        elif os.path.exists(gpkg_path):
            import geopandas as gpd
            gdf = gpd.read_file(gpkg_path)
            return pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
        else:
            st.error("Nenhum arquivo de resultados encontrado na nuvem")
            return None
            
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

# Interface principal
with st.container():
    st.header("1. Carregamento de Dados")
    df = carregar_dados_da_nuvem()

    if df is None:
        st.warning("""
            ❌ Dados não encontrados. Por favor:
            1. Execute primeiro o código de processamento
            2. Certifique-se que os dados foram salvos na nuvem
        """)
        st.stop()
    
    st.success(f"✅ Dados carregados com sucesso (Total: {len(df)} registros)")
    
    with st.expander("Visualizar dados brutos"):
        st.dataframe(df.head())

# 2. Análise de Correlação
with st.container():
    st.header("2. Análise Estatística")
    
    # Selecionar colunas
    colunas_indices = [col for col in df.columns if any(x in col for x in 
                      ['NDVI', 'NDRE', 'CCCI', 'SAVI', 'GNDVI', 'NDMI', 'MSAVI2', 'NBR', 'TWI2', 'NDWI'])]
    
    if 'maduro_kg' not in df.columns:
        st.error("Coluna 'maduro_kg' não encontrada nos dados!")
        st.stop()
    
    colunas_analise = ['maduro_kg'] + colunas_indices
    
    # Teste de Normalidade
    with st.spinner("Realizando teste de normalidade..."):
        try:
            resultados_normalidade = []
            for coluna in colunas_analise:
                stat, p = shapiro(df[coluna])
                normal = p > 0.05
                resultados_normalidade.append({
                    'Variável': coluna, 
                    'p-valor': f"{p:.4f}", 
                    'Normal': 'Sim' if normal else 'Não'
                })

            df_normalidade = pd.DataFrame(resultados_normalidade)
            
            # Exibir resultados
            st.subheader("Teste de Normalidade (Shapiro-Wilk)")
            st.dataframe(df_normalidade.sort_values('p-valor'))
            
            proporcao_normal = df_normalidade['Normal'].value_counts(normalize=True).get('Sim', 0)
            st.info(f"**Proporção de variáveis normais:** {proporcao_normal:.1%}")

            # Seleção do método
            metodo = 'pearson' if proporcao_normal > 0.5 else 'spearman'
            st.success(f"**Método selecionado:** Correlação de {metodo.capitalize()}")
            
        except Exception as e:
            st.error(f"Erro no teste de normalidade: {str(e)}")
            st.stop()

    # Cálculo de Correlação
    with st.spinner("Calculando correlações..."):
        try:
            # Matriz de correlação
            corr_matrix = df[colunas_analise].corr(method=metodo.lower())
            
            # Cálculo de p-valores para Pearson
            p_values = None
            if metodo == 'pearson':
                p_values = pd.DataFrame(
                    np.zeros((len(colunas_analise), len(colunas_analise))),
                    columns=colunas_analise, 
                    index=colunas_analise
                )
                for i in colunas_analise:
                    for j in colunas_analise:
                        if i != j:
                            _, p_val = pearsonr(df[i], df[j])
                            p_values.loc[i, j] = p_val

            # Top 5 correlações
            st.subheader("Top 5 Correlações com Produtividade")
            correlacoes = corr_matrix['maduro_kg'].drop('maduro_kg')
            top5 = correlacoes.abs().sort_values(ascending=False).head(5)
            
            for idx, valor in top5.items():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric(
                        label=idx,
                        value=f"{valor:.3f}",
                        help="Positiva" if valor > 0 else "Negativa"
                    )
                with col2:
                    if metodo == 'pearson' and p_values is not None:
                        p_val = p_values.loc['maduro_kg', idx]
                        sig = "✅ Significativa" if p_val < 0.05 else "⚠️ Não significativa"
                        st.caption(f"p-valor: {p_val:.4f} ({sig})")

            # Visualização
            st.subheader("Matriz de Correlação")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt=".2f",
                ax=ax
            )
            ax.set_title(f"Matriz de Correlação ({metodo.capitalize()})")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erro no cálculo de correlação: {str(e)}")

# Seção de interpretação
with st.expander("📚 Guia de Interpretação"):
    st.markdown("""
    ## Como interpretar os resultados:
    
    **Correlação de Pearson**  
    ▸ Mede relações lineares entre variáveis contínuas  
    ▸ Requer normalidade dos dados  
    ▸ Valores próximos de 1 ou -1 indicam forte relação  
    
    **Correlação de Spearman**  
    ▸ Mede relações monotônicas (não necessariamente lineares)  
    ▸ Não requer normalidade  
    ▸ Menos sensível a outliers  
    
    **p-valor (Pearson)**  
    ▸ p < 0.05 → Correlação estatisticamente significativa  
    ▸ p ≥ 0.05 → Não podemos afirmar que há correlação  
    
    **Dicas importantes:**  
    • Correlação ≠ Causalidade  
    • Considere sempre o contexto agronômico  
    • Valores acima de 0.7 geralmente indicam relações fortes  
    """)
