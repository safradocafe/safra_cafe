import pandas as pd
import numpy as np
import os
import json
import streamlit as st
from scipy.stats import shapiro, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(page_title="Análise de correlação", layout="wide")
st.title("📊 Análise de Correlação entre índices espectrais e produtividade")

# 1. Carregamento de Dados
with st.container():
    st.header("1. Carregamento de dados")

    # Tenta carregar dados do st.session_state
    if 'gdf_resultado' in st.session_state and st.session_state['gdf_resultado'] is not None:
        df = st.session_state['gdf_resultado']
        st.success(f"✅ Dados carregados com sucesso da sessão atual (Total: {len(df)} registros)")
    else:
        st.warning("""
            ❌ Dados não encontrados na sessão atual. Por favor:
            1. Execute o código de processamento primeiro na mesma sessão.
            2. Clique no botão '▶️ Executar análise' para salvar os resultados na sessão.
        """)
        st.stop()
    
    with st.expander("Visualizar dados brutos"):
        st.dataframe(df.head())

# 2. Análise de Correlação
with st.container():
    st.header("2. Análise estatística")
    
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
                stat, p = shapiro(df[coluna].dropna()) # Adicionado .dropna() para evitar erros
                normal = p > 0.05
                resultados_normalidade.append({
                    'Variável': coluna, 
                    'p-valor': f"{p:.4f}", 
                    'Normal': 'Sim' if normal else 'Não'
                })

            df_normalidade = pd.DataFrame(resultados_normalidade)
            
            # Exibir resultados
            st.subheader("Teste de normalidade (Shapiro-Wilk)")
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
                            # Adicionado .dropna() para garantir que os dados sejam válidos
                            _, p_val = pearsonr(df[i].dropna(), df[j].dropna()) 
                            p_values.loc[i, j] = p_val

            # Top 5 correlações
            st.subheader("Top 5 Correlações com Produtividade")
            correlacoes = pd.Series({col: df[['maduro_kg', col]].corr(method=metodo.lower()).iloc[0, 1] 
                         for col in colunas_indices if col != 'maduro_kg'})
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
            
        except Exception as e:
            st.error(f"Erro no cálculo de correlação: {str(e)}")

# Seção de interpretação
with st.expander("📚 Como interpretar os resultados"):
    st.markdown("""
### 📘 Interpretação das Correlações

🔹 **Correlação de Pearson:**
- Mede a relação linear entre duas variáveis numéricas.
- Pressupõe que os dados sejam normalmente distribuídos.
- Varia de **-1** a **1**:
    + **1** → correlação perfeita positiva
    + **0** → nenhuma correlação
    + **-1** → correlação perfeita negativa
- Exemplo: um valor de **0.75** indica que quando uma variável aumenta, a outra tende a aumentar também.

🔹 **Correlação de Spearman:**
- Mede a relação monotônica (não necessariamente linear) entre duas variáveis.
- Baseia-se na ordenação dos dados (ranks).
- Não exige distribuição normal.
- Útil quando os dados possuem outliers ou relações não lineares.

🔹 **p-valor (apenas Pearson no script):**
- Indica a significância estatística da correlação.
- **p < 0.05** → correlação estatisticamente significativa (nível de confiança de 95%).

🔹 **Como interpretar a força da correlação:**
- **0.00 a 0.30** → fraca
- **0.31 a 0.50** → moderada
- **0.51 a 0.70** → forte
- **0.71 a 0.90** → muito forte
- **acima de 0.90** → quase perfeita

✅ **Dica:**
- Correlações não implicam causalidade.
- Use a análise de correlação como **etapa exploratória**, para saber se os dados analisados se correlacionam bem de alguma forma, não como prova de relação causal. Boas correlações negativas (próximo de -1) também podem indicar tendências dos dados.
    """)
