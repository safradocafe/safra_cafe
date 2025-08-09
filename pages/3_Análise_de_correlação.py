import pandas as pd
import numpy as np
from scipy.stats import shapiro, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# 1. Carregar os dados da nuvem do Streamlit
def carregar_dados_da_nuvem():
    """Carrega os dados salvos na nuvem pelo código anterior"""
    temp_dir = "/tmp/streamlit_dados"
    
    try:
        # Carregar CSV se existir
        if os.path.exists(f"{temp_dir}/resultados_analise.csv"):
            df = pd.read_csv(f"{temp_dir}/resultados_analise.csv")
            return df
        
        # Se não tiver CSV, carregar GPKG e converter
        elif os.path.exists(f"{temp_dir}/resultados_analise.gpkg"):
            import geopandas as gpd
            gdf = gpd.read_file(f"{temp_dir}/resultados_analise.gpkg")
            df = pd.DataFrame(gdf.drop(columns='geometry'))
            return df
        
        else:
            raise FileNotFoundError("Nenhum arquivo de resultados encontrado na nuvem")
            
    except Exception as e:
        print(f"Erro ao carregar dados da nuvem: {str(e)}")
        return None

# Carregar os dados
df = carregar_dados_da_nuvem()

if df is None:
    print("❌ Não foi possível carregar os dados da nuvem. Verifique se o código anterior foi executado e salvou os resultados.")
else:
    print("✅ Dados carregados com sucesso da nuvem do Streamlit")
    print(f"Total de registros: {len(df)}")
    print("\nPrimeiras linhas dos dados:")
    print(df.head())

    # 2. Selecionar colunas de interesse
    colunas_indices = [col for col in df.columns if any(x in col for x in ['NDVI', 'NDRE', 'CCCI', 'SAVI', 'GNDVI', 'NDMI', 'MSAVI2', 'NBR', 'TWI2', 'NDWI'])]
    colunas_analise = ['maduro_kg'] + colunas_indices

    # Verificar se todas as colunas necessárias existem
    colunas_faltantes = [col for col in colunas_analise if col not in df.columns]
    if colunas_faltantes:
        print(f"\n⚠️ Atenção: Algumas colunas necessárias não foram encontradas: {colunas_faltantes}")
        print("Colunas disponíveis:", df.columns.tolist())
    else:
        # 3. Teste de Shapiro-Wilk (normalidade)
        resultados_normalidade = []
        for coluna in colunas_analise:
            stat, p = shapiro(df[coluna])
            normal = 'Sim' if p > 0.05 else 'Não'
            resultados_normalidade.append({'Variável': coluna, 'p-valor': p, 'Normal': normal})

        df_normalidade = pd.DataFrame(resultados_normalidade)
        print("\nResultados do Teste de Normalidade (Shapiro-Wilk):")
        print(df_normalidade.sort_values('p-valor'))

        # 4. Proporção de variáveis normais
        proporcao_normal = df_normalidade['Normal'].value_counts(normalize=True).get('Sim', 0)
        print(f"\nProporção de variáveis normais: {proporcao_normal:.1%}")

        # 5. Escolha do método de correlação
        if proporcao_normal > 0.5:
            metodo = 'pearson'
            print("\nUsando correlação de Pearson (maioria normal)")
        else:
            metodo = 'spearman'
            print("\nUsando correlação de Spearman (maioria não-normal)")

        # 6. Matriz de correlação e p-valores
        corr_matrix = df[colunas_analise].corr(method=metodo.lower())

        # Calcular p-valores se Pearson
        if metodo == 'pearson':
            p_values = pd.DataFrame(np.zeros((len(colunas_analise), len(colunas_analise))),
                                    columns=colunas_analise, index=colunas_analise)
            for i in colunas_analise:
                for j in colunas_analise:
                    if i != j:
                        _, p_val = pearsonr(df[i], df[j])
                        p_values.loc[i, j] = p_val
                    else:
                        p_values.loc[i, j] = np.nan

        # 7. Top 5 correlações com maduro_kg
        correlacoes_maduro = corr_matrix['maduro_kg'].drop('maduro_kg')
        melhores_5 = correlacoes_maduro.abs().sort_values(ascending=False).head(5)

        print("\nTop 5 índices com maior correlação (absoluta) com produtividade (maduro_kg):")
        for idx, valor in melhores_5.items():
            if metodo == 'pearson':
                p_val = p_values.loc['maduro_kg', idx]
                print(f"- {idx}: {valor:.3f} ({'positiva' if corr_matrix.loc['maduro_kg', idx] > 0 else 'negativa'}), p-valor: {p_val:.4f}")
            else:
                print(f"- {idx}: {valor:.3f} ({'positiva' if corr_matrix.loc['maduro_kg', idx] > 0 else 'negativa'})")

        # 8. Explicação didática sobre correlação
        print("\n=== 📘 Interpretação das Correlações ===")
        print("""
🔹 Correlação de Pearson:
    - Mede a relação linear entre duas variáveis numéricas.
    - Pressupõe que os dados sejam normalmente distribuídos.
    - Varia de -1 a 1:
        + 1 → correlação perfeita positiva
        0 → nenhuma correlação
        -1 → correlação perfeita negativa
    - Exemplo: um valor de 0.75 indica que quando uma variável aumenta, a outra tende a aumentar também.

🔹 Correlação de Spearman:
    - Mede a relação monotônica (não necessariamente linear) entre duas variáveis.
    - Baseia-se na ordenação dos dados (ranks).
    - Não exige distribuição normal.
    - Útil quando os dados possuem outliers ou relações não lineares.

🔹 p-valor (apenas Pearson no script):
    - Indica a significância estatística da correlação.
    - p < 0.05 → correlação estatisticamente significativa (nível de confiança de 95%).

🔹 Como interpretar a força da correlação:
    - 0.00 a 0.30 → fraca
    - 0.31 a 0.50 → moderada
    - 0.51 a 0.70 → forte
    - 0.71 a 0.90 → muito forte
    - acima de 0.90 → quase perfeita

✅ Dica:
    - Correlações não implicam causalidade.
    - Use a análise de correlação como **etapa exploratória**, para saber se os dados analisados se correlacionam bem de alguma forma, não como prova de relação causal. Boas correlações negativas (próximo de -1) também podem indicar tendências dos dados.
        """)

        # 9. Visualização (opcional)
        if 'matplotlib' in sys.modules:
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(f"Matriz de Correlação ({metodo.capitalize()})")
            plt.tight_layout()
            plt.show()
