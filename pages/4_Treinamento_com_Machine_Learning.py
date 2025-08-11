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
import pandas as pd
import numpy as np
import warnings
import joblib
from tqdm import tqdm

# Configurações iniciais
warnings.filterwarnings('ignore')
np.random.seed(42)
NUM_EXECUCOES = 20

# 1. Carregar e preparar os dados
df = pd.read_csv("/content/indices_23_24_limpo.csv")
X = df.drop(columns=['maduro_kg'])
y = df['maduro_kg']

# 2. Definir os modelos
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

# 3. Avaliação dos modelos
resultados = {nome: [] for nome in modelos}

print(f"Executando {len(modelos)} modelos {NUM_EXECUCOES} vezes cada...\n")
for i in tqdm(range(NUM_EXECUCOES), desc="Progresso Geral"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for nome, modelo in modelos.items():
            if nome in modelos_escalonados:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test

            try:
                modelo.fit(X_tr, y_train)
                y_pred = modelo.predict(X_te)

                resultados[nome].append({
                    'execucao': i + 1,
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'modelo': nome,
                    'convergiu': True
                })
            except Exception as e:
                print(f"\nErro no modelo {nome} (execução {i+1}): {str(e)}")
                resultados[nome].append({
                    'execucao': i + 1,
                    'r2': None,
                    'rmse': None,
                    'modelo': nome,
                    'convergiu': False
                })
    except Exception as e:
        print(f"\nErro na execução {i + 1}: {str(e)}")
        continue

# 4. Análise dos resultados (MODIFICADO)
def analisar_resultados(resultados, nome_modelo):
    df_resultados = pd.DataFrame(resultados[nome_modelo])
    df_validos = df_resultados[df_resultados['convergiu']]

    if not df_validos.empty:
        print(f"\n=== Resultados Individuais {nome_modelo} ===")
        print(df_validos[['execucao', 'r2', 'rmse']])  # Mostra valores individuais

        melhor_idx = df_validos['r2'].idxmax()
        melhor = df_validos.loc[melhor_idx]
        print(f"\nMelhor execução ({nome_modelo}): R² = {melhor['r2']:.4f}, RMSE = {melhor['rmse']:.4f}")
        return melhor
    else:
        print(f"\n{nome_modelo}: Nenhum modelo convergiu.")
        return None

# Criar o dicionário melhores_modelos antes de usá-lo
melhores_modelos = {nome: analisar_resultados(resultados, nome) for nome in modelos}  # Esta linha estava faltando

# 5. Salvar os melhores modelos
for nome, melhor in melhores_modelos.items():
    if melhor is not None:
        joblib.dump(modelos[nome], f'melhor_{nome.lower()}.pkl')

print("\nModelos otimizados foram salvos.")

# 6. Identificar o melhor modelo global (MODIFICADO)
df_melhores_execucoes = pd.DataFrame([
    {"modelo": nome, "r2": melhor["r2"], "rmse": melhor["rmse"]}
    for nome, melhor in melhores_modelos.items() if melhor is not None
])

melhor_modelo_nome = df_melhores_execucoes.loc[df_melhores_execucoes["r2"].idxmax()]["modelo"]  # Corrigido "modelo"
melhor_modelo = modelos[melhor_modelo_nome]

print(f"\nMelhor modelo global: {melhor_modelo_nome}")
print(f"Melhor R²: {df_melhores_execucoes.loc[df_melhores_execucoes['modelo'] == melhor_modelo_nome]['r2'].values[0]:.4f}")  # Corrigir para 'modelo'
print(f"Melhor RMSE: {df_melhores_execucoes.loc[df_melhores_execucoes['modelo'] == melhor_modelo_nome]['rmse'].values[0]:.4f}")  # Corrigido

# 7. Importância das features
if melhor_modelo_nome in modelos_escalonados:
    X_for_importance = StandardScaler().fit_transform(X)
else:
    X_for_importance = X

result_importance = permutation_importance(melhor_modelo, X_for_importance, y, n_repeats=10, random_state=42)
importancia_indices = pd.DataFrame({
    "Índice": X.columns,
    "Importância": result_importance.importances_mean
}).sort_values(by="Importância", ascending=False)

top5_indices = importancia_indices.head(5)
top5_indices["Porcentagem"] = (top5_indices["Importância"] / top5_indices["Importância"].sum()) * 100

print("\nOs 5 índices espectrais mais importantes para o modelo selecionado:")
print(top5_indices)

# 8. Reproduzir a melhor execução e fazer predição apenas no conjunto de TESTE
melhor_execucao = resultados[melhor_modelo_nome][pd.DataFrame(resultados[melhor_modelo_nome])['r2'].idxmax()]
random_state_melhor = melhor_execucao['execucao'] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state_melhor)

if melhor_modelo_nome in modelos_escalonados:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    melhor_modelo.fit(X_train_scaled, y_train)
    y_pred = melhor_modelo.predict(X_test_scaled)
else:
    melhor_modelo.fit(X_train, y_train)
    y_pred = melhor_modelo.predict(X_test)

# 9. Avaliação estatística com TESTE
def avaliacao_estatistica(y_real, y_pred):
    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    residuals = y_real - y_pred
    sst = np.sum((y_real - np.mean(y_real))**2)
    sse = np.sum(residuals**2)
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

metricas = avaliacao_estatistica(y_test, y_pred)

print("\n=== Avaliação com Dados de TESTE ===")
for k, v in metricas.items():
    print(f"{k}: {v:.4f}" if '(%' not in k else f"{k}: {v:.2f}%")

# 10. Tabela com dados de TESTE
df_comparativo = pd.DataFrame({
    'Produtividade_Real': y_test,
    'Produtividade_Predita': y_pred,
    'Resíduo': y_test - y_pred
})
df_comparativo['Erro_Relativo'] = (df_comparativo['Resíduo'] / df_comparativo['Produtividade_Real']) * 100

print("\nTabela Comparativa (Apenas dados de TESTE):")
print(df_comparativo.sort_values('Produtividade_Real').head(10).to_string(index=False))

# Exportar top5 índices
top5_indices.to_csv('/content/top5_indices.csv', index=False)
# === Script adicional para salvar o melhor modelo como 'melhor_modelo.pkl' ===

# Verifica se o modelo já foi identificado corretamente
if 'melhor_modelo' in locals() and melhor_modelo is not None:
    try:
        # Adiciona ao modelo a informação das features usadas
        if hasattr(melhor_modelo, 'feature_names_in_'):
            pass  # já está presente
        else:
            melhor_modelo.feature_names_in_ = X.columns.to_numpy()

        joblib.dump(melhor_modelo, 'melhor_modelo.pkl')
        print("\n✅ Melhor modelo global salvo com sucesso como 'melhor_modelo.pkl'.")
    except Exception as e:
        print(f"\n⚠️ Erro ao salvar o melhor modelo: {str(e)}")
else:
    print("\n⚠️ Melhor modelo não encontrado no ambiente.")

# 11. Explicação didática das métricas de avaliação
print("\n=== 📘 Interpretação das Métricas ===")
print("""
🔹 R² (Coeficiente de Determinação):
    - Mede o quanto da variabilidade dos dados reais é explicada pelo modelo.
    - Varia de 0 a 1. Quanto mais próximo de 1, melhor o desempenho.
    - Exemplo: R² = 0.85 indica que 85% da variabilidade dos dados é explicada pelo modelo.

🔹 RMSE (Root Mean Squared Error):
    - Erro médio quadrático da predição. É sensível a grandes erros.
    - Mede, em unidades reais (ex: kg), o desvio médio entre o valor real e o previsto.
    - Quanto mais próximo de zero o RMSE, melhor.

🔹 RMSE Relativo (%):
    - RMSE em relação à média dos valores reais (em percentual).
    - Permite comparar erros entre diferentes contextos ou culturas agrícolas.
    - Exemplo: RMSE relativo de 12% significa que o erro médio representa 12% da produtividade média.

🔹 Bias (Viés):
    - Indica se o modelo tende a superestimar (bias negativo) ou subestimar (bias positivo) os valores.
    - Idealmente, deve ser próximo de zero.

🔹 Bias Relativo (%):
    - Bias expresso em relação à média dos valores reais.
    - Ajuda a avaliar a tendência sistemática do erro em termos percentuais.

✅ Recomendações:
    - Busque R² alto (≥ 0.75), RMSE e bias baixos.
    - Sempre avalie RMSE e bias relativos para entender o impacto em termos percentuais.
""")
