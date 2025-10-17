# pages/4_2_Treinamento_com_Machine_Learning.py
import os, glob, io, csv, warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
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

# --------------------------
# Página / estilo
# --------------------------
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")
np.random.seed(42)

st.markdown("<h3>🧠 Treinamento e Avaliação de Modelos</h3>", unsafe_allow_html=True)
st.caption("Usa o CSV mais recente salvo na aba **Previsão da safra** (processamento).")

BASE_TMP = "/tmp/streamlit_dados"
TOKENS_IDX = ["NDVI", "GNDVI", "NDRE", "CCCI", "MSAVI2", "NDWI", "NDMI", "NBR", "TWI2"]

# --------------------------
# Utilitários (iguais à aba de correlação)
# --------------------------
def _find_latest_indices_csv(base=BASE_TMP):
    if not os.path.isdir(base):
        return None
    pats = glob.glob(os.path.join(base, "salvamento-*", "indices_espectrais_pontos_*.csv"))
    if not pats:
        return None
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pats[0]

def _sniff_delim_and_decimal(sample_bytes: bytes):
    text = sample_bytes.decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(text[:10000], delimiters=[",",";","\t","|"])
        delim = dialect.delimiter
    except Exception:
        delim = ";" if text.count(";") > text.count(",") else ","
    decimal = "."
    for line in text.splitlines()[:50]:
        if any(ch.isdigit() for ch in line):
            if "," in line and (delim == ";" or line.count(",") > line.count(".")):
                decimal = ","
                break
    return delim, decimal

@st.cache_data(show_spinner=False)
def _read_csv_robusto_cached(raw: bytes):
    delim, dec = _sniff_delim_and_decimal(raw)
    df = None
    for enc in ("utf-8","latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, decimal=dec, encoding=enc)
            break
        except Exception:
            pass
    if df is None:
        df = pd.read_csv(io.BytesIO(raw))
    if df.shape[1] == 1:
        other = ";" if delim == "," else ","
        for enc in ("utf-8","latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=other, decimal=dec, encoding=enc)
                break
            except Exception:
                pass
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(axis=1, how="all")

@st.cache_data(show_spinner=False)
def _prepare_training_df_cached(raw: bytes):
    df = _read_csv_robusto_cached(raw)
    drop_cols = [c for c in ["Code","latitude","longitude","geometry"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    idx_cols = [c for c in df.columns if any(tok in c for tok in TOKENS_IDX)]
    keep = (["maduro_kg"] if "maduro_kg" in df.columns else []) + idx_cols
    if not keep:
        return pd.DataFrame()
    df = df[keep].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["maduro_kg"], how="any")
    return df.dropna(axis=1, how="all")

# --------------------------
# Carregamento automático do CSV (com upload opcional)
# --------------------------
latest_csv_path = _find_latest_indices_csv()
raw_bytes = None

if latest_csv_path and os.path.exists(latest_csv_path):
    with open(latest_csv_path, "rb") as f:
        raw_bytes = f.read()
    st.success(f"✅ CSV mais recente carregado: `{latest_csv_path}`")

up = st.file_uploader("Ou selecione manualmente o CSV gerado na aba de Processamento", type=["csv"])
if up is not None:
    raw_bytes = up.getvalue()
    st.info("CSV enviado via upload será utilizado.")

if not raw_bytes:
    st.error("❌ Nenhum CSV válido encontrado/enviado. Gere o arquivo na aba **Previsão da safra**.")
    st.stop()

df = _prepare_training_df_cached(raw_bytes)
if df.empty or "maduro_kg" not in df.columns:
    st.error("❌ Não foi possível montar a base de treinamento (falta 'maduro_kg' ou índices).")
    st.stop()

with st.expander("Pré-visualização (base tratada para ML)"):
    st.dataframe(df.head(), use_container_width=True)

X = df.drop(columns=["maduro_kg"])
y = df["maduro_kg"]

# --------------------------
# Modelos
# --------------------------
modelos = {
    "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), activation="relu", solver="adam",
                        max_iter=2000, early_stopping=True, random_state=42),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Lasso": Lasso(alpha=0.1, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
}
modelos_escalonados = {"MLP","SVR","KNN","Ridge","Lasso","ElasticNet"}

# --------------------------
# Parâmetros e Treinamento
# --------------------------
st.sidebar.header("Parâmetros do Treinamento")
num_execucoes = st.sidebar.number_input(
    "Número de execuções", min_value=1, value=20,
    help="Quantas vezes cada modelo será treinado para buscar o melhor resultado (diferentes seeds)."
)

if st.sidebar.button("▶️ Iniciar Treinamento e Avaliação"):
    st.subheader("Análise dos Resultados")
    progress_bar = st.progress(0)
    status_text = st.empty()

    resultados_df = pd.DataFrame()
    melhores = {}  # guarda melhor execução por modelo
    TEST_SIZE = 0.30  # ✅ fixo (remoção da 'Proporção de teste')

    for i in range(num_execucoes):
        status_text.text(f"Executando modelos... (Execução {i + 1}/{num_execucoes})")
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=i)

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            for nome, modelo in modelos.items():
                Xtr, Xte = (X_tr_sc, X_te_sc) if nome in modelos_escalonados else (X_tr, X_te)
                try:
                    modelo.fit(Xtr, y_tr)
                    y_pred = modelo.predict(Xte)
                    r2  = r2_score(y_te, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

                    if (nome not in melhores) or (r2 > melhores[nome]["r2"]):
                        melhores[nome] = dict(modelo=modelo, r2=r2, rmse=rmse,
                                              exec=i+1, X_te=Xte, y_te=y_te)

                    resultados_df = pd.concat([resultados_df, pd.DataFrame([{
                        "execucao": i+1, "modelo": nome, "r2": r2, "rmse": rmse
                    }])], ignore_index=True)
                except Exception as e:
                    st.warning(f"Erro no modelo {nome} (execução {i+1}): {e}")
            progress_bar.progress((i + 1) / num_execucoes)
        except Exception as e:
            st.error(f"Erro na execução {i+1}: {e}")

    status_text.success("✅ Treinamento concluído!")
    st.subheader("Resultados de Todas as Execuções")
    with st.expander("Ver tabela de resultados"):
        st.dataframe(resultados_df, use_container_width=True)

    if resultados_df.empty:
        st.warning("Nenhum resultado de treinamento foi gerado.")
        st.stop()

    # Melhor modelo global
    df_best = pd.DataFrame([{"modelo": k, "r2": v["r2"], "rmse": v["rmse"]} for k, v in melhores.items()])
    best_name = df_best.loc[df_best["r2"].idxmax(), "modelo"]
    best_bundle = melhores[best_name]

    st.markdown("---")
    st.subheader("🏆 Melhor Modelo Global")
    c1, c2, c3 = st.columns(3)
    c1.metric("Modelo", best_name)
    c2.metric("Melhor R²", f"{best_bundle['r2']:.4f}")
    c3.metric("Melhor RMSE", f"{best_bundle['rmse']:.4f}")

    # Download do melhor modelo
    buffer = io.BytesIO()
    joblib.dump(best_bundle["modelo"], buffer)
    st.download_button(
        label=f"💾 Baixar o melhor modelo ({best_name}.pkl)",
        data=buffer.getvalue(),
        file_name=f"melhor_modelo_{best_name}.pkl",
        mime="application/octet-stream",
    )

    # ❌ REMOVIDO: Importância das Features (Permutation Importance)

    # Avaliação com TESTE
    st.markdown("---")
    st.subheader("Avaliação com Dados de TESTE")

    y_te_final = best_bundle["y_te"]
    y_pred_final = best_bundle["modelo"].predict(best_bundle["X_te"])

    def avaliacao_estatistica(y_real, y_pred):
        r2   = r2_score(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        resid = y_real - y_pred
        rmse_rel = (rmse / np.mean(y_real)) * 100
        bias = np.mean(resid)
        bias_rel = (bias / np.mean(y_real)) * 100
        return {"R²": r2, "RMSE": rmse, "RMSE Relativo (%)": rmse_rel,
                "Bias": bias, "Bias Relativo (%)": bias_rel}

    metr = avaliacao_estatistica(y_te_final, y_pred_final)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R²", f"{metr['R²']:.4f}")
    m2.metric("RMSE", f"{metr['RMSE']:.4f}")
    m3.metric("RMSE Relativo (%)", f"{metr['RMSE Relativo (%)']:.2f}%")
    m4.metric("Bias", f"{metr['Bias']:.4f}")
    m5.metric("Bias Relativo (%)", f"{metr['Bias Relativo (%)']:.2f}%")

    st.markdown("---")
    st.subheader("Comparativo: Real vs. Predito (Teste)")
    df_cmp = pd.DataFrame({
        "Produtividade_Real": y_te_final,
        "Produtividade_Predita": y_pred_final,
        "Resíduo": y_te_final - y_pred_final
    }).reset_index(drop=True)
    df_cmp["Erro_Relativo_%"] = (df_cmp["Resíduo"] / df_cmp["Produtividade_Real"]) * 100

    st.dataframe(df_cmp.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df_cmp["Produtividade_Real"], df_cmp["Produtividade_Predita"], alpha=0.7)
    rng = [min(df_cmp["Produtividade_Real"].min(), df_cmp["Produtividade_Predita"].min()),
           max(df_cmp["Produtividade_Real"].max(), df_cmp["Produtividade_Predita"].max())]
    ax.plot(rng, rng, "r--", lw=2, label="Linha 1:1")
    ax.set_xlabel("Produtividade Real (kg)")
    ax.set_ylabel("Produtividade Predita (kg)")
    ax.set_title("Produtividade Predita vs. Real (Teste)")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
with st.expander("📘 Interpretação das Métricas"):
    st.markdown("""
- **R²**: quanto da variabilidade é explicada (0–1; maior é melhor).  
- **RMSE**: erro quadrático médio (unidades do alvo).  
- **RMSE Relativo**: RMSE em % da média do real.  
- **Bias**: viés médio (positivo = subestima; negativo = superestima).  
""")
