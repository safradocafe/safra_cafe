# pages/4_2_Treinamento_ML.py
import os, glob, io, csv, random, joblib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
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

# =========================
# Página / estilo
# =========================
st.set_page_config(layout="wide")
st.markdown("## 🧠 Treinamento e Avaliação de Modelos (20 execuções por algoritmo)")
st.caption("Carrega o CSV mais recente salvo na nuvem, treina 11 modelos 20x cada, guarda o melhor de cada e destaca o melhor global.")

BASE_TMP = "/tmp/streamlit_dados"
NUM_EXECUCOES = 20
GLOBAL_SEED = 42

# Fixar seeds globais (para consistência entre libs que usam numpy/random)
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

TOKENS_IDX = ["NDVI", "GNDVI", "NDRE", "CCCI", "MSAVI2", "NDWI", "NDMI", "NBR", "TWI2"]

# =========================
# Utilitários de dados
# =========================
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
    # delimitador
    try:
        dialect = csv.Sniffer().sniff(text[:10000], delimiters=[",", ";", "\t", "|"])
        delim = dialect.delimiter
    except Exception:
        delim = ";" if text.count(";") > text.count(",") else ","
    # decimal
    decimal = "."
    for line in text.splitlines()[:50]:
        if any(ch.isdigit() for ch in line):
            if "," in line and (delim == ";" or line.count(",") > line.count(".")):
                decimal = ","
                break
    return delim, decimal

@st.cache_data(show_spinner=False)
def _read_csv_robusto(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read()
    delim, dec = _sniff_delim_and_decimal(raw)
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, decimal=dec, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(io.BytesIO(raw))  # tentativa final

    if df.shape[1] == 1:
        other = ";" if delim == "," else ","
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=other, decimal=dec, encoding=enc)
                break
            except Exception:
                pass

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df

def _filter_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Remove colunas não usadas e mantém 'maduro_kg' + TODOS os índices espectrais (min, mean, max, etc.)
    drop_cols = [c for c in ["Code", "latitude", "longitude", "geometry"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    idx_cols = [c for c in df.columns if any(tok in c for tok in TOKENS_IDX)]
    keep = (["maduro_kg"] if "maduro_kg" in df.columns else []) + idx_cols
    df = df[keep].copy()

    # Coerção numérica
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Exige alvo presente
    df = df.dropna(subset=["maduro_kg"], how="any")
    # Remove colunas totalmente NaN
    df = df.dropna(axis=1, how="all")
    return df

# =========================
# 1) Carregamento do CSV mais recente (auto)
# =========================
latest_csv_path = _find_latest_indices_csv()
if not latest_csv_path or not os.path.exists(latest_csv_path):
    st.error("❌ CSV de índices não encontrado automaticamente. Gere-o na aba **Previsão da safra**.")
    st.stop()

df_raw = _read_csv_robusto(latest_csv_path)
df = _filter_training_columns(df_raw)

if df.empty or "maduro_kg" not in df.columns:
    st.error("❌ Não foi possível encontrar 'maduro_kg' e colunas de índices no CSV carregado.")
    st.stop()

st.success(f"✅ CSV carregado: `{latest_csv_path}`")
with st.expander("Pré-visualização (dados tratados para ML)"):
    st.dataframe(df.head(), use_container_width=True)

X = df.drop(columns=["maduro_kg"])
y = df["maduro_kg"].copy()

# =========================
# 2) Definição dos modelos (padrão solicitado)
# =========================
modelos = {
    "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam',
                        max_iter=2000, early_stopping=True, random_state=GLOBAL_SEED),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=GLOBAL_SEED),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=GLOBAL_SEED),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=GLOBAL_SEED),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=GLOBAL_SEED),
    "DecisionTree": DecisionTreeRegressor(random_state=GLOBAL_SEED),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Ridge": Ridge(alpha=1.0, random_state=GLOBAL_SEED),
    "Lasso": Lasso(alpha=0.1, random_state=GLOBAL_SEED),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=GLOBAL_SEED),
}
modelos_escalonados = {"MLP", "SVR", "KNN", "Ridge", "Lasso", "ElasticNet"}

# =========================
# 3) Treinamento 20x por modelo (cacheado na sessão)
# =========================
def _run_training_once(X: pd.DataFrame, y: pd.Series):
    resultados = {nome: [] for nome in modelos.keys()}
    progress = st.progress(0.0)
    status = st.empty()

    total_iters = NUM_EXECUCOES * len(modelos)

    k = 0
    for i in range(NUM_EXECUCOES):
        status.text(f"Executando modelos... (execução {i+1}/{NUM_EXECUCOES})")
        # Split muda a cada i, como no seu padrão
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=i
        )
        # scaler para os modelos que precisam
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        for nome, modelo in modelos.items():
            if nome in modelos_escalonados:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test

            try:
                modelo.fit(X_tr, y_train)
                y_pred = modelo.predict(X_te)
                r2  = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                resultados[nome].append({
                    "execucao": i + 1,
                    "r2": r2,
                    "rmse": rmse,
                    "convergiu": True
                })
            except Exception:
                resultados[nome].append({
                    "execucao": i + 1,
                    "r2": np.nan,
                    "rmse": np.nan,
                    "convergiu": False
                })
            k += 1
            progress.progress(min(1.0, k/total_iters))

    status.success("✅ Treinamento concluído!")
    return resultados

if "ml_results" not in st.session_state:
    st.session_state["ml_results"] = _run_training_once(X, y)

resultados = st.session_state["ml_results"]

# =========================
# 4) Melhor de cada modelo + melhor global
# =========================
melhores_modelos = {}
linhas_best = []
for nome, lst in resultados.items():
    dfm = pd.DataFrame(lst)
    dfm = dfm[dfm["convergiu"]].dropna(subset=["r2"])
    if dfm.empty:
        continue
    idx = dfm["r2"].idxmax()
    best = dfm.loc[idx].to_dict()
    melhores_modelos[nome] = best
    linhas_best.append({"modelo": nome, "execucao": int(best["execucao"]), "r2": best["r2"], "rmse": best["rmse"]})

df_melhores = pd.DataFrame(linhas_best).sort_values("r2", ascending=False).reset_index(drop=True)

st.markdown("### 🏁 Melhor resultado de **cada** modelo (entre 20 execuções)")
st.dataframe(df_melhores.style.format({"r2": "{:.4f}", "rmse": "{:.4f}"}), use_container_width=True)

if df_melhores.empty:
    st.error("Nenhum modelo convergiu.")
    st.stop()

melhor_modelo_nome = df_melhores.iloc[0]["modelo"]
melhor_exec       = int(df_melhores.iloc[0]["execucao"])
st.success(f"🏆 **Melhor modelo global:** `{melhor_modelo_nome}` (execução {melhor_exec})")

# =========================
# 5) Reproduzir a melhor execução e avaliar no TESTE
# =========================
# Recria split com o random_state da melhor execução - 1 (porque execucao = i+1)
rs = melhor_exec - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=rs)

# recria o modelo "do zero" com os mesmos hiperparâmetros
def _make_model(name: str):
    return {
        "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam',
                            max_iter=2000, early_stopping=True, random_state=GLOBAL_SEED),
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=GLOBAL_SEED),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=GLOBAL_SEED),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=GLOBAL_SEED),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=GLOBAL_SEED),
        "DecisionTree": DecisionTreeRegressor(random_state=GLOBAL_SEED),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Ridge": Ridge(alpha=1.0, random_state=GLOBAL_SEED),
        "Lasso": Lasso(alpha=0.1, random_state=GLOBAL_SEED),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=GLOBAL_SEED),
    }[name]

best_model = _make_model(melhor_modelo_nome)

if melhor_modelo_nome in {"MLP", "SVR", "KNN", "Ridge", "Lasso", "ElasticNet"}:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

# métricas
r2  = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
residuals = y_test - y_pred
rmse_rel = (rmse / y_test.mean()) * 100
bias = residuals.mean()
bias_rel = (bias / y_test.mean()) * 100

st.markdown("### 📏 Avaliação (dados de TESTE) do melhor modelo global")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("R²", f"{r2:.4f}")
c2.metric("RMSE", f"{rmse:.4f}")
c3.metric("RMSE Rel. (%)", f"{rmse_rel:.2f}%")
c4.metric("Bias", f"{bias:.4f}")
c5.metric("Bias Rel. (%)", f"{bias_rel:.2f}%")

# comparativo e gráfico
df_cmp = pd.DataFrame({
    "Produtividade_Real": y_test,
    "Produtividade_Predita": y_pred,
    "Resíduo": residuals
}).reset_index(drop=True)
df_cmp["Erro_Relativo_%"] = (df_cmp["Resíduo"] / df_cmp["Produtividade_Real"]) * 100

st.markdown("#### Comparativo (amostra do TESTE)")
st.dataframe(df_cmp.sort_values("Produtividade_Real").head(10).round(4), use_container_width=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df_cmp["Produtividade_Real"], df_cmp["Produtividade_Predita"], alpha=0.7)
low = float(min(df_cmp["Produtividade_Real"].min(), df_cmp["Produtividade_Predita"].min()))
high= float(max(df_cmp["Produtividade_Real"].max(), df_cmp["Produtividade_Predita"].max()))
ax.plot([low, high], [low, high], "r--", lw=2, label="Linha 1:1")
ax.set_xlabel("Produtividade Real (kg)")
ax.set_ylabel("Produtividade Predita (kg)")
ax.set_title(f"Produtividade Predita vs. Real (Teste) — {melhor_modelo_nome}")
ax.legend()
st.pyplot(fig, use_container_width=True)

# =========================
# 6) Salvar o melhor modelo (pickle)
# =========================
# anexa nomes de features para uso posterior
try:
    best_model.feature_names_in_ = X.columns.to_numpy()
except Exception:
    pass

# se for modelo escalonado, também salva o scaler
bundle = {"model": best_model, "features": list(X.columns)}
if melhor_modelo_nome in {"MLP", "SVR", "KNN", "Ridge", "Lasso", "ElasticNet"}:
    bundle["scaler"] = scaler

out_path = os.path.join(BASE_TMP, f"melhor_modelo_{melhor_modelo_nome}.pkl")
os.makedirs(BASE_TMP, exist_ok=True)
joblib.dump(bundle, out_path)

st.download_button(
    "💾 Baixar melhor modelo (PKL)",
    data=open(out_path, "rb").read(),
    file_name=f"melhor_modelo_{melhor_modelo_nome}.pkl",
    mime="application/octet-stream"
)

st.caption(f"Modelo salvo temporariamente em: `{out_path}`")
