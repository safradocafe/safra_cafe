# pages/2_Treinamento.py

# pages/4_2. Treinamento_com_Machine_Learning.py
import os, glob, io, csv, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# Modelos
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# =========================
# Página / estilo
# =========================
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")
np.random.seed(42)

st.markdown("<h3>🧠 Treinamento e avaliação dos modelos</h3>", unsafe_allow_html=True)
st.caption("Usa o CSV salvo na aba **Previsão da safra** (mín/méd/máx por ponto e data) e treina modelos de regressão para prever `maduro_kg`.")

BASE_TMP = "/tmp/streamlit_dados"
MODEL_OUT = os.path.join(BASE_TMP, "melhor_modelo.pkl")

TOKENS_IDX = ["NDVI", "GNDVI", "NDRE", "CCCI", "MSAVI2", "NDWI", "NDMI", "NBR", "TWI2"]

# =========================
# Utilitários CSV (mesmos da aba de correlação)
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

def _read_csv_robusto(path_or_bytes):
    if isinstance(path_or_bytes, (str, os.PathLike)):
        with open(path_or_bytes, "rb") as f:
            raw = f.read()
    else:
        raw = path_or_bytes

    delim, dec = _sniff_delim_and_decimal(raw)
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, decimal=dec, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(io.BytesIO(raw))

    # se ficou 1 coluna só, tenta outro separador
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

def _filter_ml_columns(df: pd.DataFrame):
    """Mantém 'maduro_kg' + TODAS as colunas de índices (inclui _min/_mean/_max por data) e torna tudo numérico."""
    # Remove colunas não usadas
    df = df.drop(columns=[c for c in ["Code", "latitude", "longitude", "geometry"] if c in df.columns],
                 errors="ignore")

    # Seleciona índices (tudo que contenha os tokens)
    idx_cols = [c for c in df.columns if any(tok in c for tok in TOKENS_IDX)]
    cols = (["maduro_kg"] if "maduro_kg" in df.columns else []) + idx_cols
    df = df[cols].copy()

    # Coerção numérica
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove linhas sem alvo
    if "maduro_kg" in df.columns:
        df = df.dropna(subset=["maduro_kg"], how="any")

    # Remove colunas totalmente NaN
    df = df.dropna(axis=1, how="all")

    return df

# =========================
# 1) Carregar CSV (auto + upload)
# =========================
st.markdown("#### 1) Dados de treinamento")
latest_csv_path = _find_latest_indices_csv()

df_raw = None
if latest_csv_path and os.path.exists(latest_csv_path):
    try:
        df_raw = _read_csv_robusto(latest_csv_path)
        st.success(f"✅ CSV carregado automaticamente: `{latest_csv_path}`")
    except Exception as e:
        st.error(f"Falha ao ler o CSV automático: {e}")

up = st.file_uploader("Ou selecione manualmente o CSV gerado na aba de Processamento", type=["csv"])
if up is not None:
    try:
        df_raw = _read_csv_robusto(up.getvalue())
        st.info("CSV enviado via upload foi carregado e será usado.")
    except Exception as e:
        st.error(f"Falha ao ler o CSV enviado: {e}")

if df_raw is None or df_raw.empty:
    st.error("❌ Nenhum CSV válido foi encontrado/enviado. Gere o arquivo na aba **Previsão da safra**.")
    st.stop()

with st.expander("Pré-visualização (bruto)"):
    st.dataframe(df_raw.head(), use_container_width=True)

# =========================
# 2) Preparação dos dados
# =========================
df = _filter_ml_columns(df_raw)
if df.empty or "maduro_kg" not in df.columns:
    st.error("❌ Não foi possível encontrar 'maduro_kg' e/ou colunas de índices espectrais no CSV.")
    st.stop()

st.caption(f"Linhas para treinamento: **{len(df)}** | Features: **{len(df.columns)-1}**")
with st.expander("Visualizar dados tratados (maduro_kg + todos os índices)"):
    st.dataframe(df.head(), use_container_width=True)

y = df["maduro_kg"].copy()
X = df.drop(columns=["maduro_kg"]).copy()

# =========================
# 3) Definição dos modelos
# =========================
modelos = {
    "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam',
                        max_iter=2000, early_stopping=True, random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=300, random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
}
if XGB_OK:
    modelos["XGBoost"] = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )

modelos_escalonados = ["MLP", "SVR", "KNN", "Ridge", "Lasso", "ElasticNet"]

# =========================
# 4) Parâmetros e execução
# =========================
st.sidebar.header("Parâmetros do Treinamento")
num_execucoes = st.sidebar.number_input("Número de execuções", min_value=1, value=20,
                                        help="Cada execução muda o random_state do split.")
test_size = st.sidebar.slider("Proporção de teste", 0.1, 0.5, 0.3, 0.05)
start_btn = st.sidebar.button("▶️ Iniciar Treinamento e Avaliação")

if start_btn:
    st.subheader("Análise dos Resultados")
    progress_bar = st.progress(0)
    status_text = st.empty()

    resultados_df = pd.DataFrame()
    melhores = {}

    for i in range(num_execucoes):
        status_text.text(f"Executando modelos... (Execução {i + 1}/{num_execucoes})")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            for nome, modelo in modelos.items():
                X_tr, X_te = (X_train_s, X_test_s) if nome in modelos_escalonados else (X_train, X_test)

                try:
                    modelo.fit(X_tr, y_train)
                    y_pred = modelo.predict(X_te)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    # guarda melhor por modelo
                    if (nome not in melhores) or (r2 > melhores[nome]['r2']):
                        melhores[nome] = {
                            'modelo': modelo,
                            'r2': r2,
                            'rmse': rmse,
                            'execucao': i + 1,
                            'X_tr': X_tr, 'y_tr': y_train,
                            'X_te': X_te, 'y_te': y_test,
                            'scaler_used': (nome in modelos_escalonados),
                            'feature_names': list(X.columns)
                        }

                    resultados_df = pd.concat([resultados_df, pd.DataFrame([{
                        'execucao': i + 1, 'modelo': nome, 'r2': r2, 'rmse': rmse
                    }])], ignore_index=True)

                except Exception as e:
                    st.warning(f"Erro no modelo {nome} (execução {i+1}): {str(e)}")

            progress_bar.progress((i + 1) / num_execucoes)

        except Exception as e:
            st.error(f"Erro na execução {i + 1}: {str(e)}")

    status_text.success("✅ Treinamento concluído!")

    # -------------------------
    # Resultados agregados
    # -------------------------
    st.subheader("Resultados de todas as execuções")
    with st.expander("Ver tabela de resultados"):
        st.dataframe(resultados_df, use_container_width=True)

    if not resultados_df.empty and len(melhores) > 0:
        df_best = pd.DataFrame([{"modelo": k, "r2": v['r2'], "rmse": v['rmse']} for k, v in melhores.items()])
        best_name = df_best.loc[df_best["r2"].idxmax(), "modelo"]
        best = melhores[best_name]

        st.markdown("---")
        st.subheader("🏆 Melhor Modelo Global")
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelo", best_name)
        c2.metric("Melhor R²", f"{best['r2']:.4f}")
        c3.metric("Melhor RMSE", f"{best['rmse']:.4f}")

        # Download do melhor modelo
        # (Opcional: persistir no /tmp também)
        os.makedirs(BASE_TMP, exist_ok=True)
        joblib.dump(best['modelo'], MODEL_OUT)

        buf = io.BytesIO()
        joblib.dump(best['modelo'], buf)
        st.download_button(
            label=f"💾 Baixar melhor modelo ({best_name}.pkl)",
            data=buf.getvalue(),
            file_name=f"melhor_modelo_{best_name}.pkl",
            mime="application/octet-stream"
        )
        st.caption(f"Modelo salvo temporariamente em: `{MODEL_OUT}`")

        # Importância das features (permuta)
        st.markdown("---")
        st.subheader("Importância das Features (Permutation Importance)")
        try:
            imp = permutation_importance(best['modelo'], best['X_tr'], best['y_tr'],
                                         n_repeats=10, random_state=42)
            imp_df = pd.DataFrame({
                "feature": best['feature_names'],
                "importance": imp.importances_mean
            }).sort_values("importance", ascending=False)

            top5 = imp_df.head(5).copy()
            total = top5["importance"].sum()
            top5["%"] = np.where(total > 0, 100 * top5["importance"] / total, 0)

            st.dataframe(top5.round(4), use_container_width=True)

            csv_top5 = top5.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Baixar TOP-5 (CSV)", data=csv_top5,
                               file_name="top5_indices.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Não foi possível calcular a importância por permutação: {e}")

        # Avaliação no TESTE
        st.markdown("---")
        st.subheader("Avaliação no conjunto de TESTE")

        y_te = best['y_te']
        y_pred = best['modelo'].predict(best['X_te'])

        def eval_stats(y_real, y_pred):
            r2 = r2_score(y_real, y_pred)
            rmse = np.sqrt(mean_squared_error(y_real, y_pred))
            residuals = y_real - y_pred
            rmse_rel = (rmse / np.mean(y_real)) * 100 if np.mean(y_real) else np.nan
            bias = np.mean(residuals)
            bias_rel = (bias / np.mean(y_real)) * 100 if np.mean(y_real) else np.nan
            return r2, rmse, rmse_rel, bias, bias_rel

        r2v, rmsev, rmser, bias, biasr = eval_stats(y_te, y_pred)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("R²", f"{r2v:.4f}")
        c2.metric("RMSE", f"{rmsev:.4f}")
        c3.metric("RMSE Rel. (%)", f"{rmser:.2f}%")
        c4.metric("Bias", f"{bias:.4f}")
        c5.metric("Bias Rel. (%)", f"{biasr:.2f}%")

        # Scatter real vs predito
        st.markdown("---")
        st.subheader("Real vs Predito (Teste)")

        df_cmp = pd.DataFrame({
            "Real": y_te,
            "Predito": y_pred,
            "Resíduo": y_te - y_pred
        }).reset_index(drop=True)
        df_cmp["Erro_rel_%"] = np.where(df_cmp["Real"] != 0,
                                        100 * df_cmp["Resíduo"] / df_cmp["Real"], np.nan)
        st.dataframe(df_cmp.head(20).round(4), use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(df_cmp["Real"], df_cmp["Predito"], alpha=0.75)
        mi, ma = float(np.nanmin(df_cmp["Real"])), float(np.nanmax(df_cmp["Real"]))
        ax.plot([mi, ma], [mi, ma], "r--", lw=2, label="Linha 1:1")
        ax.set_xlabel("Produtividade Real (kg)"); ax.set_ylabel("Produtividade Predita (kg)")
        ax.set_title(f"{best_name} — Teste")
        ax.legend(); ax.grid(alpha=.3)
        st.pyplot(fig, use_container_width=True)

    else:
        st.warning("⚠️ Nenhum resultado de treinamento foi gerado. Verifique os dados.")
else:
    st.info("Defina os parâmetros no painel à esquerda e clique em **Iniciar Treinamento e Avaliação**.")

# Explicação didática das métricas
st.markdown("---")
with st.expander("📘 Interpretação das métricas"):
    st.markdown("""
🔹 **R² (Coeficiente de determinação):**
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
