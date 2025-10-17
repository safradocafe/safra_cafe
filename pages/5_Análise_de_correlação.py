# pages/4_1. Análise_de_correlação.py
import os, glob, io, csv
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import shapiro, pearsonr

# =========================
# Página / estilo
# =========================
st.set_page_config(page_title="Análise de correlação", layout="wide")
st.markdown("## 📊 Análise de Correlação entre índices espectrais e produtividade")

BASE_TMP = "/tmp/streamlit_dados"

# =========================
# Utilitários
# =========================
def _find_latest_indices_csv(base=BASE_TMP):
    """Encontra o CSV mais recente gerado pela aba de Processamento/Previsão."""
    if not os.path.isdir(base):
        return None
    pats = glob.glob(os.path.join(base, "salvamento-*", "indices_espectrais_pontos_*.csv"))
    if not pats:
        return None
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pats[0]

def _sniff_delim_and_decimal(sample_bytes: bytes):
    """Infere delimitador e separador decimal (vírgula/ponto)."""
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
    """Lê CSV com detecção de separador/decimal e corrige casos de 'uma coluna só'."""
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
        # última tentativa
        df = pd.read_csv(io.BytesIO(raw))

    # caso tenha ficado tudo em 1 coluna, tenta o outro separador
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

TOKENS_IDX = ["NDVI", "GNDVI", "NDRE", "CCCI", "MSAVI2", "NDWI", "NDMI", "NBR", "TWI2"]

def _filter_corr_columns(df: pd.DataFrame):
    """Mantém maduro_kg e TODAS as colunas de índices (inclui _min/_mean/_max etc.)."""
    drop_cols = [c for c in ["Code", "latitude", "longitude", "geometry"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    idx_cols = [c for c in df.columns if any(tok in c for tok in TOKENS_IDX)]
    keep = (["maduro_kg"] if "maduro_kg" in df.columns else []) + idx_cols
    if not keep:
        return pd.DataFrame()

    # Coerção numérica
    df_num = df[keep].copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    # Remove linhas sem alvo
    if "maduro_kg" in df_num.columns:
        df_num = df_num.dropna(subset=["maduro_kg"], how="any")
    # Remove colunas totalmente NaN
    df_num = df_num.dropna(axis=1, how="all")
    return df_num

def _choose_method(df: pd.DataFrame):
    """Testa normalidade (Shapiro) e decide Pearson (se >50% normais) ou Spearman."""
    cols = [c for c in df.columns if c != "maduro_kg"]
    test_cols = ["maduro_kg"] + cols
    normal, tested = 0, 0
    for c in test_cols:
        s = df[c].dropna()
        if len(s) < 3:
            continue
        tested += 1
        try:
            _, p = shapiro(s)
            if p > 0.05:
                normal += 1
        except Exception:
            pass
    prop = (normal / tested) if tested else 0.0
    metodo = "pearson" if prop > 0.5 else "spearman"
    return metodo, prop

def _corr_top5(df: pd.DataFrame, metodo: str):
    cols = [c for c in df.columns if c != "maduro_kg"]
    vals = {}
    for c in cols:
        try:
            vals[c] = df[["maduro_kg", c]].corr(method=metodo).iloc[0, 1]
        except Exception:
            vals[c] = np.nan
    s = pd.Series(vals).dropna()
    top5 = s.abs().sort_values(ascending=False).head(5).index.tolist()
    return s, top5

def _pearson_pvals(df: pd.DataFrame, cols):
    """p-valores de Pearson para o Top 5 (informativo)."""
    pvals = {}
    a = pd.to_numeric(df["maduro_kg"], errors="coerce").dropna()
    for c in cols:
        b = pd.to_numeric(df[c], errors="coerce").dropna()
        m = min(len(a), len(b))
        if m < 3:
            pvals[c] = np.nan
            continue
        try:
            _, p = pearsonr(a.iloc[:m], b.iloc[:m])
            pvals[c] = p
        except Exception:
            pvals[c] = np.nan
    return pvals

# =========================
# 1) Carregamento do CSV (auto + upload)
# =========================
st.markdown("#### 1) Carregamento de dados")

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
        st.info("CSV enviado via upload foi carregado e será usado nesta análise.")
    except Exception as e:
        st.error(f"Falha ao ler o CSV enviado: {e}")

if df_raw is None or df_raw.empty:
    st.error("❌ Nenhum CSV válido foi encontrado/enviado. Gere o arquivo na aba **Previsão da safra**.")
    st.stop()

with st.expander("Pré-visualização do CSV (bruto)"):
    st.dataframe(df_raw.head(), use_container_width=True)

# =========================
# 2) Tratamento: mantém 'maduro_kg' + TODOS os índices
# =========================
df = _filter_corr_columns(df_raw)
if df.empty or "maduro_kg" not in df.columns:
    st.error("❌ Não foi possível encontrar 'maduro_kg' e/ou colunas de índices espectrais no CSV.")
    st.stop()

st.caption(f"Linhas após tratamento: **{len(df)}** | Colunas de análise: **{len(df.columns)}**")
with st.expander("Visualizar dados tratados (maduro_kg + TODOS os índices)"):
    st.dataframe(df.head(), use_container_width=True)

# =========================
# 3) Análise automática
# =========================
st.markdown("### 2) Análise estatística (automática)")

metodo, prop_norm = _choose_method(df)
st.write(f"**Método selecionado:** {'Pearson' if metodo=='pearson' else 'Spearman'} "
         f"(variáveis com normalidade: {prop_norm:.0%})")

# Matriz de correlação
corr = df.corr(method=metodo)
st.subheader("Matriz de correlação")
st.dataframe(corr.style.background_gradient(cmap="RdYlGn").format("{:.2f}"),
             use_container_width=True)

# Top 5
st.subheader("Top 5 correlações (|r|) com **maduro_kg**")
s_all, top5_cols = _corr_top5(df, metodo)
pvals = _pearson_pvals(df, top5_cols) if metodo == "pearson" else {}

for c in top5_cols:
    r = s_all[c]
    col1, col2 = st.columns([1, 4])
    with col1:
        st.metric(label=c, value=f"{r:.3f}")
    with col2:
        if metodo == "pearson":
            p = pvals.get(c, np.nan)
            if np.isnan(p):
                st.caption("p-valor: n/d")
            else:
                sig = "✅ Significativa" if p < 0.05 else "⚠️ Não significativa"
                st.caption(f"p-valor: {p:.4f} ({sig})")

with st.expander("📚 Como interpretar os resultados"):
    st.markdown("""
- **Pearson**: relação linear (pressupõe normalidade).  
- **Spearman**: relação monotônica (robusto a outliers, não exige normalidade).  
- **p-valor** (quando Pearson): < 0.05 sugere significância estatística.  
- Correlação **não** implica causalidade — use como etapa exploratória.
""")
