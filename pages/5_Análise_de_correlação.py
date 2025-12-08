# pages/4_1. Análise_de_correlação.py
import os, glob, io, csv
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import shapiro, pearsonr

st.set_page_config(page_title="Análise de correlação", layout="wide")
st.markdown("## 📊 Análise de Correlação entre índices espectrais e produtividade")

BASE_TMP = "/tmp/streamlit_dados"
TOKENS_IDX = ["NDVI", "GNDVI", "NDRE", "CCCI", "MSAVI2", "NDWI", "NDMI", "NBR", "TWI2"]

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
    try:
        dialect = csv.Sniffer().sniff(text[:10000], delimiters=[",", ";", "\t", "|"])
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
    """Lê CSV a partir de bytes com detecção de separador/decimal; cacheado por conteúdo."""
    delim, dec = _sniff_delim_and_decimal(raw)
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, decimal=dec, encoding=enc)
            break
        except Exception:
            pass
    if df is None:
        df = pd.read_csv(io.BytesIO(raw))
    
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

@st.cache_data(show_spinner=False)
def _prepare_for_corr_cached(raw: bytes):
    """Trata o CSV bruto e retorna df pronto para correlação (cacheado)."""
    df = _read_csv_robusto_cached(raw)
    
    drop_cols = [c for c in ["Code", "latitude", "longitude", "geometry"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    idx_cols = [c for c in df.columns if any(tok in c for tok in TOKENS_IDX)]
    keep = (["maduro_kg"] if "maduro_kg" in df.columns else []) + idx_cols
    df = df[keep].copy()
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
   
    if "maduro_kg" in df.columns:
        df = df.dropna(subset=["maduro_kg"], how="any")
    
    df = df.dropna(axis=1, how="all")
    return df

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

@st.cache_data(show_spinner=False)
def _compute_corr_all_cached(df_bytes: bytes):
    """
    Computa método, proporção de normalidade, matriz de correlação,
    série completa de correlações com maduro_kg, top5 e p-valores (se Pearson).
    Cacheia pelo conteúdo serializado do DF (bytes).
    """    
    df = _prepare_for_corr_cached(df_bytes)
    if df.empty or "maduro_kg" not in df.columns:
        return None

    metodo, prop_norm = _choose_method(df)
    corr = df.corr(method=metodo)
    
    cols = [c for c in df.columns if c != "maduro_kg"]
    vals = {}
    for c in cols:
        try:
            vals[c] = df[["maduro_kg", c]].corr(method=metodo).iloc[0, 1]
        except Exception:
            vals[c] = np.nan
    s_all = pd.Series(vals).dropna()
    top5_cols = s_all.abs().sort_values(ascending=False).head(5).index.tolist()

    pvals = {}
    if metodo == "pearson":
        a = pd.to_numeric(df["maduro_kg"], errors="coerce").dropna()
        for c in top5_cols:
            b = pd.to_numeric(df[c], errors="coerce").dropna()
            m = min(len(a), len(b))
            if m < 3:
                pvals[c] = np.nan
            else:
                try:
                    _, p = pearsonr(a.iloc[:m], b.iloc[:m])
                    pvals[c] = p
                except Exception:
                    pvals[c] = np.nan
    
    return {
        "method": metodo,
        "prop_norm": prop_norm,
        "corr": corr,
        "s_all": s_all,
        "top5": top5_cols,
        "pvals": pvals,
    }

st.markdown("#### 1) Carregamento de dados")

latest_csv_path = _find_latest_indices_csv()
raw_bytes = None

if latest_csv_path and os.path.exists(latest_csv_path):
    with open(latest_csv_path, "rb") as f:
        raw_bytes = f.read()
    st.success(f"✅ CSV carregado automaticamente")

up = st.file_uploader("Ou selecione manualmente o CSV gerado na aba de Processamento", type=["csv"])
if up is not None:
    raw_bytes = up.getvalue()
    st.info("CSV enviado via upload foi carregado e será usado nesta análise.")

if not raw_bytes:
    st.error("❌ Nenhum CSV válido foi encontrado/enviado. Gere o arquivo na aba **Previsão da safra**.")
    st.stop()

with st.expander("Pré-visualização do CSV (bruto)"):
    df_raw_preview = _read_csv_robusto_cached(raw_bytes)
    st.dataframe(df_raw_preview.head(), use_container_width=True)

df = _prepare_for_corr_cached(raw_bytes)
if df.empty or "maduro_kg" not in df.columns:
    st.error("❌ Não foi possível encontrar 'maduro_kg' e/ou colunas de índices espectrais no CSV.")
    st.stop()

st.caption(f"Linhas após tratamento: **{len(df)}** | Colunas de análise: **{len(df.columns)}**")
with st.expander("Visualizar dados tratados (maduro_kg + TODOS os índices)"):
    st.dataframe(df.head(), use_container_width=True)

st.markdown("### 2) Análise estatística (automática)")
res = _compute_corr_all_cached(raw_bytes)
if res is None:
    st.error("Não foi possível calcular as correlações.")
    st.stop()

metodo = res["method"]
prop_norm = res["prop_norm"]
corr = res["corr"]
s_all = res["s_all"]
top5_cols = res["top5"]
pvals = res["pvals"]

st.write(f"**Método selecionado:** {'Pearson' if metodo=='pearson' else 'Spearman'} "
         f"(variáveis com normalidade: {prop_norm:.0%})")

st.subheader("Matriz de correlação")
st.dataframe(corr.style.background_gradient(cmap="RdYlGn").format("{:.2f}"),
             use_container_width=True)

st.subheader("Top 5 correlações (|r|) com **maduro_kg**")
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
