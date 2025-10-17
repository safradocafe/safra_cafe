import os, glob, re
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import shapiro, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(page_title="Análise de correlação", layout="wide")
st.title("📊 Análise de Correlação entre índices espectrais e produtividade")

BASE_TMP = "/tmp/streamlit_dados"
IDX_NAMES = ["NDVI","GNDVI","NDRE","CCCI","MSAVI2","NDWI","NDMI","NBR","TWI2"]
STAT_OPTS = {"Mínimo":"min", "Médio":"mean", "Máximo":"max"}

# -----------------------------
# Utilitários
# -----------------------------
def _find_latest_save_dir(base=BASE_TMP):
    if not os.path.isdir(base): return None
    cands = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not cands: return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

@st.cache_data(show_spinner=False)
def load_latest_csv():
    """Localiza o CSV mais recente indices_espectrais_pontos_*.csv nos salvamentos."""
    latest = _find_latest_save_dir()
    if not latest:
        return None, None
    csvs = sorted(glob.glob(os.path.join(latest, "indices_espectrais_pontos_*.csv")))
    if not csvs:
        return None, latest
    csv_path = csvs[-1]
    df = pd.read_csv(csv_path)
    return (df, csv_path)

def parse_dates_from_columns(columns):
    """Extrai datas únicas do padrão 'YYYY-MM-DD_IDX_stat'."""
    dates = set()
    rx = re.compile(r"^(\d{4}-\d{2}-\d{2})_([A-Z0-9]+)_(min|mean|max)$")
    for c in columns:
        m = rx.match(c)
        if m:
            dates.add(m.group(1))
    return sorted(list(dates))

def build_matrix(df_raw, stat="mean", date_choice="Média de todas as datas"):
    """
    Retorna um DataFrame com colunas:
      maduro_kg + {NDVI, GNDVI, ...} de acordo com 'stat' e 'date_choice'.
    - Se date_choice == 'Média de todas as datas': calcula a média por ponto de todas as colunas daquele índice e 'stat'.
    - Se date_choice é uma data específica: usa somente as colunas daquela data.
    """
    df = df_raw.copy()

    # Remove colunas não usadas
    for dropcol in ["Code", "latitude", "longitude"]:
        if dropcol in df.columns:
            df = df.drop(columns=[dropcol])

    # Garante maduro_kg
    if "maduro_kg" not in df.columns:
        raise ValueError("Coluna 'maduro_kg' não encontrada no CSV!")

    out = pd.DataFrame(index=df.index)
    out["maduro_kg"] = df["maduro_kg"]

    if date_choice == "Média de todas as datas":
        # Para cada índice, calcula a média (por linha) de todas as colunas *_IDX_stat
        for idx in IDX_NAMES:
            cols_idx = [c for c in df.columns if c.endswith(f"_{idx}_{stat}")]
            # Observação: no processamento, o nome é 'YYYY-MM-DD_IDX_stat'
            # então endswith é suficiente para pegar todas as datas.
            if not cols_idx:
                continue
            out[idx] = df[cols_idx].mean(axis=1, skipna=True)
    else:
        # Usa somente as colunas da data específica
        prefix = f"{date_choice}_"
        for idx in IDX_NAMES:
            col_name = f"{prefix}{idx}_{stat}"
            if col_name in df.columns:
                out[idx] = df[col_name]
            # se não existir essa banda nessa data, deixa como NaN (col não criada)

    # remove colunas totalmente vazias (caso algum índice não exista para aquela data)
    keep_cols = ["maduro_kg"] + [c for c in out.columns if c != "maduro_kg" and not out[c].isna().all()]
    out = out[keep_cols]
    return out

# -----------------------------
# Carregamento de dados
# -----------------------------
st.header("1) Carregamento do CSV processado")
df_raw, csv_path = load_latest_csv()

col_up1, col_up2 = st.columns([2, 3])
with col_up1:
    if df_raw is not None:
        st.success(f"✅ CSV localizado: `{csv_path}`")
    else:
        st.warning("❌ CSV de índices não encontrado automaticamente.")
with col_up2:
    up = st.file_uploader("Ou selecione um CSV manualmente", type=["csv"])
    if up is not None:
        df_raw = pd.read_csv(up)
        csv_path = "(upload)"

if df_raw is None:
    st.stop()

with st.expander("👀 Amostra dos dados brutos (do CSV)", expanded=False):
    # mostra sem truncar colunas em excesso
    st.dataframe(df_raw.head(), use_container_width=True)

# -----------------------------
# Seleções de análise
# -----------------------------
st.header("2) Preparar matriz para correlação")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    stat_label = st.radio("Estatística dos índices", list(STAT_OPTS.keys()), index=1, horizontal=True)
    stat = STAT_OPTS[stat_label]

with c2:
    # Datas disponíveis
    dates = parse_dates_from_columns(df_raw.columns)
    date_choice = st.selectbox("Data a analisar", ["Média de todas as datas"] + dates, index=0)

with c3:
    dropna = st.checkbox("Remover linhas com NaN", value=True)

# Monta a matriz X/y conforme escolhas
try:
    df = build_matrix(df_raw, stat=stat, date_choice=date_choice)
except Exception as e:
    st.error(f"Erro ao montar a matriz de análise: {e}")
    st.stop()

if dropna:
    df = df.dropna()

if df.empty or df.shape[1] < 2:
    st.warning("Sem dados suficientes após o tratamento. Ajuste as opções acima.")
    st.stop()

with st.expander("👀 Amostra da matriz tratada (somente maduro_kg + índices)", expanded=False):
    st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# 3) Teste de normalidade + método
# -----------------------------
st.header("3) Teste de normalidade e método de correlação")
try:
    resultados_normalidade = []
    for col in df.columns:
        # shapiro exige >=3 valores
        series = df[col].dropna()
        if len(series) >= 3:
            stat_v, p_v = shapiro(series)
            normal = p_v > 0.05
            resultados_normalidade.append({"Variável": col, "p-valor": f"{p_v:.4f}", "Normal": "Sim" if normal else "Não"})
        else:
            resultados_normalidade.append({"Variável": col, "p-valor": "n/a", "Normal": "n/a"})

    df_norm = pd.DataFrame(resultados_normalidade)
    st.dataframe(df_norm, use_container_width=True)

    proporcao_normal = (df_norm["Normal"] == "Sim").mean()
    metodo = "pearson" if proporcao_normal > 0.5 else "spearman"
    st.success(f"Método selecionado: **{metodo.capitalize()}**")
except Exception as e:
    st.error(f"Erro no teste de normalidade: {e}")
    st.stop()

# -----------------------------
# 4) Correlações
# -----------------------------
st.header("4) Correlações com a produtividade (maduro_kg)")

try:
    # Matriz completa
    corr = df.corr(method=metodo)
    # Série maduro vs índices (exclui a própria maduro_kg)
    idx_cols = [c for c in df.columns if c != "maduro_kg"]
    serie_corr = pd.Series({col: corr.loc["maduro_kg", col] for col in idx_cols if col in corr.columns})

    # p-valores (apenas Pearson)
    pvals = {}
    if metodo == "pearson":
        for col in idx_cols:
            common = df[["maduro_kg", col]].dropna()
            if len(common) >= 3:
                _, p_val = pearsonr(common["maduro_kg"], common[col])
                pvals[col] = p_val

    # Top 5 absolutos
    top5 = serie_corr.abs().sort_values(ascending=False).head(5).index.tolist()

    # Métricas em cards
    st.subheader("Top 5 correlações (|r|)")
    for col in top5:
        r = serie_corr[col]
        c1, c2 = st.columns([1, 4])
        with c1:
            st.metric(label=col, value=f"{r:.3f}", help="Positiva" if r > 0 else "Negativa")
        with c2:
            if metodo == "pearson" and col in pvals:
                p = pvals[col]
                sig = "✅ Significativa (p<0.05)" if p < 0.05 else "⚠️ Não significativa (p≥0.05)"
                st.caption(f"p-valor: {p:.4f} — {sig}")

    # Heatmap
    st.subheader("Heatmap de correlação (matriz completa)")
    fig, ax = plt.subplots(figsize=(min(12, 1.2*len(corr.columns)), 7))
    sns.heatmap(corr, cmap="RdYlGn", vmin=-1, vmax=1, annot=False, square=False, ax=ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Dispersões para os Top 3 (opcional)
    st.subheader("Dispersões (Top 3)")
    for col in top5[:3]:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.scatter(df[col], df["maduro_kg"], alpha=0.7)
        ax2.set_xlabel(col)
        ax2.set_ylabel("maduro_kg")
        ax2.grid(True, alpha=.3)
        st.pyplot(fig2, use_container_width=False)

except Exception as e:
    st.error(f"Erro no cálculo de correlações: {e}")
