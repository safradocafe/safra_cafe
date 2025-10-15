import os, glob, json
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.markdown("<h3>🔗 Análise de correlação</h3>", unsafe_allow_html=True)

BASE_TMP = "/tmp/streamlit_dados"

def _find_latest_save_dir(base=BASE_TMP):
    if not os.path.isdir(base): return None
    cands = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not cands: return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

@st.cache_data(show_spinner=False)
def load_points():
    latest = _find_latest_save_dir()
    if not latest: return None, None
    pts_path = os.path.join(latest, "pontos_produtividade.gpkg")
    gdf = gpd.read_file(pts_path) if os.path.exists(pts_path) else None
    return gdf, latest

gdf_pts, latest_dir = load_points()
if gdf_pts is None or gdf_pts.empty:
    st.warning("Sem pontos encontrados. Volte à seção 1 e clique em **Salvar dados**.")
    st.stop()

st.caption(f"Origem dos dados: `{latest_dir}`")
num_cols = [c for c in gdf_pts.columns if pd.api.types.is_numeric_dtype(gdf_pts[c])]

c1, c2 = st.columns([1.2, 1])
with c1:
    y = st.selectbox("Alvo (y)", ["maduro_kg"] + [c for c in num_cols if c != "maduro_kg"], key="corr_y")
    X = st.multiselect("Variáveis explicativas (X)", [c for c in num_cols if c != y],
                       default=[c for c in num_cols if c != y][:8], key="corr_X")

with c2:
    dropna = st.checkbox("Remover linhas com NaN", value=True, key="corr_dropna")

if st.button("Calcular correlações", key="btn_corr"):
    if not X:
        st.warning("Escolha ao menos uma variável em X.")
        st.stop()
    df = gdf_pts[[y] + X].copy()
    if dropna:
        df = df.dropna()
    if df.empty:
        st.warning("Sem dados após remoção de NaN.")
        st.stop()

    corr = df.corr(method="pearson")
    st.dataframe(corr.style.background_gradient(cmap="RdYlGn").format("{:.2f}"), use_container_width=True)

    # Heatmap opcional
    fig, ax = plt.subplots(figsize=(min(10, 1.2*len(corr)), 6))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(corr.index)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Correlação de Pearson")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

