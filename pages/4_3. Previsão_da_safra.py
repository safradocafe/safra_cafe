import os, glob, json, joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

st.set_page_config(layout="wide")
st.markdown("<h3>📅 Previsão da safra</h3>", unsafe_allow_html=True)

BASE_TMP = "/tmp/streamlit_dados"
MODEL_PATH = os.path.join(BASE_TMP, "modelo_produtividade.pkl")

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

# Carrega o modelo
if not os.path.exists(MODEL_PATH):
    st.info("Treine um modelo na aba **Treinamento (ML)** para liberar a previsão.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model   = bundle["model"]
features= bundle["features"]
target  = bundle["target"]

st.caption(f"Modelo carregado: alvo **{target}** | features: {', '.join(features)}")
st.caption(f"Origem dos dados: `{latest_dir}`")

fonte = st.radio("Dados para aplicar o modelo:", ["Pontos atuais (passo 1)"], horizontal=True, key="apply_source")

df_apply = gdf_pts.copy()
missing = [f for f in features if f not in df_apply.columns]
if missing:
    st.error(f"A base não contém todas as features do modelo: {missing}")
    st.stop()

if st.button("Gerar previsão", key="btn_predict"):
    base = df_apply.dropna(subset=features).copy()
    if base.empty:
        st.warning("Sem linhas válidas após remover NaN nas features.")
        st.stop()

    base["pred_" + target] = model.predict(base[features].values)
    st.success("Previsões geradas!")
    show_cols = [*features, target] if target in base.columns else features
    show_cols += ["pred_" + target]
    st.dataframe(base[show_cols], use_container_width=True)

    # (opcional) salvar GPKG para uso em mapas
    if "geometry" in base.columns:
        out_path = os.path.join(BASE_TMP, "pontos_com_previsao.gpkg")
        base.to_file(out_path, driver="GPKG")
        st.caption(f"Arquivo com previsões salvo temporariamente em: `{out_path}`")
