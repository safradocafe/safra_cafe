import os, glob, json, joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide")
st.markdown("<h3>🧠 Treinamento (Machine Learning)</h3>", unsafe_allow_html=True)

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

st.caption(f"Origem dos dados: `{latest_dir}`")
num_cols = [c for c in gdf_pts.columns if pd.api.types.is_numeric_dtype(gdf_pts[c])]

y = st.selectbox("Alvo (y)", ["maduro_kg"] + [c for c in num_cols if c != "maduro_kg"], key="ml_y")
X = st.multiselect("Features (X)", [c for c in num_cols if c != y],
                   default=[c for c in num_cols if c != y][:12], key="ml_X")

c1, c2, c3 = st.columns(3)
with c1:
    test_size = st.slider("Proporção de teste", 0.1, 0.5, 0.2, 0.05, key="ml_test")
with c2:
    n_estimators = st.slider("Árvores (RandomForest)", 100, 1500, 500, 50, key="ml_trees")
with c3:
    random_state = st.number_input("Random state", value=42, step=1, key="ml_rs")

if st.button("Treinar modelo", key="btn_train"):
    if not X:
        st.warning("Escolha ao menos uma feature.")
        st.stop()

    df = gdf_pts[[y] + X].dropna()
    if df.empty:
        st.warning("Sem dados após remoção de NaN.")
        st.stop()

    X_mat = df[X].values
    y_vec = df[y].values

    X_tr, X_te, y_tr, y_te = train_test_split(X_mat, y_vec, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    with st.spinner("Treinando e validando..."):
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        r2  = r2_score(y_te, preds)
        rmse = mean_squared_error(y_te, preds, squared=False)

    st.success("Modelo treinado!")
    st.write(f"R² (CV 5-fold): **{cv_scores.mean():.3f} ± {cv_scores.std():.3f}**")
    st.write(f"R² (holdout): **{r2:.3f}** | RMSE: **{rmse:.2f}**")

    # Importância de variáveis
    st.subheader("Importância das features")
    st.bar_chart(pd.Series(model.feature_importances_, index=X).sort_values(ascending=False))

    # Persistir
    os.makedirs(BASE_TMP, exist_ok=True)
    joblib.dump({"model": model, "features": X, "target": y}, MODEL_PATH)
    st.caption(f"Modelo salvo temporariamente em: `{MODEL_PATH}`")
