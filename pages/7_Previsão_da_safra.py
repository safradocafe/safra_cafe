import os, glob, io, csv, json
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.path import Path

from sklearn.preprocessing import StandardScaler
import joblib

try:
    from pyproj import CRS, Transformer
    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False

# =========================
# Página / estilo
# =========================
st.set_page_config(layout="wide")
st.markdown("## 🔮 Predição para todos os pontos + Mapa de variabilidade")
st.caption("Usa o **melhor modelo salvo** na aba de Treinamento e o CSV de índices salvo na aba de Processamento.")

BASE_TMP = "/tmp/streamlit_dados"
TOKENS_IDX = ["NDVI", "GNDVI", "NDRE", "CCCI", "MSAVI2", "NDWI", "NDMI", "NBR", "TWI2"]

# =========================
# Utilitários de descoberta/IO
# =========================
def _find_latest_save_dir(base=BASE_TMP):
    if not os.path.isdir(base):
        return None
    cands = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def _find_latest_indices_csv(base=BASE_TMP):
    pats = glob.glob(os.path.join(base, "salvamento-*", "indices_espectrais_pontos_*.csv"))
    if not pats:
        return None
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pats[0]

def _find_points_gpkg(save_dir):
    # tenta os nomes mais usados
    cands = [
        "pontos_produtividade.gpkg",           # usado nas outras abas
        "prod_requinte_colab.gpkg",            # fallback do colab
        "pontos_com_previsao.gpkg",            # caso já tenha saído de outra etapa
    ]
    for name in cands:
        p = os.path.join(save_dir, name)
        if os.path.exists(p):
            return p
    # último recurso: qualquer .gpkg com ponto
    gps = glob.glob(os.path.join(save_dir, "*.gpkg"))
    for p in gps:
        try:
            g = gpd.read_file(p)
            if "Point" in g.geom_type.unique()[0]:
                return p
        except Exception:
            pass
    return None

def _find_area_gpkg(save_dir):
    cands = [
        "area_amostral.gpkg",
        "area_poligono.gpkg",
        "area_total_poligono.gpkg",
        "requinte_colab.gpkg",   # fallback do colab
    ]
    for name in cands:
        p = os.path.join(save_dir, name)
        if os.path.exists(p):
            return p
    # último recurso: qualquer .gpkg com polígono
    gps = glob.glob(os.path.join(save_dir, "*.gpkg"))
    for p in gps:
        try:
            g = gpd.read_file(p)
            if "Polygon" in g.geom_type.unique()[0] or "MultiPolygon" in g.geom_type.unique()[0]:
                return p
        except Exception:
            pass
    return None

def _find_latest_model_pkl(base=BASE_TMP):
    pats = glob.glob(os.path.join(base, "melhor_modelo_*.pkl"))
    if not pats:
        # fallback: qualquer modelo salvo
        pats = glob.glob(os.path.join(base, "*.pkl"))
    if not pats:
        return None
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pats[0]

def _sniff_delim_and_decimal(sample_bytes: bytes):
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

def _filter_feature_columns(df: pd.DataFrame):
    drop_cols = [c for c in ["Code", "latitude", "longitude", "geometry"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    idx_cols = [c for c in df.columns if any(tok in c for tok in TOKENS_IDX)]
    if "maduro_kg" in df.columns:
        keep = ["maduro_kg"] + idx_cols
    else:
        keep = idx_cols
    out = df[keep].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(axis=1, how="all")
    return out

def _auto_utm_epsg_from_gdf(gdf: gpd.GeoDataFrame):
    """Retorna um EPSG UTM adequado (SIRGAS/WGS84) a partir do centróide."""
    if not HAVE_PYPROJ:
        return 4326
    centroid = gdf.geometry.unary_union.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    if south:
        # SIRGAS 2000 UTM sul: 319xx (Am Sul) não é universal. Vamos usar WGS84 UTM sul 327xx, que está disponível sempre.
        return int(f"327{zone:02d}")
    else:
        return int(f"326{zone:02d}")

# =========================
# 1) Descoberta automática de insumos
# =========================
save_dir = _find_latest_save_dir()
csv_indices_path = _find_latest_indices_csv()
model_pkl_path = _find_latest_model_pkl()

if not save_dir or not csv_indices_path or not model_pkl_path:
    st.error("❌ Não encontrei os insumos necessários automaticamente (diretório de salvamento, CSV de índices ou modelo).")
    st.stop()

pts_gpkg_path  = _find_points_gpkg(save_dir)
area_gpkg_path = _find_area_gpkg(save_dir)

st.caption(f"Origem dos dados: `{save_dir}`")
st.caption(f"CSV de índices: `{csv_indices_path}`")
st.caption(f"Modelo: `{model_pkl_path}`")

if not pts_gpkg_path or not os.path.exists(pts_gpkg_path):
    st.error("❌ GPKG de pontos não encontrado no salvamento.")
    st.stop()
if not area_gpkg_path or not os.path.exists(area_gpkg_path):
    st.warning("⚠️ GPKG de área não encontrado; usarei o envelope dos pontos como máscara do mapa.")

# =========================
# 2) Carregar dados / preparar X
# =========================
df_raw = _read_csv_robusto(csv_indices_path)
df_feat = _filter_feature_columns(df_raw)

if df_feat.empty:
    st.error("❌ Não há colunas de índices espectrais utilizáveis no CSV.")
    st.stop()

with st.expander("Pré-visualização (dados de entrada do modelo)"):
    st.dataframe(df_feat.head(), use_container_width=True)

# Carrega geometria (para anexar e mapear depois)
gdf_pts = gpd.read_file(pts_gpkg_path)
if gdf_pts.crs is None:
    gdf_pts = gdf_pts.set_crs(4326)
else:
    gdf_pts = gdf_pts.to_crs(4326)

if len(gdf_pts) != len(df_feat):
    st.warning(f"⚠️ Quantidade de pontos ({len(gdf_pts)}) difere do CSV ({len(df_feat)}). Vou alinhar pelo menor tamanho.")
n = min(len(gdf_pts), len(df_feat))
gdf_pts = gdf_pts.iloc[:n].reset_index(drop=True)
df_feat = df_feat.iloc[:n].reset_index(drop=True)

# =========================
# 3) Carregar melhor modelo e preparar features
# =========================
bundle = joblib.load(model_pkl_path)
model  = bundle.get("model", None)
features = bundle.get("features", list(df_feat.columns.drop("maduro_kg")) if "maduro_kg" in df_feat.columns else list(df_feat.columns))
scaler = bundle.get("scaler", None)

# Garante as features corretas e na ordem
missing = [f for f in features if f not in df_feat.columns]
if missing:
    st.error(f"❌ O CSV não contém todas as features esperadas pelo modelo: {missing}")
    st.stop()

X = df_feat[features].copy()
y_real = df_feat["maduro_kg"] if "maduro_kg" in df_feat.columns else None

# =========================
# 4) Predição para todos os pontos
# =========================
if scaler is not None:
    Xn = scaler.transform(X.values)
else:
    Xn = X.values

y_pred = model.predict(Xn)
pred_col = "Produtividade_Predita_kg"
df_out = pd.DataFrame({pred_col: y_pred})

# monta GeoDataFrame de saída com geometria dos pontos
gdf_pred = gpd.GeoDataFrame(
    pd.concat([df_feat.reset_index(drop=True), df_out], axis=1),
    geometry=gdf_pts.geometry, crs=gdf_pts.crs
)

# salva CSV e GPKG no mesmo salvamento
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_out = os.path.join(save_dir, f"produtividade_predita_pontos_{ts}.csv")
gpkg_out = os.path.join(save_dir, f"produtividade_predita_pontos_{ts}.gpkg")

gdf_pred_no_geom = pd.DataFrame(gdf_pred.drop(columns=["geometry"], errors="ignore"))
gdf_pred_no_geom.to_csv(csv_out, index=False)
gdf_pred.to_file(gpkg_out, driver="GPKG")

st.success("✅ Predições calculadas e salvas!")
st.caption(f"CSV: `{csv_out}`")
st.caption(f"GPKG: `{gpkg_out}`")

with st.expander("Prévia das predições (5 primeiras linhas)"):
    st.dataframe(gdf_pred_no_geom.head(), use_container_width=True)

st.download_button("📥 Baixar CSV de predições", data=open(csv_out, "rb").read(),
                   file_name=os.path.basename(csv_out), mime="text/csv")

# =========================
# 5) Mapa de variabilidade (interpolação spline/RBF)
# =========================
st.markdown("---")
st.markdown("### 🗺️ Mapa de variabilidade espacial (Spline/RBF)")

# Carrega área para máscara (ou usa envelope)
if area_gpkg_path and os.path.exists(area_gpkg_path):
    gdf_area = gpd.read_file(area_gpkg_path)
    if gdf_area.crs is None:
        gdf_area = gdf_area.set_crs(4326)
    else:
        gdf_area = gdf_area.to_crs(4326)
else:
    # fallback: polígono = envelope dos pontos
    gdf_area = gpd.GeoDataFrame(geometry=[gdf_pts.geometry.unary_union.envelope], crs=4326)

# Decide UTM automaticamente
try:
    epsg_utm = _auto_utm_epsg_from_gdf(gdf_area)
except Exception:
    epsg_utm = 32722  # fallback

gdf_area_utm = gdf_area.to_crs(epsg=epsg_utm)
gdf_pred_utm = gdf_pred.to_crs(epsg=epsg_utm)

# Grade regular
xmin, ymin, xmax, ymax = gdf_area_utm.total_bounds
res = 5  # metros
xi = np.arange(xmin, xmax, res)
yi = np.arange(ymin, ymax, res)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# dados (x,y,z)
vals = gdf_pred_utm[pred_col].values.astype(float)
xy = np.array([(p.x, p.y) for p in gdf_pred_utm.geometry])
xyz = np.c_[xy, vals]

# Spline/RBF (thin-plate)
from scipy.interpolate import Rbf
rbf = Rbf(xyz[:, 0], xyz[:, 1], xyz[:, 2], function="thin_plate")
zi = rbf(xi_grid, yi_grid)

# Máscara do polígono
poly_path = Path(gdf_area_utm.geometry.unary_union.exterior.coords)
mask = np.array([poly_path.contains_point((x, y)) for x, y in zip(xi_grid.flatten(), yi_grid.flatten())])
zi_masked = np.full_like(zi, np.nan, dtype=float)
zi_masked_flat = zi_masked.flatten()
zi_flat = zi.flatten()
zi_masked_flat[mask] = zi_flat[mask]
zi_masked = zi_masked_flat.reshape(zi.shape)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(
    zi_masked,
    extent=(xmin, xmax, ymin, ymax),
    origin="lower",
    cmap="YlGn",
    interpolation="nearest"
)
gdf_area_utm.boundary.plot(ax=ax, color="black", linewidth=1)
gdf_pred_utm.plot(ax=ax, color="red", markersize=16)

cbar = plt.colorbar(img, ax=ax, shrink=0.75)
cbar.set_label("Produtividade Predita (kg)")

# Rosa dos ventos simples
ax.annotate('N', xy=(0.96, 0.96), xycoords='axes fraction', fontsize=12, fontweight='bold')
ax.annotate('', xy=(0.96, 0.92), xytext=(0.96, 0.96), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=1.3))

# Barra de escala (aproximada)
ax.plot([0.85, 0.90], [0.05, 0.05], color='black', linewidth=3, transform=ax.transAxes)
ax.text(0.875, 0.065, '100 m', transform=ax.transAxes, ha='center', fontsize=9)

ax.set_title("Variabilidade Espacial da Produtividade Predita — Spline (RBF)")
ax.set_xlabel(f"UTM Easting (m) | EPSG:{epsg_utm}")
ax.set_ylabel(f"UTM Northing (m)")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

st.info("Se quiser GeoTIFF do raster interpolado, dá pra gerar em outra etapa. Aqui evitei GDAL para não trazer dependências pesadas.")
