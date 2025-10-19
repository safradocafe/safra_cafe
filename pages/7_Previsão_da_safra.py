# pages/4_3_Predicao_e_Mapa.py
import os, glob, io, csv, json
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.prepared import prep
from scipy.interpolate import Rbf
import joblib

try:
    from pyproj import Transformer
    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False

# =========================
# Página / estilo
# =========================
st.set_page_config(layout="wide")
st.markdown("## 🔮 Predição para todos os pontos + Mapa de variabilidade")
st.caption("Usa automaticamente o **CSV de índices** salvo na aba Processamento e o **melhor modelo** salvo na aba Treinamento. "
           "As métricas de produtividade dependem da **densidade de plantas (pés/ha)** informada pelo usuário.")

BASE_TMP = "/tmp/streamlit_dados"
TOKENS_IDX = ["NDVI","GNDVI","NDRE","CCCI","MSAVI2","NDWI","NDMI","NBR","TWI2"]

# =========================
# Utilidades de descoberta/IO
# =========================
def _find_latest_save_dir(base=BASE_TMP):
    if not os.path.isdir(base): return None
    cands = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not cands: return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def _find_latest_indices_csv(base=BASE_TMP):
    pats = glob.glob(os.path.join(base, "salvamento-*", "indices_espectrais_pontos_*.csv"))
    if not pats: return None
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pats[0]

def _find_latest_model(base=BASE_TMP):
    pats = glob.glob(os.path.join(base, "melhor_modelo_*.pkl"))
    if not pats:
        pats = glob.glob(os.path.join(base, "*.pkl"))
    if not pats: return None
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pats[0]

def _find_points_gpkg(save_dir):
    cands = ["pontos_produtividade.gpkg","pontos_com_previsao.gpkg","prod_requinte_colab.gpkg"]
    for nm in cands:
        p = os.path.join(save_dir, nm)
        if os.path.exists(p): return p
    for p in glob.glob(os.path.join(save_dir, "*.gpkg")):
        try:
            g = gpd.read_file(p)
            if any("Point" in gt for gt in g.geom_type.unique()):
                return p
        except Exception:
            pass
    return None

def _find_area_gpkg(save_dir):
    cands = ["area_amostral.gpkg","area_poligono.gpkg","area_total_poligono.gpkg","requinte_colab.gpkg"]
    for nm in cands:
        p = os.path.join(save_dir, nm)
        if os.path.exists(p): return p
    for p in glob.glob(os.path.join(save_dir, "*.gpkg")):
        try:
            g = gpd.read_file(p)
            if any(("Polygon" in gt) for gt in g.geom_type.unique()):
                return p
        except Exception:
            pass
    return None

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
def _read_csv_robusto(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read()
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
    df = df.dropna(axis=1, how="all")
    return df

def _filter_features_df(df: pd.DataFrame):
    # mantém maduro_kg + TODAS colunas de índices (min/mean/max etc.)
    drop_cols = [c for c in ["Code","latitude","longitude","geometry"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    idx_cols = [c for c in df.columns if any(tok in c for tok in TOKENS_IDX)]
    keep = (["maduro_kg"] if "maduro_kg" in df.columns else []) + idx_cols
    out = df[keep].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(axis=1, how="all")
    return out

def _auto_utm_epsg(geom_gdf: gpd.GeoDataFrame) -> int:
    if not HAVE_PYPROJ: return 32722
    centroid = geom_gdf.geometry.unary_union.centroid
    lon = float(centroid.x); lat = float(centroid.y)
    zone = int((lon + 180) // 6) + 1
    return int(f"{326 if lat >= 0 else 327}{zone:02d}")

def _mask_array_with_polygon(xi_grid, yi_grid, poly_union):
    prep_poly = prep(poly_union)
    flat_pts = np.c_[xi_grid.ravel(), yi_grid.ravel()]
    mask = np.fromiter(
        (prep_poly.contains(Polygon([(x,y),(x+0.001,y),(x+0.001,y+0.001),(x,y+0.001)]).centroid)
         for x,y in flat_pts),
        dtype=bool, count=flat_pts.shape[0]
    )
    return mask.reshape(xi_grid.shape)

# =========================
# Descoberta automática dos insumos
# =========================
save_dir = _find_latest_save_dir()
csv_path = _find_latest_indices_csv()
model_path = _find_latest_model()

if not (save_dir and csv_path and model_path):
    st.error("❌ Não encontrei automaticamente: diretório de salvamento, CSV de índices e/ou melhor modelo.")
    st.stop()

pts_gpkg = _find_points_gpkg(save_dir)
area_gpkg = _find_area_gpkg(save_dir)

st.caption(f"Origem: `{save_dir}`")
st.caption(f"CSV de índices: `{csv_path}`")
st.caption(f"Modelo: `{model_path}`")

if not pts_gpkg:
    st.error("❌ GPKG de pontos não encontrado no salvamento.")
    st.stop()
if not area_gpkg:
    st.warning("⚠️ GPKG de área não encontrado; usarei o envelope dos pontos como área de máscara.")

# =========================
# Carregar dados / preparar X
# =========================
df_raw = _read_csv_robusto(csv_path)
df_feat = _filter_features_df(df_raw)
if df_feat.empty:
    st.error("❌ CSV não contém colunas de índices válidas.")
    st.stop()

gdf_pts = gpd.read_file(pts_gpkg)
if gdf_pts.crs is None: gdf_pts = gdf_pts.set_crs(4326)
else: gdf_pts = gdf_pts.to_crs(4326)

# alinhar tamanho
n = min(len(df_feat), len(gdf_pts))
df_feat = df_feat.iloc[:n].reset_index(drop=True)
gdf_pts  = gdf_pts.iloc[:n].reset_index(drop=True)

with st.expander("Prévia dos dados de entrada (maduro_kg + índices)"):
    st.dataframe(df_feat.head(), use_container_width=True)

# =========================
# Carregar modelo e prever TODOS os pontos
# =========================
bundle = joblib.load(model_path)
if isinstance(bundle, dict) and "model" in bundle:
    model   = bundle.get("model")
    features= bundle.get("features", [c for c in df_feat.columns if c != "maduro_kg"])
    scaler  = bundle.get("scaler", None)
else:
    model   = bundle
    features= [c for c in df_feat.columns if c != "maduro_kg"]
    scaler  = None

missing = [f for f in features if f not in df_feat.columns]
if missing:
    st.error(f"❌ O CSV não contém todas as features esperadas pelo modelo: {missing}")
    st.stop()

X = df_feat[features].values
if scaler is not None:
    try:
        X = scaler.transform(X)
    except Exception:
        pass

y_pred = model.predict(X)

# GeoDataFrame com predições (kg por planta — usado com densidade para kg/ha)
gdf_pred = gpd.GeoDataFrame(
    pd.DataFrame({**{c: df_feat[c].values for c in df_feat.columns if c != "geometry"},
                  "Produtividade_Predita_kg": y_pred}),
    geometry=gdf_pts.geometry, crs=gdf_pts.crs
)

st.success("✅ Predição concluída para todos os pontos!")

# =========================
# Área total e densidade (obrigatória) — só então calcular métricas
# =========================
# Área (para métricas e máscara do mapa)
if area_gpkg:
    gdf_area = gpd.read_file(area_gpkg)
    if gdf_area.crs is None: gdf_area = gdf_area.set_crs(4326)
    else: gdf_area = gdf_area.to_crs(4326)
else:
    gdf_area = gpd.GeoDataFrame(geometry=[gdf_pts.geometry.unary_union.envelope], crs=4326)

epsg_utm = _auto_utm_epsg(gdf_area)
gdf_area_utm = gdf_area.to_crs(epsg=epsg_utm)
area_total_ha = float(gdf_area_utm.area.sum() / 10_000.0)
st.caption(f"• Área total do(s) polígono(s): **{area_total_ha:.4f} ha** (EPSG:{epsg_utm})")

st.markdown("### 📏 Densidade de plantas (obrigatória para calcular produtividade)")
densidade_pes_ha = st.number_input("Densidade de plantas (pés/ha)", min_value=1, value=None, step=1, format="%d", placeholder="ex.: 4000")

if not densidade_pes_ha:
    st.info("Informe a **densidade de plantas (pés/ha)** para ver as métricas de produtividade.")
else:
    # Conversão com densidade:
    # média(kg por planta) -> kg/ha = média * densidade; sacas/ha = kg/ha / 60
    media_kg_por_planta = float(np.nanmean(gdf_pred["Produtividade_Predita_kg"]))
    prod_kg_ha  = media_kg_por_planta * float(densidade_pes_ha)
    prod_sc_ha  = prod_kg_ha / 60.0
    total_kg    = prod_kg_ha * area_total_ha
    total_sacas = prod_sc_ha * area_total_ha

    st.markdown("### 📈 Produtividade prevista (ajustada por densidade)")
    c1, c2 = st.columns(2)
    c1.metric("Produtividade média prevista (kg/ha)", f"{prod_kg_ha:.2f}")
    c2.metric("Produtividade média prevista (sacas/ha)", f"{prod_sc_ha:.2f}")

    c3, c4 = st.columns(2)
    c3.metric("Produção total estimada (kg)", f"{total_kg:.0f}")
    c4.metric("Produção total estimada (sacas)", f"{total_sacas:.2f}")

    # Salvar CSV/GPKG + métricas
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_out  = os.path.join(save_dir, f"produtividade_predita_pontos_{ts}.csv")
    gpkg_out = os.path.join(save_dir, f"produtividade_predita_pontos_{ts}.gpkg")
    met_out  = os.path.join(save_dir, f"produtividade_media_ajustada_{ts}.csv")

    # (mantém kg por ponto; NÃO cria sc/ha por ponto, pois a densidade é global)
    gdf_pred.drop(columns=["geometry"], errors="ignore").to_csv(csv_out, index=False)
    gdf_pred.to_file(gpkg_out, driver="GPKG")

    df_metrics = pd.DataFrame([{
        "densidade_pes_ha": int(densidade_pes_ha),
        "area_total_ha": area_total_ha,
        "media_kg_por_planta": media_kg_por_planta,
        "prod_kg_ha": prod_kg_ha,
        "prod_sacas_ha": prod_sc_ha,
        "total_kg": total_kg,
        "total_sacas": total_sacas,
    }])
    df_metrics.to_csv(met_out, index=False)

    st.caption(f"CSV (predições por ponto): `{csv_out}`")
    st.caption(f"GPKG (predições por ponto): `{gpkg_out}`")
    st.caption(f"CSV (métricas): `{met_out}`")
    st.download_button("📥 Baixar CSV de predições", data=open(csv_out, "rb").read(),
                       file_name=os.path.basename(csv_out), mime="text/csv")
    st.download_button("📥 Baixar CSV de métricas", data=open(met_out, "rb").read(),
                       file_name=os.path.basename(met_out), mime="text/csv")

# =========================
# Mapa de variabilidade (interpolação spline/RBF) — independe da densidade
# =========================
st.markdown("---")
st.subheader("🗺️ Mapa de variabilidade espacial da produtividade prevista (kg por planta)")

gdf_pred_utm = gdf_pred.to_crs(epsg=epsg_utm)

# Grade regular (5 m)
xmin, ymin, xmax, ymax = gdf_area_utm.total_bounds
res = 5.0  # metros
xi = np.arange(xmin, xmax, res)
yi = np.arange(ymin, ymax, res)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# Dados para spline (x, y, z)
xy = np.array([(p.x, p.y) for p in gdf_pred_utm.geometry])
z  = gdf_pred_utm["Produtividade_Predita_kg"].astype(float).values

# Interpolação RBF thin-plate spline
rbf = Rbf(xy[:,0], xy[:,1], z, function="thin_plate")
zi = rbf(xi_grid, yi_grid)

# Máscara por polígono
poly_union = gdf_area_utm.geometry.unary_union
mask = _mask_array_with_polygon(xi_grid, yi_grid, poly_union)
zi_masked = np.full_like(zi, np.nan, dtype=float)
zi_masked[mask] = zi[mask]

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
cbar.set_label("Produtividade Predita (kg por planta)")

ax.set_title(f"Variabilidade Espacial — Spline (RBF) | EPSG:{epsg_utm}")
ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# Download da figura
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
st.download_button("🖼️ Baixar mapa (PNG)", data=buf.getvalue(),
                   file_name=f"mapa_variabilidade_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png", mime="image/png")

