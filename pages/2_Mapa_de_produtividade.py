
import os
import glob
import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

st.set_page_config(layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 0rem !important; padding-bottom: 0.5rem; }
header, footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helpers
# -------------------------------
def _find_latest_save_dir(base="/tmp/streamlit_dados"):
    if not os.path.isdir(base):
        return None
    candidates = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def _load_area_and_points():
    """
    Tenta ler caminhos salvos no session_state; se não houver,
    busca a pasta de salvamento mais recente em /tmp/streamlit_dados.
    Retorna (gdf_area, gdf_pontos, info_dir)
    """
    # 1) tenta session_state
    save_dir = st.session_state.get("tmp_save_dir")
    area_path = st.session_state.get("tmp_area_path")
    pontos_path = st.session_state.get("tmp_pontos_path")

    # 2) fallback: busca último salvamento no /tmp
    if not save_dir or not os.path.exists(save_dir):
        latest = _find_latest_save_dir()
        if latest:
            save_dir = latest
            area_path = os.path.join(save_dir, "area_amostral.gpkg")
            pontos_path = os.path.join(save_dir, "pontos_produtividade.gpkg")

    if not save_dir or not os.path.exists(save_dir):
        return None, None, None

    # área é obrigatória
    if not area_path or not os.path.exists(area_path):
        return None, None, save_dir

    gdf_area = gpd.read_file(area_path)
    if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
        gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)

    gdf_pontos = None
    if pontos_path and os.path.exists(pontos_path):
        gdf_pontos = gpd.read_file(pontos_path)
        if gdf_pontos.crs is None or gdf_pontos.crs.to_epsg() != 4326:
            gdf_pontos = gdf_pontos.set_crs(4326) if gdf_pontos.crs is None else gdf_pontos.to_crs(4326)

        # valida colunas exigidas (sem 'fid')
        required = ["Code", "maduro_kg", "latitude", "longitude", "geometry"]
        faltando = [c for c in required if c not in gdf_pontos.columns]
        if faltando:
            st.error("Arquivo de pontos encontrado, mas faltam colunas obrigatórias: " + ", ".join(faltando))
            gdf_pontos = None

    return gdf_area, gdf_pontos, save_dir

def _fmt_num(v, nd=6):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

# -------------------------------
# UI – Controles
# -------------------------------
st.title("🗺️ Mapa de variabilidade da produtividade (dados reais)")

with st.sidebar:
    st.subheader("Exibição")
    heat_radius = st.slider("Raio do HeatMap (px)", 8, 40, 18, 1)
    heat_blur   = st.slider("Desfoque (blur)", 8, 35, 20, 1)
    heat_opacity = st.slider("Opacidade do HeatMap", 0.2, 1.0, 0.6, 0.05)
    show_points = st.checkbox("Mostrar pontos e popups", value=True)
    show_area   = st.checkbox("Mostrar polígono da área", value=True)

# -------------------------------
# Carrega dados
# -------------------------------
gdf_area, gdf_pontos, base_dir = _load_area_and_points()

if gdf_area is None:
    st.warning("⚠️ Não encontrei a área amostral em /tmp. Volte para 'Adicionar informações' e salve os dados.")
    st.stop()

centroid = gdf_area.to_crs(4326).geometry.unary_union.centroid
start_loc = [centroid.y, centroid.x]

# -------------------------------
# Mapa Folium (leve)
# -------------------------------
m = folium.Map(location=start_loc, zoom_start=17, tiles="OpenStreetMap")
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satélite', overlay=False, control=True
).add_to(m)

# Polígono da área
if show_area:
    folium.GeoJson(
        gdf_area[['geometry']].__geo_interface__,
        name="Área Amostral",
        style_function=lambda x: {"color": "blue", "fillColor": "blue", "fillOpacity": 0.15, "weight": 2}
    ).add_to(m)

# HeatMap com pesos em maduro_kg
if gdf_pontos is not None and not gdf_pontos.empty:
    vals = gdf_pontos["maduro_kg"].astype(float).values
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if vmax == vmin:
        weights = np.ones_like(vals)  # tudo igual
    else:
        weights = (vals - vmin) / (vmax - vmin)  # 0..1

    heat_data = [
        [row["latitude"], row["longitude"], float(w)]
        for (_, row), w in zip(gdf_pontos.iterrows(), weights)
        if np.isfinite(row["latitude"]) and np.isfinite(row["longitude"])
    ]

    if heat_data:
        HeatMap(
            heat_data,
            radius=heat_radius,
            blur=heat_blur,
            min_opacity=heat_opacity,
            max_zoom=19,
            name="Variabilidade (HeatMap, peso = maduro_kg)"
        ).add_to(m)
    else:
        st.info("Não há dados válidos para o HeatMap.")

    # pontos clicáveis (opcional)
    if show_points:
        for _, row in gdf_pontos.iterrows():
            lat = row.get("latitude")
            lon = row.get("longitude")
            if not (np.isfinite(lat) and np.isfinite(lon)):
                continue
            popup_html = (
                f"<b>Code:</b> {row.get('Code','-')}"
                f"<br><b>Produtividade (kg):</b> {row.get('maduro_kg','-')}"
                f"<br><b>Lat/Lon:</b> {_fmt_num(lat,6)}, {_fmt_num(lon,6)}"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="green",
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
else:
    st.warning("⚠️ Pontos de produtividade não encontrados. O HeatMap será ocultado.")

folium.LayerControl(collapsed=False).add_to(m)

# Render
st.markdown(f"**Origem dos dados:** `{base_dir or 'desconhecida'}`")
st_folium(m, width=900, height=600, key="mapa_produtividade")
