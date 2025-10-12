# pages/3_Monitoramento.py

import os
import io
import json
import glob
import warnings
from datetime import date

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import folium
from shapely.geometry import Polygon
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# =========================
# Estilo e título
# =========================
st.set_page_config(layout="wide")
st.markdown("""
<style>
.block-container { padding-top: .25rem !important; padding-bottom: .5rem !important; }
header, footer {visibility:hidden;}
.chips { display:flex; gap:.5rem; flex-wrap:wrap; margin:.25rem 0 1rem 0; }
.chip { background:#f1f3f4; border-radius:999px; padding:.25rem .6rem; font-size:.85rem; }
.chip b { margin-right:.25rem; }
.hint { color:#666; font-size:.9rem; margin:.25rem 0 .5rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("### 📡 Monitoramento por índices espectrais (Sentinel-2)")
st.markdown("""
<div class="chips">
  <div class="chip"><b>NDVI</b>vigor vegetativo</div>
  <div class="chip"><b>GNDVI</b>sensível ao N foliar</div>
  <div class="chip"><b>NDRE</b>sensível à clorofila</div>
  <div class="chip"><b>CCCI</b>clorofila na copa</div>
  <div class="chip"><b>MSAVI2</b>reduz influência do solo</div>
  <div class="chip"><b>NDWI</b>conteúdo de água na folha</div>
  <div class="chip"><b>NDMI</b>umidade no dossel</div>
  <div class="chip"><b>NBR</b>estresse térmico</div>
  <div class="chip"><b>TWI2</b>umidade do ar</div>
</div>
<div class="hint">Escolha o período, os índices e a área (amostral salva no passo 1 ou um polígono da fazenda) para visualizar no mapa e ver a série temporal (média no polígono).</div>
""", unsafe_allow_html=True)

# =========================
# Utils: encontrar último salvamento (passo 1)
# =========================
def _find_latest_save_dir(base="/tmp/streamlit_dados"):
    if not os.path.isdir(base):
        return None
    candidates = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def load_area_from_tmp():
    latest = _find_latest_save_dir()
    if not latest:
        return None, None
    area_path = os.path.join(latest, "area_amostral.gpkg")
    pontos_path = os.path.join(latest, "pontos_produtividade.gpkg")
    if not os.path.exists(area_path):
        return None, None
    gdf_area = gpd.read_file(area_path)
    if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
        gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)
    return gdf_area, latest

# =========================
# EE init (com variável de ambiente)
# =========================
def ensure_ee_init():
    if ee.data._credentials:  # já inicializado
        return
    key_json = os.environ.get("GEE_SA_KEY_JSON", "")
    if not key_json:
        st.error("Credenciais do Google Earth Engine não encontradas. Configure a variável de ambiente **GEE_SA_KEY_JSON** no Cloud Run.")
        st.stop()
    try:
        creds = json.loads(key_json)
        credentials = ee.ServiceAccountCredentials(email=creds["client_email"], key_data=key_json)
        ee.Initialize(credentials)
    except Exception as e:
        st.error(f"Erro ao inicializar o Google Earth Engine: {e}")
        st.stop()

ensure_ee_init()

# =========================
# Visual configs por índice (paletas e ranges típicos)
# =========================
INDEX_VIS = {
    "NDVI":   dict(min=0.0, max=1.0, palette=["#d73027","#fdae61","#ffffbf","#a6d96a","#1a9850"]),
    "GNDVI":  dict(min=0.0, max=1.0, palette=["#fee08b","#e6f598","#abdda4","#66c2a5","#3288bd"]),
    "NDRE":   dict(min=0.0, max=0.5, palette=["#f7fcf0","#ccebc5","#7bccc4","#2b8cbe","#08589e"]),
    "CCCI":   dict(min=0.0, max=2.0, palette=["#f7f4f9","#e7e1ef","#c994c7","#dd1c77","#980043"]),
    "MSAVI2": dict(min=0.0, max=1.0, palette=["#ffffcc","#c2e699","#78c679","#31a354","#006837"]),
    "NDWI":   dict(min=-1.0, max=1.0, palette=["#f7fbff","#deebf7","#9ecae1","#3182bd","#08519c"]),
    "NDMI":   dict(min=-1.0, max=1.0, palette=["#fff7bc","#fec44f","#fe9929","#d95f0e","#993404"]),
    "NBR":    dict(min=-1.0, max=1.0, palette=["#ffffcc","#c2e699","#78c679","#31a354","#006837"]),
    "TWI2":   dict(min=-1.0, max=1.0, palette=["#f1eef6","#bdc9e1","#74a9cf","#2b8cbe","#045a8d"]),
}

# =========================
# Sidebar – controles
# =========================
with st.sidebar:
    st.subheader("Configurações")
    area_opt = st.radio(
        "Área de interesse:",
        ["Usar área amostral salva (passo 1)", "Fazer upload de polígono (GPKG)"],
        index=0
    )

    c1, c2 = st.columns(2)
    start = c1.date_input("Início", value=date(2024,1,1))
    end   = c2.date_input("Fim", value=date.today())

    indices_sel = st.multiselect(
        "Índices",
        list(INDEX_VIS.keys()),
        default=["NDVI", "GNDVI", "NDRE", "MSAVI2", "NDWI"]
    )

    cloud_thr = st.slider("Nuvem máxima (%)", 0, 60, 10, 1)
    btn = st.button("▶️ Processar")

# =========================
# Carregar área
# =========================
gdf_area = None
base_dir = None
if area_opt.startswith("Usar"):
    gdf_area, base_dir = load_area_from_tmp()
    if gdf_area is None:
        st.warning("Não encontrei a área amostral salva. Volte ao passo 1 e clique em **Salvar dados**.")
else:
    up = st.file_uploader("Carregue um polígono (GPKG)", type=["gpkg"])
    if up:
        tmp = f"/tmp/_poly_upload.gpkg"
        with open(tmp, "wb") as f:
            f.write(up.getbuffer())
        gdf_area = gpd.read_file(tmp)
        if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
            gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)

if gdf_area is None:
    st.stop()

# Converte polígono para EE
try:
    ee_poly = geemap.gdf_to_ee(gdf_area[["geometry"]])
except Exception as e:
    st.error(f"Falha ao converter polígono para EE: {e}")
    st.stop()

# =========================
# Funções EE
# =========================
def add_indices(img, wanted):
    out = img
    band = {b: img.select(b) for b in ["B3","B4","B5","B8","B11","B12","B9"] if b in img.bandNames().getInfo()}
    def nd(x,y): return img.normalizedDifference([x,y])

    if "NDVI"  in wanted:   out = out.addBands(nd("B8","B4").rename("NDVI"))
    if "GNDVI" in wanted:   out = out.addBands(nd("B8","B3").rename("GNDVI"))
    if "NDRE"  in wanted:   out = out.addBands(nd("B8","B5").rename("NDRE"))
    if "CCCI"  in wanted:   out = out.addBands(nd("B8","B5").divide(nd("B8","B4")).rename("CCCI"))
    if "MSAVI2" in wanted:
        msavi2 = img.expression(
            "(2*NIR + 1 - sqrt((2*NIR + 1)**2 - 8*(NIR-RED)))/2",
            {"NIR": img.select("B8"), "RED": img.select("B4")}
        ).rename("MSAVI2")
        out = out.addBands(msavi2)
    if "NDWI"  in wanted:   out = out.addBands(nd("B3","B8").rename("NDWI"))
    if "NDMI"  in wanted:   out = out.addBands(nd("B8","B11").rename("NDMI"))
    if "NBR"   in wanted:   out = out.addBands(nd("B8","B12").rename("NBR"))
    if "TWI2"  in wanted:   out = out.addBands(nd("B9","B8").rename("TWI2"))
    return out

def ee_tilelayer_from_image(image, vis):
    """Retorna dict com URL de tiles para usar em Folium.TileLayer."""
    m = ee.Image(image).getMapId(vis)
    return m["tile_fetcher"].url_format

# =========================
# Estado persistente
# =========================
if "mon_tiles" not in st.session_state:
    st.session_state["mon_tiles"] = None   # {idx: tile_url}
if "mon_series" not in st.session_state:
    st.session_state["mon_series"] = None  # pd.DataFrame
if "mon_bounds" not in st.session_state:
    st.session_state["mon_bounds"] = None  # [[S,W],[N,E]]

# =========================
# Processamento (uma vez) e guarda em sessão
# =========================
def do_process():
    # Coleção Sentinel-2 SR harmonizada
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(ee_poly.geometry())
           .filterDate(ee.Date(str(start)), ee.Date(str(end)))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thr))
           .map(lambda im: add_indices(im, indices_sel)))

    # Se não houver imagens, aborta educadamente
    nimgs = col.size().getInfo()
    if nimgs == 0:
        st.warning("Nenhuma imagem encontrada no período/área com esse limite de nuvem.")
        return

    # Para o mapa: usa mediana no período (leve e estável)
    median = col.median().clip(ee_poly)

    # Gera URL de tiles por índice selecionado
    tiles = {}
    for idx in indices_sel:
        vis = INDEX_VIS[idx]
        tiles[idx] = ee_tilelayer_from_image(median.select(idx), vis)

    # Série temporal (média do índice no polígono por data)
    # Monta datas únicas
    dates = (col.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY-MM-dd"))).distinct().getInfo()
    dates = sorted(list(set(dates)))
    rows = []
    for d in dates:
        # Mediana diária (ou mosaico por dia)
        day_coll = col.filterDate(ee.Date(d), ee.Date(d).advance(1, "day"))
        if day_coll.size().getInfo() == 0:
            continue
        day_img = day_coll.median().clip(ee_poly)
        row = {"date": d}
        for idx in indices_sel:
            try:
                val = (day_img.select(idx)
                       .reduceRegion(ee.Reducer.mean(), geometry=ee_poly, scale=10, maxPixels=1e9)
                       .get(idx).getInfo())
            except Exception:
                val = None
            row[idx] = val
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date")

    # Bounds do polígono (para folium)
    g = gdf_area.geometry.unary_union.envelope.bounds  # (minx,miny,maxx,maxy)
    bounds = [[g[1], g[0]], [g[3], g[2]]]

    st.session_state["mon_tiles"]  = tiles
    st.session_state["mon_series"] = df
    st.session_state["mon_bounds"] = bounds

if btn:
    with st.spinner("Processando imagens e gerando camadas..."):
        try:
            do_process()
            st.success("Pronto! Veja o mapa e a série temporal abaixo.")
        except Exception as e:
            st.error(f"Falha no processamento: {e}")

# =========================
# Render sempre a partir do estado
# =========================
tiles = st.session_state.get("mon_tiles")
series = st.session_state.get("mon_series")
bounds = st.session_state.get("mon_bounds")

if tiles:
    # Mapa Folium com camadas alternáveis
    center = [gdf_area.geometry.unary_union.centroid.y, gdf_area.geometry.unary_union.centroid.x]
    m = folium.Map(location=center, zoom_start=16, tiles="OpenStreetMap")

    # Base Satélite opcional
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satélite", overlay=False, control=True
    ).add_to(m)

    # Polígono
    folium.GeoJson(
        gdf_area[["geometry"]].__geo_interface__,
        name="Área",
        style_function=lambda x: {"color":"#1976d2","weight":2,"fillColor":"#1976d2","fillOpacity":0.08}
    ).add_to(m)

    # Camadas por índice (a partir das URLs de tiles)
    first = True
    for idx, url in tiles.items():
        folium.raster_layers.TileLayer(
            tiles=url, name=f"{idx}", attr="Google Earth Engine",
            overlay=True, control=True, show=first, opacity=0.95
        ).add_to(m)
        first = False

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=1000, height=620)

    # Série temporal (média no polígono)
    if isinstance(series, pd.DataFrame) and not series.empty:
        st.subheader("📈 Série temporal (média do índice no polígono)")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        series_plot = series.copy()
        series_plot["date"] = pd.to_datetime(series_plot["date"])
        for idx in indices_sel:
            if idx in series_plot.columns:
                ax.plot(series_plot["date"], series_plot[idx], label=idx)
        ax.set_xlabel("Data")
        ax.set_ylabel("Valor médio")
        ax.grid(True, alpha=.3)
        ax.legend(ncols=min(4, len(indices_sel)), fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Sem pontos suficientes para montar a série temporal nesse período.")
else:
    st.info("Aguardando processamento. Defina período/índices e clique em **Processar**.")
