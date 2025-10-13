import os
import json
import glob
from datetime import date, timedelta

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# ----------------------------
# Estilo e cabeçalho
# ----------------------------
st.set_page_config(layout="wide")
st.markdown("""
<style>
.block-container { padding-top: .25rem !important; padding-bottom: .5rem !important; }
header, footer {visibility:hidden;}
.chips { display:flex; gap:.5rem; flex-wrap:wrap; margin:.25rem 0 1rem 0; }
.chip { background:#f1f3f4; border-radius:999px; padding:.25rem .6rem; font-size:.85rem; }
.chip b { margin-right:.25rem; }
.hint { color:#666; font-size:.9rem; margin:.25rem 0 .5rem 0;}
.legend-box{
  position: absolute; bottom: 16px; right: 16px; z-index: 9999;
  background: rgba(255,255,255,.92); border: 1px solid #ddd; border-radius: 8px;
  padding: 10px 12px; font-size: 12px; line-height: 1.2; box-shadow: 0 2px 12px rgba(0,0,0,.15);
}
.legend-title{ font-weight: 700; margin-bottom: 6px; }
.legend-gradient{
  width: 220px; height: 12px; border-radius: 6px; margin: 6px 0;
  background: linear-gradient(to right, var(--grad)); border: 1px solid #999;
}
.legend-scale { display:flex; justify-content: space-between; font-size: 11px; color:#333; }
.legend-meta { color:#444; margin-top: 4px; }
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
<div class="hint">Escolha o período, os índices e a área (amostral do passo 1 ou um polígono da fazenda), depois use o carrossel de datas para alternar a imagem (uma única cena por data). Abaixo, a série temporal mostra a média no polígono.</div>
""", unsafe_allow_html=True)

# ----------------------------
# Util: último salvamento do passo 1
# ----------------------------
def _find_latest_save_dir(base="/tmp/streamlit_dados"):
    if not os.path.isdir(base): return None
    candidates = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not candidates: return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def load_area_from_tmp():
    latest = _find_latest_save_dir()
    if not latest: return None, None
    area_path = os.path.join(latest, "area_amostral.gpkg")
    if not os.path.exists(area_path): return None, None
    gdf_area = gpd.read_file(area_path)
    if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
        gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)
    return gdf_area, latest

# ----------------------------
# EE init via env var
# ----------------------------
def ensure_ee_init():
    try:
        _ = ee.Number(1).getInfo()
        return
    except Exception:
        pass
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

# ----------------------------
# Ranges padrão por índice + Paletas
# ----------------------------
INDEX_DEFAULT = {
    "NDVI":   dict(min=0.0,  max=1.0),
    "GNDVI":  dict(min=0.0,  max=1.0),
    "NDRE":   dict(min=0.0,  max=0.5),
    "CCCI":   dict(min=0.0,  max=2.0),
    "MSAVI2": dict(min=0.0,  max=1.0),
    "NDWI":   dict(min=-1.0, max=1.0),
    "NDMI":   dict(min=-1.0, max=1.0),
    "NBR":    dict(min=-1.0, max=1.0),
    "TWI2":   dict(min=-1.0, max=1.0),
}

PALETTES = {
    "YlGn":        ["#ffffcc","#c2e699","#78c679","#31a354","#006837"],
    "Viridis":     ["#440154","#3b528b","#21918c","#5ec962","#fde725"],
    "Plasma":      ["#0d0887","#6a00a8","#b12a90","#e16462","#fca636","#f0f921"],
    "Magma":       ["#000004","#1f0c48","#550f6d","#88226a","#b63655","#fb8260"],
    "Inferno":     ["#000004","#1b0c41","#4a0c6b","#781c6d","#a52c60","#f07042"],
    "Turbo":       ["#30123b","#4145ab","#2ab7c4","#7bd151","#fde725"],
    "Cividis":     ["#00204c","#394b76","#7c7b78","#b7ad6b","#ffd43b"],
    "Spectral":    ["#9e0142","#f46d43","#fdae61","#fee08b","#e6f598","#abdda4","#66c2a5","#3288bd","#5e4fa2"],
    "RdYlGn":      ["#a50026","#d73027","#f46d43","#fdae61","#fee08b","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"],
    "BrBG":        ["#543005","#8c510a","#bf812d","#dfc27d","#f6e8c3","#c7eae5","#80cdc1","#35978f","#01665e","#003c30"],
}

# ----------------------------
# Sidebar – controles
# ----------------------------
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
        list(INDEX_DEFAULT.keys()),
        default=["NDVI", "GNDVI", "NDRE", "MSAVI2", "NDWI"]
    )

    cloud_thr    = st.slider("Nuvem máxima (%)", 0, 60, 10, 1)
    palette_name = st.selectbox("Paleta de cores", list(PALETTES.keys()), index=0)
    auto_stretch = st.checkbox("Ajuste automático 2–98% (por data/índice)", value=True)
    show_rgb     = st.checkbox("Adicionar camada RGB (B4,B3,B2)", value=True)
    legend_for   = st.selectbox("Legenda para índice", options=indices_sel if indices_sel else ["NDVI"])
    btn = st.button("▶️ Processar")

# ----------------------------
# Carregar área
# ----------------------------
gdf_area = None
if area_opt.startswith("Usar"):
    gdf_area, _ = load_area_from_tmp()
    if gdf_area is None:
        st.warning("Não encontrei a área amostral salva. Volte ao passo 1 e clique em **Salvar dados**.")
else:
    up = st.file_uploader("Carregue um polígono (GPKG)", type=["gpkg"])
    if up:
        tmp = "/tmp/_poly_upload.gpkg"
        with open(tmp, "wb") as f: f.write(up.getbuffer())
        gdf_area = gpd.read_file(tmp)
        if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
            gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)
if gdf_area is None:
    st.stop()

# Polígono EE
try:
    ee_poly = geemap.gdf_to_ee(gdf_area[["geometry"]])
except Exception as e:
    st.error(f"Falha ao converter polígono para EE: {e}")
    st.stop()

# ----------------------------
# Funções EE
# ----------------------------
def add_indices(img, wanted):
    def nd(a, b): return img.normalizedDifference([a, b])
    out = img
    if "NDVI"   in wanted: out = out.addBands(nd("B8", "B4").rename("NDVI"))
    if "GNDVI"  in wanted: out = out.addBands(nd("B8", "B3").rename("GNDVI"))
    if "NDRE"   in wanted: out = out.addBands(nd("B8", "B5").rename("NDRE"))
    if "CCCI"   in wanted:
        ndre = nd("B8", "B5"); ndvi = nd("B8", "B4")
        out  = out.addBands(ndre.divide(ndvi).rename("CCCI"))
    if "MSAVI2" in wanted:
        msavi2 = img.expression(
            "(2*NIR + 1 - sqrt((2*NIR + 1)**2 - 8*(NIR - RED)))/2",
            {"NIR": img.select("B8"), "RED": img.select("B4")}
        ).rename("MSAVI2")
        out = out.addBands(msavi2)
    if "NDWI"   in wanted: out = out.addBands(nd("B3", "B8").rename("NDWI"))
    if "NDMI"   in wanted: out = out.addBands(nd("B8", "B11").rename("NDMI"))
    if "NBR"    in wanted: out = out.addBands(nd("B8", "B12").rename("NBR"))
    if "TWI2"   in wanted: out = out.addBands(nd("B9", "B8").rename("TWI2"))
    return out

def ee_tile_url(image, vis):
    return ee.Image(image).getMapId(vis)["tile_fetcher"].url_format

def reduce_percentiles(img, band, geom, scale=10):
    d = img.select(band).reduceRegion(
        reducer=ee.Reducer.percentile([2,98]),
        geometry=geom, scale=scale, maxPixels=1e9, bestEffort=True
    )
    p2  = d.get(f"{band}_p2")
    p98 = d.get(f"{band}_p98")
    p2  = None if p2 is None else ee.Number(p2).getInfo()
    p98 = None if p98 is None else ee.Number(p98).getInfo()
    return p2, p98

def time_series_mean(ic, ee_geom, indices, scale=10):
    def per_image(img):
        stats = ee.Dictionary({})
        for name in indices:
            mean_val = img.select(name).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=ee_geom,
                scale=scale, maxPixels=1e9, bestEffort=True
            ).get(name)
            stats = stats.set(name, mean_val)
        date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
        return ee.Feature(None, stats.set('date', date_str))
    fc = ee.FeatureCollection(ic.map(per_image))
    dates = fc.aggregate_array('date').getInfo()
    data = {'date': pd.to_datetime(dates)}
    for name in indices:
        data[name] = fc.aggregate_array(name).getInfo()
    return pd.DataFrame(data).sort_values('date').reset_index(drop=True)

def list_dates(ic):
    """Retorna lista de datas (YYYY-MM-dd) com imagens válidas no período."""
    dates = (ic.aggregate_array("system:time_start")
               .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
               .distinct()
             ).getInfo()
    return sorted(list(set(dates)))

def best_image_on_date(ic, date_str, geom):
    """Para a data desejada, pega a cena menos nublada (single scene), recorta no polígono."""
    d0 = ee.Date(date_str)
    d1 = d0.advance(1, "day")
    sub = ic.filterDate(d0, d1).sort("CLOUDY_PIXEL_PERCENTAGE")
    if sub.size().getInfo() == 0:
        return None, None
    img = sub.first().clip(geom)
    img_date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    return img, img_date

# ----------------------------
# Estado
# ----------------------------
for key, default in [
    ("mon_dates", None),         # lista de datas
    ("mon_selected_date", None), # string YYYY-MM-dd
    ("mon_tiles", None),         # {idx: tile_url} + opcional 'RGB'
    ("mon_vis", None),           # {idx: {min,max,palette}}
    ("mon_series", None),        # DataFrame
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------
# Processamento principal
# ----------------------------
def build_collection():
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
             .filterBounds(ee_poly.geometry())
             .filterDate(ee.Date(str(start)), ee.Date(str(end)))
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thr))
             .map(lambda im: add_indices(im, indices_sel)))

def build_tiles_for_image(img):
    tiles = {}
    vis_dict = {}
    palette = PALETTES[palette_name]

    # índices
    for idx in indices_sel:
        if auto_stretch:
            try:
                p2, p98 = reduce_percentiles(img, idx, ee_poly, scale=10)
                if p2 is None or p98 is None or not np.isfinite(p2) or not np.isfinite(p98) or p2 >= p98:
                    vmin, vmax = INDEX_DEFAULT[idx]["min"], INDEX_DEFAULT[idx]["max"]
                else:
                    vmin, vmax = p2, p98
            except Exception:
                vmin, vmax = INDEX_DEFAULT[idx]["min"], INDEX_DEFAULT[idx]["max"]
        else:
            vmin, vmax = INDEX_DEFAULT[idx]["min"], INDEX_DEFAULT[idx]["max"]

        vis = {"min": float(vmin), "max": float(vmax), "palette": palette}
        tiles[idx] = ee_tile_url(img.select(idx), vis)
        vis_dict[idx] = vis

    # RGB opcional (usa min/max padrão Sentinel-2 SR)
    if show_rgb:
        rgb_vis = {"bands": ["B4","B3","B2"], "min": 0, "max": 3000, "gamma": 1.2}
        tiles["RGB"] = ee_tile_url(img, rgb_vis)
        vis_dict["RGB"] = rgb_vis

    return tiles, vis_dict

def do_process():
    if len(indices_sel) == 0:
        st.warning("Selecione pelo menos um índice.")
        st.stop()

    ic = build_collection()
    if ic.size().getInfo() == 0:
        st.warning("Não há imagens disponíveis no período/limite de nuvens escolhido.")
        st.stop()

    # lista de datas disponíveis (carrossel)
    dates = list_dates(ic)
    if not dates:
        st.warning("Não há datas válidas no período selecionado.")
        st.stop()

    # série temporal (média no polígono) com TODAS as imagens
    df_ts = time_series_mean(ic, ee_poly, indices_sel, scale=10)

    # primeira data selecionada (ou mantem se já existir e estiver na lista)
    sel_date = st.session_state["mon_selected_date"]
    if sel_date not in dates:
        sel_date = dates[0]

    # imagem ÚNICA da data escolhida
    img, real_date = best_image_on_date(ic, sel_date, ee_poly)
    if img is None:
        # fallback: tenta próxima data disponível
        for d in dates:
            img, real_date = best_image_on_date(ic, d, ee_poly)
            if img is not None:
                sel_date = d
                break
        if img is None:
            st.warning("Não foi possível obter uma cena válida para as datas disponíveis.")
            st.stop()

    tiles, vis_dict = build_tiles_for_image(img)

    st.session_state["mon_dates"]         = dates
    st.session_state["mon_selected_date"] = real_date  # usa a data real da imagem
    st.session_state["mon_tiles"]         = tiles
    st.session_state["mon_vis"]           = vis_dict
    st.session_state["mon_series"]        = df_ts

if btn:
    with st.spinner("Processando imagens e gerando camadas..."):
        try:
            do_process()
            st.success("Pronto! Use o seletor de datas, veja o mapa e a série temporal abaixo.")
        except Exception as e:
            st.error(f"Falha no processamento: {e}")

# ----------------------------
# Seletor de datas (carrossel) e re-build quando muda
# ----------------------------
dates = st.session_state.get("mon_dates")
if dates:
    # slider de datas
    st.subheader("🗓️ Selecione a data da cena")
    sel = st.select_slider("Data da imagem", options=dates, value=st.session_state["mon_selected_date"])
    if sel != st.session_state["mon_selected_date"]:
        # rebuild apenas para essa data
        ic = build_collection()
        img, real_date = best_image_on_date(ic, sel, ee_poly)
        if img is not None:
            tiles, vis_dict = build_tiles_for_image(img)
            st.session_state["mon_selected_date"] = real_date
            st.session_state["mon_tiles"] = tiles
            st.session_state["mon_vis"] = vis_dict

# ----------------------------
# Render (mapa + legenda + série)
# ----------------------------
tiles   = st.session_state.get("mon_tiles")
series  = st.session_state.get("mon_series")
visdict = st.session_state.get("mon_vis")
seldate = st.session_state.get("mon_selected_date")

if tiles:
    center = [gdf_area.geometry.unary_union.centroid.y, gdf_area.geometry.unary_union.centroid.x]
    m = folium.Map(location=center, zoom_start=16, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satélite", overlay=False, control=True
    ).add_to(m)

    # polígono
    folium.GeoJson(
        gdf_area[["geometry"]].__geo_interface__,
        name="Área",
        style_function=lambda x: {"color":"#1976d2","weight":2,"fillColor":"#1976d2","fillOpacity":0.08}
    ).add_to(m)

    # camadas por índice + RGB
    first = True
    for layer_name, url in tiles.items():
        folium.raster_layers.TileLayer(
            tiles=url, name=f"{layer_name}", attr="Google Earth Engine",
            overlay=True, control=True, show=first, opacity=0.95
        ).add_to(m)
        first = False

    # legenda (índice escolhido no seletor "Legenda para índice")
    legend_index = legend_for if legend_for in tiles else (list(tiles.keys())[0] if tiles else "NDVI")
    vis = visdict.get(legend_index, {"min":0, "max":1, "palette": PALETTES["YlGn"]})
    vmin, vmax = vis.get("min", 0), vis.get("max", 1)
    grad = ",".join(vis.get("palette", PALETTES["YlGn"]))
    legend_html = f"""
    <div class="legend-box" style="--grad: {grad};">
      <div class="legend-title">{legend_index}</div>
      <div class="legend-gradient"></div>
      <div class="legend-scale"><span>{vmin:.2f}</span><span>{vmax:.2f}</span></div>
      <div class="legend-meta">Imagem: <b>{seldate or '-'}</b></div>
      <div class="legend-meta">Paleta: {palette_name}{' (auto 2–98%)' if auto_stretch else ''}</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=1000, height=640)

    # Série temporal
    if isinstance(series, pd.DataFrame) and not series.empty:
        st.subheader("📈 Série temporal (média do índice no polígono)")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        series_plot = series.copy()
        series_plot["date"] = pd.to_datetime(series_plot["date"])
        for idx in [c for c in series_plot.columns if c != "date"]:
            ax.plot(series_plot["date"], series_plot[idx], label=idx)
        ax.set_xlabel("Data")
        ax.set_ylabel("Valor médio")
        ax.grid(True, alpha=.3)
        ax.legend(ncols=min(4, len(series_plot.columns)-1), fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
else:
    st.info("Aguardando processamento. Defina período/índices e clique em **Processar**.")
