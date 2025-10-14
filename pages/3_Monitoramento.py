import os
import json
import glob
from datetime import date

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from branca.element import Template, MacroElement
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex

# -------------------------
# Estilo e cabeçalho
# -------------------------
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

# -------------------------
# Utilitários
# -------------------------
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
    if not os.path.exists(area_path):
        return None, None
    gdf_area = gpd.read_file(area_path)
    if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
        gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)
    return gdf_area, latest

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

def mpl_palette(name: str, n: int = 7):
    cmap = cm.get_cmap(name)
    return [to_hex(cmap(x)) for x in np.linspace(0, 1, n)]

PALETTES = {
    "YlGn": mpl_palette("YlGn"),
    "viridis": mpl_palette("viridis"),
    "plasma": mpl_palette("plasma"),
    "magma": mpl_palette("magma"),
    "cividis": mpl_palette("cividis"),
    "turbo": mpl_palette("turbo"),
    "Spectral": mpl_palette("Spectral"),
    "RdYlGn": mpl_palette("RdYlGn"),
    "BrBG": mpl_palette("BrBG"),
    "PuBuGn": mpl_palette("PuBuGn"),
}

INDEX_RANGES = {
    "NDVI":   dict(min=0.0, max=1.0),
    "GNDVI":  dict(min=0.0, max=1.0),
    "NDRE":   dict(min=0.0, max=0.5),
    "CCCI":   dict(min=0.0, max=2.0),
    "MSAVI2": dict(min=0.0, max=1.0),
    "NDWI":   dict(min=-1.0, max=1.0),
    "NDMI":   dict(min=-1.0, max=1.0),
    "NBR":    dict(min=-1.0, max=1.0),
    "TWI2":   dict(min=-1.0, max=1.0),
}

# -------------------------
# Sidebar – controles (entrada)
# -------------------------
with st.sidebar:
    st.subheader("Configurações")
    area_opt = st.radio(
        "Área de interesse:",
        ["Usar área amostral salva (passo 1)", "Fazer upload de polígono (GPKG)"],
        index=0
    )
    c1, c2 = st.columns(2)
    start = c1.date_input("Início", value=date(2024, 1, 1))
    end   = c2.date_input("Fim", value=date.today())
    indices_sel = st.multiselect(
        "Índices para processar (série temporal)",
        list(INDEX_RANGES.keys()),
        default=["NDVI", "GNDVI", "NDRE", "MSAVI2", "NDWI"]
    )
    palette_name = st.selectbox("Paleta de cores (mapa)", list(PALETTES.keys()), index=0)
    cloud_thr = st.slider("Nuvem máxima (%)", 0, 60, 10, 1)
    btn = st.button("▶️ Processar")

# -------------------------
# Carregar área
# -------------------------
gdf_area = None
base_dir = None
if area_opt.startswith("Usar"):
    gdf_area, base_dir = load_area_from_tmp()
    if gdf_area is None:
        st.warning("Não encontrei a área amostral salva. Volte ao passo 1 e clique em **Salvar dados**.")
else:
    up = st.file_uploader("Carregue um polígono (GPKG)", type=["gpkg"])
    if up:
        tmp = "/tmp/_poly_upload.gpkg"
        with open(tmp, "wb") as f:
            f.write(up.getbuffer())
        gdf_area = gpd.read_file(tmp)
        if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
            gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)

if gdf_area is None:
    st.stop()

try:
    ee_poly = geemap.gdf_to_ee(gdf_area[["geometry"]])
except Exception as e:
    st.error(f"Falha ao converter polígono para EE: {e}")
    st.stop()

# -------------------------
# Funções EE
# -------------------------
def add_indices(img, wanted):
    def nd(a, b):
        return img.normalizedDifference([a, b])
    out = img
    if "NDVI" in wanted:
        out = out.addBands(nd("B8", "B4").rename("NDVI"))
    if "GNDVI" in wanted:
        out = out.addBands(nd("B8", "B3").rename("GNDVI"))
    if "NDRE" in wanted:
        out = out.addBands(nd("B8", "B5").rename("NDRE"))
    if "CCCI" in wanted:
        ndre = nd("B8", "B5")
        ndvi = nd("B8", "B4")
        out  = out.addBands(ndre.divide(ndvi).rename("CCCI"))
    if "MSAVI2" in wanted:
        msavi2 = img.expression(
            "(2*NIR + 1 - sqrt((2*NIR + 1)**2 - 8*(NIR - RED)))/2",
            {"NIR": img.select("B8"), "RED": img.select("B4")}
        ).rename("MSAVI2")
        out = out.addBands(msavi2)
    if "NDWI" in wanted:
        out = out.addBands(nd("B3", "B8").rename("NDWI"))
    if "NDMI" in wanted:
        out = out.addBands(nd("B8", "B11").rename("NDMI"))
    if "NBR" in wanted:
        out = out.addBands(nd("B8", "B12").rename("NBR"))
    if "TWI2" in wanted:
        out = out.addBands(nd("B9", "B8").rename("TWI2"))
    return out

def time_series_mean(ic, ee_geom, indices, scale=10):
    def per_image(img):
        stats = ee.Dictionary({})
        for name in indices:
            mean_val = img.select(name).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geom,
                scale=scale,
                maxPixels=1e9,
                bestEffort=True
            ).get(name)
            stats = stats.set(name, mean_val)
        date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
        return ee.Feature(None, stats.set('date', date_str))

    fc = ee.FeatureCollection(ic.map(per_image))
    dates = fc.aggregate_array('date').getInfo()
    data = {'date': pd.to_datetime(dates)}
    for name in indices:
        data[name] = fc.aggregate_array(name).getInfo()
    df = pd.DataFrame(data).sort_values('date').reset_index(drop=True)
    return df

def ee_tile_url(image, vis):
    m = ee.Image(image).getMapId(vis)
    return m["tile_fetcher"].url_format

def get_dates_and_ts(ee_geom, start_d, end_d, indices, cloud):
    base = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(ee_geom)
            .filterDate(ee.Date(str(start_d)), ee.Date(str(end_d)))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud)))
    dates = (base.aggregate_array("system:time_start")
             .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
             .distinct()
             .getInfo())
    dates = sorted(list(set(dates)))
    if not dates:
        return [], pd.DataFrame()
    col_idx = base.map(lambda im: add_indices(im, indices))
    ts_df = time_series_mean(col_idx, ee_geom, indices, scale=10)
    return dates, ts_df

def get_best_image_for_date(ee_geom, date_str, cloud, indices):
    d0 = ee.Date(date_str)
    d1 = d0.advance(1, "day")
    daycol = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(ee_geom)
              .filterDate(d0, d1)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud))
              .sort('CLOUDY_PIXEL_PERCENTAGE', True))
    img = ee.Image(daycol.first())
    img = ee.Image(ee.Algorithms.If(img, img, ee.Image(daycol.median())))
    img = add_indices(ee.Image(img), INDEX_RANGES.keys()).clip(ee_poly)
    return img

def add_linear_legend(fmap, title, palette, vmin, vmax, date_str=None, position="bottomright"):
    pos_css = {
        "bottomright": "bottom: 12px; right: 12px;",
        "bottomleft":  "bottom: 12px; left: 12px;",
        "topright":    "top: 12px; right: 12px;",
        "topleft":     "top: 12px; left: 12px;",
    }.get(position, "bottom: 12px; right: 12px;")
    gradient = "linear-gradient(to right, " + ", ".join(palette) + ")"
    date_html = f"<div style='font-size:.8rem; color:#111; margin-top:2px;'>Data: <b>{date_str}</b></div>" if date_str else ""
    html = f"""
    <div style="
        position:absolute; {pos_css}
        z-index:9999; background:rgba(255,255,255,.92);
        border:1px solid #bbb; border-radius:6px;
        box-shadow:0 1px 6px rgba(0,0,0,.15);
        padding:.5rem .6rem .6rem .6rem;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">
      <div style="font-weight:700; font-size:.9rem; margin-bottom:.35rem;">{title}</div>
      <div style="width:220px; height:12px; background:{gradient}; border-radius:3px; border:1px solid #aaa;"></div>
      <div style="display:flex; justify-content:space-between; font-size:.8rem; color:#222; margin-top:2px;">
        <span>{vmin:.2f}</span><span>{vmax:.2f}</span>
      </div>
      {date_html}
    </div>
    """
    tmpl = Template(html)
    macro = MacroElement()
    macro._template = tmpl
    fmap.get_root().add_child(macro)

# -------------------------
# Estado mínimo (datas + série)
# -------------------------
if "mon_dates" not in st.session_state:    st.session_state["mon_dates"] = []
if "mon_ts" not in st.session_state:       st.session_state["mon_ts"] = pd.DataFrame()

# Processar (calcula datas e série; o mapa é sempre renderizado conforme seleção atual)
if btn:
    with st.spinner("Processando imagens, listando datas e calculando séries..."):
        try:
            dates, ts_df = get_dates_and_ts(ee_poly.geometry(), start, end, indices_sel, cloud_thr)
            if not dates:
                st.warning("Nenhuma imagem encontrada no período com o limite de nuvem escolhido.")
            st.session_state["mon_dates"] = dates
            st.session_state["mon_ts"] = ts_df
            st.success("Pronto! Use os controles abaixo para visualizar.")
        except Exception as e:
            st.error(f"Falha no processamento: {e}")

# -------------------------
# Controles pós-processamento (visual)
# -------------------------
dates = st.session_state.get("mon_dates", [])
ts_df = st.session_state.get("mon_ts", pd.DataFrame())

if dates:
    # UI de visualização (sem guardar ativo em estado; recalcula no ato)
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        # Radio: qual camada exibir no mapa (índices + RGB)
        idx_for_map = st.radio(
            "Camada do mapa",
            options=["RGB (B4/B3/B2)"] + indices_sel,
            index=0
        )
    with c2:
        date_choice = st.select_slider("Data (cena única)", options=dates, value=dates[0])
    with c3:
        palette_name = st.selectbox("Paleta (mapa)", list(PALETTES.keys()), index=list(PALETTES.keys()).index(palette_name))

    # Monta a imagem única para a data escolhida (menor nuvem do dia)
    img_date = get_best_image_for_date(ee_poly.geometry(), date_choice, cloud_thr, indices_sel)

    # Construção do mapa
    center = [gdf_area.geometry.unary_union.centroid.y, gdf_area.geometry.unary_union.centroid.x]
    m = folium.Map(location=center, zoom_start=16, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satélite", overlay=False, control=True
    ).add_to(m)

    folium.GeoJson(
        gdf_area[["geometry"]].__geo_interface__,
        name="Área",
        style_function=lambda x: {"color":"#1976d2","weight":2,"fillColor":"#1976d2","fillOpacity":0.08}
    ).add_to(m)

    # Camada do mapa: RGB ou índice
    if idx_for_map == "RGB (B4/B3/B2)":
        rgb_url = ee_tile_url(img_date, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000})
        folium.raster_layers.TileLayer(
            tiles=rgb_url, name="RGB (B4/B3/B2)", attr="Google Earth Engine",
            overlay=True, control=True, show=True, opacity=0.95
        ).add_to(m)
        # legenda simples para RGB
        add_linear_legend(m, "RGB (B4/B3/B2)", ["#000000", "#FFFFFF"], 0, 3000, date_str=date_choice)
    else:
        # índice escolhido
        vis_range = INDEX_RANGES[idx_for_map]
        vis_palette = PALETTES[palette_name]
        idx_url = ee_tile_url(img_date.select(idx_for_map), {**vis_range, "palette": vis_palette})
        folium.raster_layers.TileLayer(
            tiles=idx_url, name=idx_for_map, attr="Google Earth Engine",
            overlay=True, control=True, show=True, opacity=0.95
        ).add_to(m)
        add_linear_legend(
            m, idx_for_map, vis_palette, vis_range["min"], vis_range["max"], date_str=date_choice
        )

    folium.LayerControl(collapsed=False).add_to(m)

    # 🔑 chave do mapa inclui índice + data + paleta => força re-render imediato
    map_key = f"map_{idx_for_map}_{date_choice}_{palette_name}"
    st_folium(m, width=1000, height=620, key=map_key)

    # Série temporal
    if isinstance(ts_df, pd.DataFrame) and not ts_df.empty:
        st.subheader("📈 Série temporal (média do índice no polígono)")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        plot_df = ts_df.copy()
        plot_df["date"] = pd.to_datetime(plot_df["date"])
        for idx in indices_sel:
            if idx in plot_df.columns:
                ax.plot(plot_df["date"], plot_df[idx], label=idx)
        ax.set_xlabel("Data")
        ax.set_ylabel("Valor médio")
        ax.grid(True, alpha=.3)
        ax.legend(ncols=min(4, len(indices_sel)), fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Sem dados suficientes para montar a série temporal nesse período.")
else:
    st.info("Aguardando processamento. Defina período/índices e clique em **Processar**.")
