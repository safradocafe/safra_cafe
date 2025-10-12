# pages/3_Monitoramento.py
import os, glob, json, tempfile
import geopandas as gpd
import pandas as pd
import numpy as np
import streamlit as st
import ee, geemap
import folium
from streamlit_folium import st_folium

# -----------------------
# Aparência
# -----------------------
st.set_page_config(layout="wide")
st.markdown("""
<style>
.block-container { padding-top: .4rem !important; padding-bottom: .6rem !important; }
header, footer {visibility:hidden;}
.badges {display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0 .8rem 0;}
.badge {background:#f3f4f6; border:1px solid #e5e7eb; border-radius:999px; padding:.25rem .6rem; font-size:0.85rem;}
.badge b{margin-right:.35rem}
.note {font-size:.9rem; color:#4b5563;}
.legend-box {background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:.5rem .75rem; font-size:.85rem; box-shadow:0 1px 2px rgba(0,0,0,.04);}
</style>
""", unsafe_allow_html=True)

st.markdown("<h3 style='margin:0 0 .5rem 0;'>🛰️ Monitoramento com índices espectrais (Sentinel-2)</h3>", unsafe_allow_html=True)

# Badges com explicações (resumo do seu texto)
st.markdown("""
<div class="badges">
  <span class="badge"><b>NDVI</b> vigor vegetativo</span>
  <span class="badge"><b>GNDVI</b> sensível ao N foliar</span>
  <span class="badge"><b>NDRE</b> sensível à clorofila</span>
  <span class="badge"><b>CCCI</b> sensível à clorofila na copa das plantas</span>
  <span class="badge"><b>MSAVI2</b> reduz a influência do solo</span>
  <span class="badge"><b>NDWI</b> conteúdo de água na folha</span>
  <span class="badge"><b>NDMI</b> umidade no dossel</span>
  <span class="badge"><b>NBR</b> sensível a estresse térmico</span>
  <span class="badge"><b>TWI2</b> umidade do ar</span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="note">Escolha o período, os índices e a área (amostral salva no passo 1 ou um polígono da fazenda) para visualizar no mapa e ver a série temporal (média no polígono).</div>', unsafe_allow_html=True)

# -----------------------
# Utilidades
# -----------------------
def _find_latest_area():
    base = "/tmp/streamlit_dados"
    if not os.path.isdir(base): return None
    candidates = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not candidates: return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    area = os.path.join(candidates[0], "area_amostral.gpkg")
    return area if os.path.exists(area) else None

import os, json, ee, streamlit as st

def init_ee():
    try:
        if "GEE_SA_KEY_JSON" in os.environ:
            creds_dict = json.loads(os.environ["GEE_SA_KEY_JSON"])
            ee.Initialize(ee.ServiceAccountCredentials(
                creds_dict["client_email"],
                key_data=json.dumps(creds_dict)
            ))
        else:
            raise RuntimeError("Variável GEE_SA_KEY_JSON não encontrada no ambiente.")
    except Exception as e:
        st.error(f"Erro ao inicializar o Google Earth Engine: {e}")
        st.stop()


# Paletas e faixas por índice (ajustáveis)
INDEX_VIS = {
    "NDVI":  {"min": -0.2, "max": 0.9, "palette": ["#f7fbf0","#c7e9c0","#74c476","#238b45","#00441b"]},
    "GNDVI": {"min": -0.2, "max": 0.9, "palette": ["#f7fbf0","#d9f0a3","#78c679","#238443","#004529"]},
    "NDRE":  {"min": -0.2, "max": 0.6, "palette": ["#f7f4f9","#c7a9cf","#7f62a8","#4c2c7f","#2d115a"]},
    "CCCI":  {"min":  0.0, "max": 1.5, "palette": ["#f7fcf0","#a1dab4","#41b6c4","#2c7fb8","#253494"]},
    "MSAVI2":{"min":  0.0, "max": 1.0, "palette": ["#fff7ec","#fee8c8","#fdbb84","#e34a33","#7f0000"]},
    "NDWI":  {"min": -0.5, "max": 0.7, "palette": ["#fff5f0","#fcbba1","#fc9272","#9ecae1","#08519c"]},
    "NDMI":  {"min": -0.5, "max": 0.7, "palette": ["#ffffe5","#f7fcb9","#addd8e","#31a354","#006837"]},
    "NBR":   {"min": -0.5, "max": 0.8, "palette": ["#ffffe5","#fee391","#fec44f","#fe9929","#d95f0e"]},
    "TWI2":  {"min": -0.5, "max": 0.8, "palette": ["#fff7fb","#ece7f2","#a6bddb","#2b8cbe","#014636"]},
}

# Cálculo dos índices
def add_indices(img, wanted):
    bands = {}
    if "NDVI" in wanted:  bands["NDVI"]  = img.normalizedDifference(["B8","B4"]).rename("NDVI")
    if "GNDVI" in wanted: bands["GNDVI"] = img.normalizedDifference(["B8","B3"]).rename("GNDVI")
    if "NDRE" in wanted:  bands["NDRE"]  = img.normalizedDifference(["B8","B5"]).rename("NDRE")
    if "CCCI" in wanted:
        ndvi = img.normalizedDifference(["B8","B4"])
        ndre = img.normalizedDifference(["B8","B5"])
        bands["CCCI"] = ndre.divide(ndvi).rename("CCCI")
    if "MSAVI2" in wanted:
        msavi2 = img.expression(
            "(2*NIR + 1 - sqrt((2*NIR + 1)**2 - 8*(NIR - RED)))/2",
            {"NIR": img.select("B8"), "RED": img.select("B4")}
        ).rename("MSAVI2")
        bands["MSAVI2"] = msavi2
    if "NDWI" in wanted:  bands["NDWI"]  = img.normalizedDifference(["B3","B8"]).rename("NDWI")
    if "NDMI" in wanted:  bands["NDMI"]  = img.normalizedDifference(["B8","B11"]).rename("NDMI")
    if "NBR" in wanted:   bands["NBR"]   = img.normalizedDifference(["B8","B12"]).rename("NBR")
    if "TWI2" in wanted:  bands["TWI2"]  = img.normalizedDifference(["B9","B8"]).rename("TWI2")
    # Junta as bandas pedidas
    if bands:
        img = img.addBands(ee.Image.cat(list(bands.values())))
    return img

# Série temporal (média no polígono) para cada índice
def time_series_mean(ic, geom, indices):
    rows = []
    for idx in indices:
        col = ic.select([idx])
        # mean por imagem no polígono
        def _reduce(img):
            d = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            mean = img.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9).get(idx)
            return ee.Feature(None, {"date": d, "index": idx, "mean": mean})
        feats = col.map(_reduce)
        rows += feats.getInfo()["features"]
    # Converte em DataFrame
    data = [{"date": f["properties"]["date"],
             "index": f["properties"]["index"],
             "mean":  f["properties"]["mean"]} for f in rows if f["properties"]["mean"] is not None]
    if not data:
        return pd.DataFrame(columns=["date","index","mean"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["index","date"])
    return df

# -----------------------
# Sidebar – Controles
# -----------------------
init_ee()

with st.sidebar:
    st.subheader("Área de interesse")
    area_choice = st.radio("Usar:", ["Área amostral (passo 1)", "Upload polígono (fazenda)"], index=0)

    gdf_area = None
    if area_choice == "Área amostral (passo 1)":
        area_path = _find_latest_area()
        if area_path:
            gdf_area = gpd.read_file(area_path)
        else:
            st.warning("Não encontrei a área amostral em /tmp. Volte ao passo 1 e clique em “Salvar dados”.")
    else:
        up = st.file_uploader("Polígono da fazenda (GPKG ou GeoJSON)", type=["gpkg","geojson"])
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
                tmp.write(up.getvalue()); tmp.flush()
                gdf_area = gpd.read_file(tmp.name)

    st.subheader("Período e índices")
    colA, colB = st.columns(2)
    with colA: start_date = st.date_input("Início", value=pd.to_datetime("2023-08-01"))
    with colB: end_date   = st.date_input("Fim",    value=pd.to_datetime("2024-05-31"))
    indices = st.multiselect(
        "Índices",
        ["NDVI","GNDVI","NDRE","CCCI","MSAVI2","NDWI","NDMI","NBR","TWI2"],
        default=["NDVI","NDRE","GNDVI"]
    )
    cloud_perc = st.slider("Máx. nuvens (%)", 0, 60, 10, 1)

go = st.sidebar.button("▶️ Processar")

# -----------------------
# Execução
# -----------------------
if not go:
    st.info("Defina a área, o período e os índices e clique em **Processar**.")
    st.stop()

if gdf_area is None or gdf_area.empty:
    st.error("Área inválida ou não carregada.")
    st.stop()

# Garante CRS
if gdf_area.crs is None:
    gdf_area = gdf_area.set_crs(4326)
elif gdf_area.crs.to_epsg() != 4326:
    gdf_area = gdf_area.to_crs(4326)

ee_geom = geemap.gdf_to_ee(gdf_area[["geometry"]])

# Coleção Sentinel-2 + índices
ic = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(ee_geom.geometry())
      .filterDate(ee.Date(str(start_date)), ee.Date(str(end_date)))
      .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_perc))
      .map(lambda img: add_indices(img, indices)))

# Garante que haja imagens
count = ic.size().getInfo()
if count == 0:
    st.warning("Nenhuma imagem encontrada com os filtros escolhidos.")
    st.stop()

# Mosaico (mediana) por índice para visualização
center = gdf_area.geometry.unary_union.centroid
m = folium.Map(location=[center.y, center.x], zoom_start=16, tiles="OpenStreetMap")
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satélite', overlay=False, control=True
).add_to(m)

# camada da área
folium.GeoJson(
    gdf_area[['geometry']].__geo_interface__,
    name="Área de interesse",
    style_function=lambda _: {"color":"#2563eb","weight":2,"fillColor":"#60a5fa","fillOpacity":0.08}
).add_to(m)

# Adiciona uma camada por índice
for idx in indices:
    vis = INDEX_VIS.get(idx, {"min":-0.2,"max":0.8,"palette":["#ffffff","#000000"]})
    median_img = ic.select(idx).median()
    geemap.folium_add_ee_layer(
        m, median_img, vis_params={"min":vis["min"],"max":vis["max"],"palette":vis["palette"]},
        name=f"{idx} (mediana)"
    )

folium.LayerControl(collapsed=False).add_to(m)

# Layout: mapa + “legendas por índice”
col_map, col_leg = st.columns([3,1], gap="large")
with col_map:
    st_folium(m, width=900, height=600, key="gee_indices_map")

with col_leg:
    st.markdown("**Legendas (faixa de valores)**")
    for idx in indices:
        v = INDEX_VIS[idx]
        # desenha uma pequena barra com CSS inline
        grad = f"linear-gradient(90deg, {', '.join(v['palette'])})"
        st.markdown(
            f"""
            <div class="legend-box" style="margin-bottom:.6rem;">
              <div style="font-weight:600; margin-bottom:.25rem;">{idx}</div>
              <div style="height:12px; border-radius:6px; background: {grad};"></div>
              <div style="display:flex; justify-content:space-between; font-size:.8rem; margin-top:.15rem;">
                <span>{v['min']}</span><span>{v['max']}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# -----------------------
# Série temporal (média no polígono)
# -----------------------
st.subheader("📈 Série temporal – média do índice no polígono")
df_ts = time_series_mean(ic, ee_geom.geometry(), indices)

if df_ts.empty:
    st.info("Não foi possível calcular a série temporal (médias nulas).")
else:
    # pivota para gráfico fácil
    pivot = df_ts.pivot_table(index="date", columns="index", values="mean")
    st.line_chart(pivot, height=300, use_container_width=True)
    with st.expander("Ver tabela (médias por data)"):
        st.dataframe(pivot.reset_index())

