import os, io, json, glob
from datetime import date
import geopandas as gpd
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import ee
import geemap

# ----------------------------
# Aparência
# ----------------------------
st.set_page_config(layout="wide")
st.markdown("""
<style>
.block-container { padding-top: .25rem !important; padding-bottom: .5rem !important; }
header, footer {visibility:hidden;}
.chips {display:flex; flex-wrap:wrap; gap:.35rem; margin:.25rem 0 1rem 0;}
.chip {background:#f1f3f5; border:1px solid #e5e7eb; padding:.25rem .5rem; border-radius:999px; font-size:.83rem;}
.chip b {margin-right:.35rem;}
.smallnote {font-size:.85rem; color:#555;}
.legend-box {background: rgba(255,255,255,.9); padding:.5rem .65rem; border-radius:.5rem; border:1px solid #e5e7eb;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Utilidades
# ----------------------------
def _find_latest_save_dir(base="/tmp/streamlit_dados"):
    if not os.path.isdir(base):
        return None
    candidates = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def load_area_from_tmp():
    """Busca a última área amostral salva no passo 1: /tmp/streamlit_dados/salvamento-*/area_amostral.gpkg"""
    save_dir = st.session_state.get("tmp_save_dir")
    area_path = st.session_state.get("tmp_area_path")

    if not save_dir or not os.path.exists(save_dir):
        latest = _find_latest_save_dir()
        if latest:
            save_dir = latest
            area_path = os.path.join(save_dir, "area_amostral.gpkg")

    if area_path and os.path.exists(area_path):
        gdf = gpd.read_file(area_path)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
        return gdf, save_dir
    return None, None

def init_ee_or_explain():
    """Inicializa o Earth Engine usando a env var GEE_SA_KEY_JSON (ou st.secrets)"""
    try:
        # 1) Var de ambiente (recomendado no Cloud Run)
        key_json = os.environ.get("GEE_SA_KEY_JSON")
        if key_json:
            creds_dict = json.loads(key_json)
            credentials = ee.ServiceAccountCredentials(
                email=creds_dict["client_email"],
                key_data=key_json,
            )
            ee.Initialize(credentials)
            return True, "GEE inicializado via variável de ambiente."
        # 2) Fallback: st.secrets["GEE_SA_KEY_JSON"]
        if "GEE_SA_KEY_JSON" in st.secrets:
            key_json2 = st.secrets["GEE_SA_KEY_JSON"]
            creds_dict2 = json.loads(key_json2)
            credentials2 = ee.ServiceAccountCredentials(
                email=creds_dict2["client_email"],
                key_data=key_json2,
            )
            ee.Initialize(credentials2)
            return True, "GEE inicializado via secrets."
        return False, "Nenhuma credencial encontrada. Configure a variável de ambiente GEE_SA_KEY_JSON no Cloud Run."
    except Exception as e:
        return False, f"Erro ao inicializar o GEE: {e}"

def add_ee_to_folium(m, ee_image, vis, name="EE layer", shown=True, opacity=1.0):
    """Adiciona uma ee.Image ao folium.Map usando geemap.ee_tile_layer."""
    layer = geemap.ee_tile_layer(
        ee_object=ee_image,
        vis_params=vis,
        name=name,
        shown=shown,
        opacity=opacity
    )
    m.add_child(layer)

# Paletas e bandas por índice (Sentinel-2 SR Harmonized)
INDEX_INFO = {
    "NDVI":  {"bands": ("B8","B4"),  "palette": ["#a6611a","#dfc27d","#80cdc1","#018571"], "vmin": -0.2, "vmax": 0.9, "hint": "vigor vegetativo"},
    "GNDVI": {"bands": ("B8","B3"),  "palette": ["#f7fcf5","#74c476","#238b45","#00441b"], "vmin": -0.2, "vmax": 0.9, "hint": "sensível ao N foliar"},
    "NDRE":  {"bands": ("B8","B5"),  "palette": ["#fff7ec","#fdd49e","#fc8d59","#d7301f"], "vmin": -0.2, "vmax": 0.6, "hint": "sensível à clorofila"},
    "CCCI":  {"bands": None,         "palette": ["#f7fbff","#6baed6","#2171b5","#08306b"], "vmin": 0.0,  "vmax": 2.0, "hint": "clorofila na copa"},
    "MSAVI2":{"bands": ("B8","B4"),  "palette": ["#fff5f0","#fcbba1","#fb6a4a","#cb181d"], "vmin": 0.0,  "vmax": 1.0, "hint": "reduz influência do solo"},
    "NDWI":  {"bands": ("B3","B8"),  "palette": ["#ffffd9","#7fcdbb","#41b6c4","#081d58"], "vmin": -0.5, "vmax": 0.8, "hint": "conteúdo de água na folha"},
    "NDMI":  {"bands": ("B8","B11"), "palette": ["#f7fcf0","#addd8e","#31a354","#006837"], "vmin": -0.5, "vmax": 0.7, "hint": "umidade no dossel"},
    "NBR":   {"bands": ("B8","B12"), "palette": ["#fee5d9","#fcae91","#fb6a4a","#a50f15"], "vmin": -0.5, "vmax": 0.8, "hint": "estresse térmico"},
    "TWI2":  {"bands": ("B9","B8"),  "palette": ["#f7fbff","#deebf7","#9ecae1","#3182bd"], "vmin": -0.5, "vmax": 0.5, "hint": "umidade do ar"},
}

# ----------------------------
# Cabeçalho + chips explicativas
# ----------------------------
st.markdown("<h3 style='margin:.25rem 0 .75rem 0;'>📡 Monitoramento por índices espectrais (Sentinel-2)</h3>", unsafe_allow_html=True)
st.markdown(
    "<div class='chips'>" +
    "".join([f"<div class='chip'><b>{k}</b>{v['hint']}</div>" for k,v in INDEX_INFO.items()]) +
    "</div>",
    unsafe_allow_html=True
)
st.markdown("<div class='smallnote'>Escolha o período, os índices e a área (amostral salva no passo 1 ou um polígono da fazenda) para visualizar no mapa e ver a série temporal (média no polígono).</div>", unsafe_allow_html=True)

# ----------------------------
# Inicializa GEE
# ----------------------------
ok, msg = init_ee_or_explain()
if not ok:
    st.error(msg)
    st.stop()

# ----------------------------
# Escolha da área
# ----------------------------
with st.sidebar:
    st.subheader("Área de interesse")
    area_source = st.radio("Fonte do polígono:", ["Amostral (passo 1)", "Enviar polígono (fazenda)"], index=0)
    gdf_area = None
    base_dir = None

    if area_source == "Amostral (passo 1)":
        gdf_area, base_dir = load_area_from_tmp()
        if gdf_area is not None:
            st.success("Área amostral carregada do passo 1.")
            st.caption(f"Caminho: {base_dir or '/tmp/streamlit_dados'}")
        else:
            st.warning("Não encontrei a área amostral salva. Volte ao passo 1 e clique em 'Salvar dados'.")
    else:
        up = st.file_uploader("Polígono da fazenda (.gpkg)", type=["gpkg"])
        if up:
            tmp = "/tmp/fazenda_area.gpkg"
            with open(tmp, "wb") as f:
                f.write(up.read())
            try:
                gdf_area = gpd.read_file(tmp)
                if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
                    gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)
                st.success("Polígono carregado.")
            except Exception as e:
                st.error(f"Falha ao ler GPKG: {e}")

    st.subheader("Período e índices")
    colA, colB = st.columns(2)
    with colA:
        start_date = st.date_input("Início", value=date(2024,1,1))
    with colB:
        end_date   = st.date_input("Fim", value=date(2024,6,30))
    indices = st.multiselect(
        "Índices",
        options=list(INDEX_INFO.keys()),
        default=["NDVI","NDRE","GNDVI","NDWI"]
    )

    cloud_pct = st.slider("Máx. cobertura de nuvens (%)", 0, 60, 10, 1)
    run_btn = st.button("▶️ Processar")

if gdf_area is None:
    st.stop()

# ----------------------------
# Processamento (carregar coleção, calcular índices, listar datas)
# ----------------------------
if run_btn:
    with st.spinner("Processando imagens Sentinel-2 e calculando índices..."):
        try:
            geom = geemap.gdf_to_ee(gdf_area[['geometry']].copy()).geometry()

            def add_indices(image):
                out = ee.Image(image)
                if "NDVI" in indices:
                    out = out.addBands(image.normalizedDifference(['B8','B4']).rename('NDVI'))
                if "GNDVI" in indices:
                    out = out.addBands(image.normalizedDifference(['B8','B3']).rename('GNDVI'))
                if "NDRE" in indices:
                    out = out.addBands(image.normalizedDifference(['B8','B5']).rename('NDRE'))
                if "MSAVI2" in indices:
                    msavi2 = image.expression(
                        '(2*NIR + 1 - sqrt((2*NIR + 1)**2 - 8*(NIR - RED)))/2',
                        {'NIR': image.select('B8'), 'RED': image.select('B4')}
                    ).rename('MSAVI2')
                    out = out.addBands(msavi2)
                if "NDWI" in indices:
                    out = out.addBands(image.normalizedDifference(['B3','B8']).rename('NDWI'))
                if "NDMI" in indices:
                    out = out.addBands(image.normalizedDifference(['B8','B11']).rename('NDMI'))
                if "NBR" in indices:
                    out = out.addBands(image.normalizedDifference(['B8','B12']).rename('NBR'))
                if "TWI2" in indices:
                    out = out.addBands(image.normalizedDifference(['B9','B8']).rename('TWI2'))
                if "CCCI" in indices:
                    # CCCI = NDRE / NDVI (evita divisão por zero)
                    ndre = out.select('NDRE')
                    ndvi = out.select('NDVI')
                    ccci = ndre.divide(ndvi.where(ndvi.eq(0), 1e-6)).rename("CCCI")
                    out = out.addBands(ccci)
                return out

            col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(geom)
                   .filterDate(ee.Date(start_date.isoformat()), ee.Date(end_date.isoformat()))
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
                   .map(add_indices))

            # Garante que só ficam imagens com o primeiro índice selecionado presente
            if indices:
                col = col.filter(ee.Filter.listContains('system:band_names', indices[0]))

            # Datas distintas
            def add_fmt_date(img):
                return img.set('date_str', ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'))
            col = col.map(add_fmt_date)
            dates = col.aggregate_array('date_str').getInfo()
            dates = sorted(list(set(dates)))

            if not dates:
                st.warning("Nenhuma imagem válida nesse intervalo/área.")
                st.stop()

            st.success(f"Imagens encontradas: {len(dates)} datas")

            # ---------------- Mapas ----------------
            left, right = st.columns([2,1])
            with right:
                sel_date = st.selectbox("Data para visualização no mapa", options=dates, index=0)
                vis_dict = {}
                for idx in indices:
                    info = INDEX_INFO[idx]
                    vis_dict[idx] = {"min": info["vmin"], "max": info["vmax"], "palette": info["palette"]}

                ts_index = st.selectbox("Índice para série temporal (média no polígono)", options=indices, index=0)

            with left:
                # imagem da data escolhida
                img_on_day = ee.Image(col.filter(ee.Filter.eq('date_str', sel_date)).first())

                # mapa centrado no polígono
                center = gdf_area.geometry.unary_union.centroid
                m = folium.Map(location=[center.y, center.x], zoom_start=16, tiles="OpenStreetMap")
                folium.GeoJson(
                    gdf_area[['geometry']].__geo_interface__,
                    name="Área",
                    style_function=lambda x: {"color":"#2563eb","weight":2,"fillOpacity":0.05}
                ).add_to(m)

                # adiciona cada índice como camada
                for idx in indices:
                    vis = vis_dict[idx]
                    if idx in ["CCCI","MSAVI2","NDVI","GNDVI","NDRE","NDWI","NDMI","NBR","TWI2"]:
                        add_ee_to_folium(m, img_on_day.select(idx), vis, name=f"{idx} ({sel_date})")

                folium.LayerControl(collapsed=False).add_to(m)

                # pequena “legenda” compacta
                legend_html = "<div class='legend-box'><b>Camadas:</b><br>" + "<br>".join(
                    [f"- {idx}: {INDEX_INFO[idx]['hint']}" for idx in indices]
                ) + "</div>"
                folium.map.Marker(
                    [center.y, center.x],
                    icon=folium.DivIcon(html=legend_html)
                ).add_to(m)

                st_folium(m, width=920, height=580, key="map_idx")

            # ---------------- Série temporal (média no polígono) ----------------
            st.markdown("### 📈 Série temporal (média no polígono)")
            # para cada data, média da banda ts_index
            def mean_on_date(d):
                im = ee.Image(col.filter(ee.Filter.eq('date_str', d)).first())
                red = im.select(ts_index).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom,
                    scale=10,
                    bestEffort=True,
                    maxPixels=1e9
                )
                return ee.Feature(None, {"date": d, "mean": red.get(ts_index)})

            feats = ee.FeatureCollection(ee.List(dates).map(mean_on_date))
            df = geemap.ee_to_df(feats).sort_values("date")
            df = df[pd.notnull(df["mean"])]

            if df.empty:
                st.info("Sem valores médios disponíveis para a série temporal.")
            else:
                df["date"] = pd.to_datetime(df["date"])
                st.line_chart(df.set_index("date")["mean"], height=260)
                st.caption(f"Índice: **{ts_index}** — valores médios por data dentro do polígono selecionado.")

        except Exception as e:
            st.error(f"Falha no processamento: {e}")
