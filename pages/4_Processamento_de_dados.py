import os, glob, json
from datetime import datetime, date
import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Processamento de dados", layout="wide")
st.title("Processamento de dados") 
st.caption("Seleciona Sentinel-2, calcula índices espectrais e extrai **mínimo, média e máximo** em **buffer** ao redor de **cada ponto** para **cada data**.")
BASE_TMP = "/tmp/streamlit_dados"

def _find_latest_save_dir(base=BASE_TMP):
    if not os.path.isdir(base): return None
    cands = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not cands: return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def load_cloud_inputs():
    """Carrega polígono e pontos salvos no passo 1."""
    latest = _find_latest_save_dir()
    if not latest:
        return None, None, None

    poly_candidates = [
        os.path.join(latest, "area_amostral.gpkg"),
        os.path.join(latest, "area_poligono.gpkg"),
    ]
    poly_path = next((p for p in poly_candidates if os.path.exists(p)), None)
    pts_path  = os.path.join(latest, "pontos_produtividade.gpkg")

    if not (poly_path and os.path.exists(pts_path)):
        return None, None, latest

    gdf_poly = gpd.read_file(poly_path)
    gdf_pts  = gpd.read_file(pts_path)

    if gdf_poly.crs is None or gdf_poly.crs.to_epsg() != 4326:
        gdf_poly = gdf_poly.set_crs(4326) if gdf_poly.crs is None else gdf_poly.to_crs(4326)
    if gdf_pts.crs is None or gdf_pts.crs.to_epsg() != 4326:
        gdf_pts = gdf_pts.set_crs(4326) if gdf_pts.crs is None else gdf_pts.to_crs(4326)
   
    if "Code" not in gdf_pts.columns:
        gdf_pts = gdf_pts.copy()
        gdf_pts["Code"] = np.arange(1, len(gdf_pts) + 1).astype(int)

    return gdf_poly, gdf_pts, latest

def ensure_ee_init():
    """Inicializa EE a partir da variável de ambiente GEE_SA_KEY_JSON (Cloud Run)."""
    try:
        _ = ee.Number(1).getInfo()
        return
    except Exception:
        pass

    key_json = os.environ.get("GEE_SA_KEY_JSON", "")
    if not key_json:
        st.error("❌ Variável de ambiente **GEE_SA_KEY_JSON** não encontrada. Configure-a no Cloud Run com o conteúdo JSON da chave do service account.")
        st.stop()
    try:
        creds_dict = json.loads(key_json)
        credentials = ee.ServiceAccountCredentials(email=creds_dict["client_email"], key_data=key_json)
        ee.Initialize(credentials)
    except Exception as e:
        st.error(f"Erro ao inicializar o Google Earth Engine: {e}")
        st.stop()

ensure_ee_init()

with st.sidebar:
    st.subheader("Configurações")
    start = st.date_input("Início", value=date(2024,1,1))
    end   = st.date_input("Fim", value=date.today())
    cloud_thr = st.slider("Nuvem máxima (%)", 0, 80, 15, 1)
    buffer_m  = st.slider("Raio do buffer (m)", 1, 30, 5, 1)   
    indices_sel = st.multiselect(
        "Índices espectrais",
        ["NDVI", "GNDVI", "NDRE", "CCCI", "MSAVI2", "NDWI", "NDMI", "NBR", "TWI2"],
        default=["NDVI", "GNDVI", "NDRE", "CCCI", "MSAVI2", "NDWI", "NDMI", "NBR", "TWI2"] 
    )
    btn = st.button("▶️ Executar processamento")

gdf_poly, gdf_pts, latest_dir = load_cloud_inputs()
if gdf_poly is None or gdf_pts is None:
    st.warning("⚠️ Não encontrei os arquivos do passo 1 em `/tmp/streamlit_dados`. Volte em **Adicionar informações** e salve novamente.")
    st.stop()

st.write(f"**Polígono**: {len(gdf_poly)} geom | **Pontos**: {len(gdf_pts)} | Ex.:")
sample_cols = [c for c in gdf_pts.columns if c != "geometry"]
st.dataframe(gdf_pts[sample_cols].head(), use_container_width=True)

def mask_s2_sr(img):
    """Máscara por QA60 (nuvem/cirrus)."""
    qa = img.select('QA60')
    cloudBitMask  = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return img.updateMask(mask).copyProperties(img, img.propertyNames())

def add_indices(img, wanted):
    """Adiciona bandas dos índices selecionados."""
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

def best_image_on_date(geom, date_str, cloud_limit, wanted_idx):
    """Retorna a melhor cena do dia (menor nuvem), mascarada e com índices."""
    d0 = ee.Date(date_str)
    d1 = d0.advance(1, "day")
    coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geom)
            .filterDate(d0, d1)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_limit))
            .map(mask_s2_sr)
            .sort('CLOUDY_PIXEL_PERCENTAGE', True))
    img = ee.Image(coll.first())
    img = ee.Image(ee.Algorithms.If(img, img, ee.Image(coll.median())))
    img = add_indices(ee.Image(img), wanted_idx)
    return ee.Image(img)

def list_dates(geom, start_d, end_d, cloud_limit):
    """Lista datas (YYYY-MM-DD) com cenas disponíveis após filtros básicos."""
    base = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geom)
            .filterDate(ee.Date(str(start_d)), ee.Date(str(end_d)))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_limit)))
    dates = (base.aggregate_array("system:time_start")
             .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
             .distinct()
             .getInfo())
    return sorted(list(set(dates)))

def process_all():
    ee_poly = geemap.gdf_to_ee(gdf_poly[["geometry"]])
    ee_pts  = geemap.gdf_to_ee(gdf_pts[["Code", "geometry"]])

    dates = list_dates(ee_poly.geometry(), start, end, cloud_thr)
    if not dates:
        st.warning("Nenhuma data encontrada no período/limite de nuvens.")
        st.stop()
   
    def _buf(f):
        return ee.Feature(f.geometry().buffer(buffer_m)).copyProperties(f)
    ee_pts_buf = ee_pts.map(_buf)
    
    reducer = (ee.Reducer.min()
               .combine(ee.Reducer.mean(), sharedInputs=True)
               .combine(ee.Reducer.max(),  sharedInputs=True))
    
    base_out = gdf_pts[["Code"]].copy()
    base_out.index = base_out["Code"].astype(str)
    props_dict = {str(k): {} for k in base_out.index}

    prog = st.progress(0.0, text="Iniciando...")

    for i, d in enumerate(dates, start=1):
        prog.progress(i/len(dates), text=f"Processando {d} ({i}/{len(dates)})")
        img = best_image_on_date(ee_poly.geometry(), d, cloud_thr, indices_sel)
        img_idx = img.select(indices_sel) 
        
        feats_fc = img_idx.reduceRegions(
            collection=ee_pts_buf,
            reducer=reducer,
            scale=10,
            tileScale=2
        )

        feats = ee.FeatureCollection(feats_fc).getInfo().get("features", [])
        
        for f in feats:
            p = f.get("properties", {})
            code = str(p.get("Code"))
            if code not in props_dict:
                continue
            for idx in indices_sel:
                vmin  = p.get(f"{idx}_min",  None)
                vmean = p.get(f"{idx}_mean", None)
                vmax  = p.get(f"{idx}_max",  None)                
                props_dict[code][f"{d}_{idx}_min"]  = float(vmin)  if vmin  is not None else np.nan
                props_dict[code][f"{d}_{idx}_mean"] = float(vmean) if vmean is not None else np.nan
                props_dict[code][f"{d}_{idx}_max"]  = float(vmax)  if vmax  is not None else np.nan

    prog.empty()
    
    gdf_out = gdf_pts.copy()
    gdf_out.index = gdf_out["Code"].astype(str)
    for code, kv in props_dict.items():
        for col, val in kv.items():
            gdf_out.loc[code, col] = val
    gdf_out.reset_index(drop=True, inplace=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_gpkg = os.path.join(latest_dir, f"indices_espectrais_pontos_{ts}.gpkg")
    out_csv  = os.path.join(latest_dir, f"indices_espectrais_pontos_{ts}.csv")
    gdf_out.to_file(out_gpkg, driver="GPKG")
    gdf_out.drop(columns=["geometry"]).to_csv(out_csv, index=False)
    
    st.session_state["proc_result_gdf"] = gdf_out
    st.session_state["proc_paths"] = {"gpkg": out_gpkg, "csv": out_csv, "dates": dates}

if btn:
    with st.spinner("Processando (pode levar alguns minutos)…"):
        try:
            process_all()
            st.success("✅ Processamento concluído! Resultados abaixo.")
        except Exception as e:
            st.error(f"Erro no processamento: {e}")

gdf_out = st.session_state.get("proc_result_gdf")
paths   = st.session_state.get("proc_paths")

if gdf_out is not None:
    st.subheader("📄 Amostra dos resultados (colunas parciais)")
    preview = gdf_out.drop(columns=["geometry"]).iloc[:, : min(20, gdf_out.shape[1]-1)]
    st.dataframe(preview, use_container_width=True)

    csv_bytes = gdf_out.drop(columns=["geometry"]).to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar CSV completo", csv_bytes, file_name="indices_espectrais_pontos.csv", mime="text/csv")

    with st.expander("ℹ️ Metadados do processamento"):
        st.write({
            "datas_processadas": paths.get("dates", []),
            "periodo": [str(start), str(end)],
            "n_pontos": len(gdf_pts),
            "indices": list(sorted(set([c.split('_')[1] for c in gdf_out.columns if '_' in c and c.split('_')[-1] in ('min','mean','max')]))),
            "buffer_m": buffer_m,
            "cloud_threshold_%": cloud_thr
        })
else:
    st.info("Defina o período/índices e clique em **Executar processamento**.")
