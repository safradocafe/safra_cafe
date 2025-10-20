import os, glob, json, io, csv, unicodedata
from datetime import datetime

import ee
import geemap
import folium
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.prepared import prep
from scipy.interpolate import Rbf

try:
    from pyproj import Transformer
    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False

st.set_page_config(layout="wide")
st.markdown("## ☕ Saiba a sua próxima safra")
st.caption(
    "Recalcula índices no GEE para **treinamento** (safra passada) e **predição** (safra futura) "
    "usando o polígono e os pontos salvos na nuvem. Em seguida aplica o **melhor modelo salvo**."
)

BASE_TMP = "/tmp/streamlit_dados"
TOKENS_IDX = ['CCCI','NDMI','NDVI','GNDVI','NDWI','NBR','TWI2','NDRE','MSAVI2']

def _find_latest_save_dir(base=BASE_TMP):
    if not os.path.isdir(base): return None
    cands = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not cands: return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def _find_points_gpkg(save_dir):
    for nm in ["pontos_produtividade.gpkg","pontos_com_previsao.gpkg","prod_requinte_colab.gpkg"]:
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
    for nm in ["area_amostral.gpkg","area_poligono.gpkg","area_total_poligono.gpkg","requinte_colab.gpkg"]:
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

def _find_best_model(base=BASE_TMP):
    pats = glob.glob(os.path.join(base, "melhor_modelo_*.pkl"))
    if not pats: pats = glob.glob(os.path.join(base, "*.pkl"))
    if not pats: return None
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pats[0]

def ensure_ee_init():
    try:
        _ = ee.Number(1).getInfo()
        return
    except Exception:
        pass
    if "GEE_CREDENTIALS" in st.secrets:
        try:
            creds = dict(st.secrets["GEE_CREDENTIALS"])
            credentials = ee.ServiceAccountCredentials(
                email=creds["client_email"],
                key_data=json.dumps(creds)
            )
            ee.Initialize(credentials); return
        except Exception:
            pass
    key_json = os.environ.get("GEE_SA_KEY_JSON", "")
    if key_json:
        creds = json.loads(key_json)
        credentials = ee.ServiceAccountCredentials(
            email=creds["client_email"],
            key_data=key_json
        )
        ee.Initialize(credentials); return
    st.error("❌ Credenciais do Google Earth Engine não encontradas.")
    st.stop()

ensure_ee_init()

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

save_dir = _find_latest_save_dir()
if not save_dir:
    st.error("❌ Não encontrei diretório de salvamento em /tmp/streamlit_dados."); st.stop()

pts_gpkg = _find_points_gpkg(save_dir)
area_gpkg = _find_area_gpkg(save_dir)
model_path = _find_best_model(BASE_TMP)
params_path = os.path.join(save_dir, "parametros_area.json")

if not pts_gpkg: st.error("❌ GPKG de pontos não encontrado."); st.stop()
if not area_gpkg: st.error("❌ GPKG de área não encontrado."); st.stop()
if not model_path: st.error("❌ Modelo salvo não encontrado (melhor_modelo_*.pkl)."); st.stop()

params = {}
if os.path.exists(params_path):
    try:
        with open(params_path, "r") as f: params = json.load(f)
    except Exception:
        pass

gdf_area = gpd.read_file(area_gpkg)
gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)

gdf_pts = gpd.read_file(pts_gpkg)
gdf_pts = gdf_pts.set_crs(4326) if gdf_pts.crs is None else gdf_pts.to_crs(4326)

roi = geemap.gdf_to_ee(gdf_area[["geometry"]])

st.sidebar.header("Parâmetros")
bandas = TOKENS_IDX[:]

c1, c2 = st.sidebar.columns(2)
train_start = c1.date_input("Treino: início", value=pd.to_datetime(params.get("data_inicio","2023-08-01")).date())
train_end   = c2.date_input("Treino: fim",    value=pd.to_datetime(params.get("data_fim","2024-05-31")).date())

p1, p2 = st.sidebar.columns(2)
pred_start = p1.date_input("Predição: início", value=pd.to_datetime(params.get("pred_inicio","2024-08-01")).date())
pred_end   = p2.date_input("Predição: fim",    value=pd.to_datetime(params.get("pred_fim","2025-05-31")).date())

cloud_train = int(params.get("cloud_thr", 5))
buffer_m    = int(params.get("buffer_m", 5))
cloud_pred  = st.sidebar.slider("Nuvens para PREDIÇÃO (%)", 0, 60, 20, 1)
st.sidebar.caption(f"Treinamento usa os parâmetros do processamento: nuvens **{cloud_train}%**, buffer **{buffer_m} m**.")

def processar_colecao(start_date, end_date, roi, limite_nuvens):
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(roi)
           .filterDate(ee.Date(str(start_date)), ee.Date(str(end_date)))
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', limite_nuvens))
           .map(lambda img: img.addBands([
               img.normalizedDifference(['B8','B5']).divide(img.normalizedDifference(['B8','B4'])).rename('CCCI'),
               img.normalizedDifference(['B8','B11']).rename('NDMI'),
               img.normalizedDifference(['B3','B8']).rename('NDWI'),
               img.normalizedDifference(['B8','B12']).rename('NBR'),
               img.normalizedDifference(['B9','B8']).rename('TWI2'),
               img.normalizedDifference(['B8','B4']).rename('NDVI'),
               img.normalizedDifference(['B8','B5']).rename('NDRE'),
               img.expression('(2*NIR + 1 - sqrt((2*NIR + 1)**2 - 8*(NIR - RED)))/2',
                              {'NIR': img.select('B8'), 'RED': img.select('B4')}).rename('MSAVI2'),
               img.normalizedDifference(['B8','B3']).rename('GNDVI')
           ]))
           .map(lambda img: img.set('data', img.date().format('YYYY-MM-dd'))))

    datas_unicas = col.aggregate_array('data').distinct()
    def _unica_por_data(d):
        d = ee.String(d)
        imgs = col.filter(ee.Filter.eq('data', d))
        return ee.Image(imgs.first())
    return ee.ImageCollection(datas_unicas.map(_unica_por_data))

def extrair_estatisticas_ponto_imagem(imagem, feature_ponto, nomes_indices, buffer_m):
    buffer = feature_ponto.geometry().buffer(buffer_m)
    out = ee.Dictionary({})
    for indice in nomes_indices:
        banda = imagem.select(indice)
        red = banda.reduceRegion(
            reducer=ee.Reducer.min().combine(
                reducer2=ee.Reducer.mean(), sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.max(), sharedInputs=True
            ),
            geometry=buffer, scale=10, maxPixels=1e8
        )
        out = out.set(f"{indice}_min",  red.get(indice + "_min"))
        out = out.set(f"{indice}_mean", red.get(indice + "_mean"))
        out = out.set(f"{indice}_max",  red.get(indice + "_max"))
    return ee.Feature(feature_ponto.geometry(), out)

def processar_ponto(ponto, colecao_imagens, lista_indices, buffer_m):
    def por_imagem(img):
        return extrair_estatisticas_ponto_imagem(img, ponto, lista_indices, buffer_m)
    fc = colecao_imagens.map(por_imagem)
    def _combinar(feat, acc):
        return ee.Dictionary(acc).combine(ee.Feature(feat).toDictionary(), overwrite=True)
    props = ee.Dictionary(fc.iterate(_combinar, ee.Dictionary({})))
    return ee.Feature(ponto.geometry(), props)

def extrair_dados_min_mean_max(colecao, gdf_pontos, nomes_indices, buffer_m):
    pts_ee = geemap.gdf_to_ee(gdf_pontos)
    pts_proc = pts_ee.map(lambda pt: processar_ponto(pt, colecao, nomes_indices, buffer_m))
    gdf_out = geemap.ee_to_gdf(pts_proc)
    for col in gdf_pontos.columns:
        if col not in gdf_out.columns and col != "geometry":
            gdf_out[col] = gdf_pontos[col].values
    return gdf_out

def _load_model_bundle(path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj.get("model"), obj.get("features"), obj.get("scaler")
    return obj, None, None

def _norm(s: str) -> str:
    import unicodedata
    s = s.strip().lower().replace(" ", "_")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
    return s.replace("__", "_")

def _smart_align(Xdf: pd.DataFrame, expected: list[str]):
    cols_map = {_norm(c): c for c in Xdf.columns}
    used, missing, mapping = [], [], {}
    for f in expected:
        nf = _norm(f)
        if nf in cols_map:
            used.append(cols_map[nf]); mapping[f] = cols_map[nf]
        else:
            for stat in ["_mean","_min","_max"]:
                alt = nf + stat
                if alt in cols_map:
                    used.append(cols_map[alt]); mapping[f] = cols_map[alt]; break
            else:
                missing.append(f)
    if used:
        return Xdf[used].copy(), used, missing, mapping
    idx_cols = [c for c in Xdf.columns if any(tok.lower() in _norm(c) for tok in TOKENS_IDX)]
    fallback = sorted([c for c in idx_cols if any(s in c for s in ["_min","_mean","_max"])])
    if not fallback:
        return pd.DataFrame(), [], expected, {}
    return Xdf[fallback].copy(), fallback, expected, {}

def _expected_features_from(model, feats_bundle: list[str] | None, feats_all: list[str]) -> list[str]:
    if feats_bundle: return list(feats_bundle)
    if hasattr(model, "feature_names_in_"):
        try: return list(model.feature_names_in_)
        except Exception: pass
    return feats_all

def _maybe_scale_fit_transform(scaler, X_train, X_pred):
    if scaler is None:
        return X_train.values, X_pred.values
    try:
        Xt = scaler.transform(X_train.values)
        Xp = scaler.transform(X_pred.values)
        return Xt, Xp
    except Exception:
        scaler.fit(X_train.values)
        return scaler.transform(X_train.values), scaler.transform(X_pred.values)

def create_interactive_map(gdf_points, gdf_area, prod_column="produtividade_kg"):
    """Cria mapa interativo com pontos coloridos por produtividade"""
    
    # Centro do mapa
    centroid = gdf_area.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=13)
    
    # Adicionar polígono da área
    folium.GeoJson(
        gdf_area.__geo_interface__,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 3,
            'fillOpacity': 0
        },
        name="Área de Estudo"
    ).add_to(m)
    
    # Normalizar valores para cores
    values = gdf_points[prod_column].astype(float)
    vmin, vmax = values.min(), values.max()
    
    # Adicionar pontos coloridos
    for idx, row in gdf_points.iterrows():
        if pd.notna(row[prod_column]) and row.geometry is not None:
            # Cor baseada no valor (verde claro a escuro)
            normalized_val = (float(row[prod_column]) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = f"#{int(34 + normalized_val * 170):02x}{int(139 + normalized_val * 116):02x}{int(34 + normalized_val * 34):02x}"
            
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=8,
                popup=folium.Popup(f"""
                    <b>Produtividade:</b> {row[prod_column]:.2f} kg<br/>
                    <b>SC/HA:</b> {row.get('produtividade_sc/ha', 0):.2f}<br/>
                    <b>Lat:</b> {row.geometry.y:.6f}<br/>
                    <b>Lon:</b> {row.geometry.x:.6f}
                """, max_width=300),
                tooltip=f"Produtividade: {row[prod_column]:.2f} kg",
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    return m

if st.button("▶️ Processar dados"):
    with st.spinner("Processando…"):
        
        col_train = processar_colecao(train_start, train_end, roi, cloud_train)
        col_pred  = processar_colecao(pred_start,  pred_end,  roi, cloud_pred)

        gdf_train = extrair_dados_min_mean_max(col_train, gdf_pts, TOKENS_IDX, buffer_m)
        gdf_pred  = extrair_dados_min_mean_max(col_pred,  gdf_pts, TOKENS_IDX, buffer_m)
      
        feats_all = [f"{b}_{stat}" for b in TOKENS_IDX for stat in ["min","mean","max"]]
        for df_ in (gdf_train, gdf_pred):
            for c in feats_all:
                if c in df_.columns:
                    df_[c] = pd.to_numeric(df_[c], errors="coerce")

        if "maduro_kg" not in gdf_train.columns:
            st.error("❌ Coluna 'maduro_kg' não encontrada nos pontos para treinamento."); st.stop()
       
        modelo, feats_bundle, scaler = _load_model_bundle(model_path)
        if modelo is None or not hasattr(modelo, "fit"):
            st.error("❌ Arquivo de modelo não contém um estimador válido."); st.stop()
        
        feats_expected = _expected_features_from(modelo, feats_bundle, feats_all)
        X_train_df, used_feats, _, _ = _smart_align(gdf_train, feats_expected)
        X_pred_df,  _,          _, _ = _smart_align(gdf_pred,  used_feats)
        if X_train_df.empty or X_pred_df.empty:
            st.error("❌ Nenhuma feature disponível para o modelo após o alinhamento."); st.stop()
       
        X_train_mat, X_pred_mat = _maybe_scale_fit_transform(scaler, X_train_df, X_pred_df)
        y = pd.to_numeric(gdf_train["maduro_kg"], errors="coerce").values        
        modelo.fit(X_train_mat, y)
        yhat = modelo.predict(X_pred_mat)
       
        gdf_pred["produtividade_kg"] = yhat
        gdf_pred["produtividade_sc/ha"] = gdf_pred["produtividade_kg"] * (1/60) * (1/0.0016)
        if "latitude" not in gdf_pred.columns and "geometry" in gdf_pred.columns:
            gdf_pred["latitude"] = gdf_pred.geometry.y
        if "longitude" not in gdf_pred.columns and "geometry" in gdf_pred.columns:
            gdf_pred["longitude"] = gdf_pred.geometry.x

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_csv = os.path.join(save_dir, f"predicao_sem_datas_{ts}.csv")
        gdf_pred.drop(columns=["geometry"], errors="ignore").to_csv(out_csv, index=False)
       
        meta = {
            "treino_inicio": str(train_start),
            "treino_fim": str(train_end),
            "pred_inicio": str(pred_start),
            "pred_fim": str(pred_end),
            "nuvens_treino_%": cloud_train,
            "nuvens_pred_%": cloud_pred,
            "buffer_m": buffer_m,
            "modelo_usado": os.path.basename(model_path),
            "features_usadas": used_feats,
        }
        out_meta = os.path.join(save_dir, f"predicao_params_{ts}.json")
        with open(out_meta, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    st.success("✅ Predição concluída e arquivos salvos!")
    st.download_button("📥 Baixar CSV de predição", data=open(out_csv,"rb").read(),
                       file_name=os.path.basename(out_csv), mime="text/csv")
    
    try:
        df_pred_csv = _read_csv_robusto(out_csv)
        if "produtividade_sc/ha" in df_pred_csv.columns:
            media_sc_ha = pd.to_numeric(df_pred_csv["produtividade_sc/ha"], errors="coerce").mean()
            st.markdown("### 📌 Produtividade média prevista (safra)")
            st.metric("Média (sacas/ha)", f"{media_sc_ha:.2f}")
        else:
            st.warning("Coluna 'produtividade_sc/ha' não encontrada no CSV de predição.")
    except Exception as e:
        st.warning(f"Não foi possível calcular a média a partir do CSV: {e}")

    st.markdown("---")
    st.subheader("🗺️ Mapa Interativo de Variabilidade Espacial")
    
    try:
        # Criar mapa interativo
        interactive_map = create_interactive_map(gdf_pred, gdf_area)
        
        # Exibir mapa
        st_folium(interactive_map, width=1200, height=600)
        
        # Legenda explicativa
        st.info("""
        **🎨 Legenda do Mapa:**
        - **Área em preto**: Limite da área de estudo
        - **Pontos coloridos**: Produtividade prevista (kg)
          - Verde mais claro: Produtividade mais baixa
          - Verde mais escuro: Produtividade mais alta
        - **Clique nos pontos** para ver detalhes da produtividade
        """)
        
    except Exception as e:
        st.error(f"Erro ao criar mapa interativo: {e}")
        
        # Fallback para visualização simples
        st.warning("Exibindo visualização alternativa...")
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            gdf_area.boundary.plot(ax=ax, color='black', linewidth=2, label='Área de estudo')
            scatter = gdf_pred.plot(ax=ax, c=gdf_pred['produtividade_kg'], 
                                  cmap='YlGn', markersize=100, alpha=0.7,
                                  legend=True, legend_kwds={'label': 'Produtividade (kg)'})
            ax.set_title('Produtividade Prevista - Pontos Amostrais')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e2:
            st.error(f"Erro também na visualização alternativa: {e2}")
