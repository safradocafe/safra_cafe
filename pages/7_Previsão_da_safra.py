# pages/7_Previsão_da_safra.py
import os, glob, json, io, csv, unicodedata
from datetime import datetime

import ee
import geemap
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import joblib
import warnings
warnings.filterwarnings("ignore")

# =========================
# Página / estilo
# =========================
st.set_page_config(layout="wide")
st.markdown("## ☕ Predição por safra (reprocessar com GEE)")
st.caption(
    "Recalcula índices no GEE para **treinamento** (safra passada) e **predição** (safra futura) "
    "usando o polígono e os pontos salvos na nuvem. Em seguida aplica o **melhor modelo salvo**."
)

BASE_TMP = "/tmp/streamlit_dados"
TOKENS_IDX = ['CCCI','NDMI','NDVI','GNDVI','NDWI','NBR','TWI2','NDRE','MSAVI2']

# =========================
# Descoberta de arquivos
# =========================
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

# =========================
# GEE init
# =========================
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
        except Exception as e:
            st.warning(f"Falha via secrets: {e}")
    key_json = os.environ.get("GEE_SA_KEY_JSON", "")
    if key_json:
        try:
            creds = json.loads(key_json)
            credentials = ee.ServiceAccountCredentials(
                email=creds["client_email"],
                key_data=key_json
            )
            ee.Initialize(credentials); return
        except Exception as e:
            st.warning(f"Falha via env: {e}")
    st.error("❌ Credenciais do Google Earth Engine não encontradas.")
    st.stop()

ensure_ee_init()

# =========================
# CSV robusto (se/nquando precisar)
# =========================
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

# =========================
# Carregar insumos
# =========================
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

st.caption(f"📂 Origem: `{save_dir}`")
st.caption(f"📍 Pontos: `{os.path.basename(pts_gpkg)}` | 🗺️ Área: `{os.path.basename(area_gpkg)}` | 🧠 Modelo: `{os.path.basename(model_path)}`")

# =========================
# Área e pontos
# =========================
gdf_area = gpd.read_file(area_gpkg)
gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)

gdf_pts = gpd.read_file(pts_gpkg)
gdf_pts = gdf_pts.set_crs(4326) if gdf_pts.crs is None else gdf_pts.to_crs(4326)

roi = geemap.gdf_to_ee(gdf_area[["geometry"]])

# =========================
# Sidebar parâmetros
# =========================
st.sidebar.header("Parâmetros")
bandas = TOKENS_IDX[:]  # mantém a mesma lista

c1, c2 = st.sidebar.columns(2)
train_start = c1.date_input("Treino: início", value=pd.to_datetime(params.get("data_inicio","2023-08-01")).date())
train_end   = c2.date_input("Treino: fim",    value=pd.to_datetime(params.get("data_fim","2024-05-31")).date())

p1, p2 = st.sidebar.columns(2)
pred_start = p1.date_input("Predição: início", value=pd.to_datetime(params.get("pred_inicio","2024-08-01")).date())
pred_end   = p2.date_input("Predição: fim",    value=pd.to_datetime(params.get("pred_fim","2025-05-31")).date())

cloud_train = int(params.get("cloud_thr", 5))     # do processamento
buffer_m    = int(params.get("buffer_m", 5))      # do processamento
cloud_pred  = st.sidebar.slider("Nuvens para PREDIÇÃO (%)", 0, 60, 20, 1)
st.sidebar.caption(f"Treinamento usa os parâmetros do processamento: nuvens **{cloud_train}%**, buffer **{buffer_m} m**.")

# =========================
# Funções GEE
# =========================
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

# =========================
# Funções utilitárias de modelo (NOVO alinhamento)
# =========================
def _load_model_bundle(path):
    """Suporta .pkl como estimador OU bundle {'model', 'features', 'scaler'}."""
    obj = joblib.load(path)
    if isinstance(obj, dict):
        model   = obj.get("model")
        feats   = obj.get("features")
        scaler  = obj.get("scaler")
        return model, feats, scaler
    else:
        return obj, None, None

def _norm(s: str) -> str:
    s = s.strip().lower().replace(" ", "_")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")  # remove acento
    s = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
    s = s.replace("__", "_")
    return s

def _smart_align(Xdf: pd.DataFrame, expected: list[str]) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """
    Casa expected features com colunas de Xdf, de forma case-insensitive e sem acento.
    Retorna: Xdf_alinhado, used_feats (na ordem esperada), missing_feats, mapping_dict
    """
    cols_map = {_norm(c): c for c in Xdf.columns}  # normalizado -> original
    used = []
    missing = []
    mapping = {}
    for f in expected:
        nf = _norm(f)
        if nf in cols_map:
            used.append(cols_map[nf])
            mapping[f] = cols_map[nf]
        else:
            # fallback: 'ndvi' vira 'ndvi_mean' se existir só essa
            for stat in ["_mean","_min","_max"]:
                alt = nf + stat
                if alt in cols_map:
                    used.append(cols_map[alt])
                    mapping[f] = cols_map[alt]
                    break
            else:
                missing.append(f)
    if used:
        return Xdf[used].copy(), used, missing, mapping
    # fallback final: usa todas as colunas *_min/_mean/_max disponíveis
    idx_cols = [c for c in Xdf.columns if any(tok.lower() in _norm(c) for tok in TOKENS_IDX)]
    fallback = sorted([c for c in idx_cols if any(s in c for s in ["_min","_mean","_max"])])
    if not fallback:
        return pd.DataFrame(), [], expected, {}
    return Xdf[fallback].copy(), fallback, expected, {f:"<fallback>" for f in expected}

def _expected_features_from(model, feats_bundle: list[str] | None, feats_all: list[str]) -> list[str]:
    if feats_bundle and len(feats_bundle) > 0:
        return list(feats_bundle)
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass
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
        Xt = scaler.transform(X_train.values)
        Xp = scaler.transform(X_pred.values)
        return Xt, Xp

# =========================
# Executar
# =========================
if st.button("▶️ Reprocessar índices no GEE e prever"):
    with st.spinner("Processando…"):
        # 1) Coleções
        col_train = processar_colecao(train_start, train_end, roi, cloud_train)
        col_pred  = processar_colecao(pred_start,  pred_end,  roi, cloud_pred)

        # 2) Extrair estatísticas por ponto
        gdf_train = extrair_dados_min_mean_max(col_train, gdf_pts, TOKENS_IDX, buffer_m)
        gdf_pred  = extrair_dados_min_mean_max(col_pred,  gdf_pts, TOKENS_IDX, buffer_m)

        # 3) Preparar features (min/mean/max)
        feats_all = [f"{b}_{stat}" for b in TOKENS_IDX for stat in ["min","mean","max"]]
        for df_ in (gdf_train, gdf_pred):
            for c in feats_all:
                if c in df_.columns:
                    df_[c] = pd.to_numeric(df_[c], errors="coerce")

        if "maduro_kg" not in gdf_train.columns:
            st.error("❌ Coluna 'maduro_kg' não encontrada nos pontos para treinamento."); st.stop()

        # 4) Carrega modelo salvo (bundle ou estimador)
        modelo, feats_bundle, scaler = _load_model_bundle(model_path)
        if modelo is None or not hasattr(modelo, "fit"):
            st.error("❌ Arquivo de modelo não contém um estimador válido em 'model'."); st.stop()

        # 5) Determina features esperadas e faz alinhamento inteligente
        feats_expected = _expected_features_from(modelo, feats_bundle, feats_all)
        X_train_df, used_feats, missing_train, mapping = _smart_align(gdf_train, feats_expected)
        X_pred_df,  _,          missing_pred, _       = _smart_align(gdf_pred,  used_feats)

        if X_train_df.empty or X_pred_df.empty:
            st.error("❌ Nenhuma feature disponível para o modelo após o alinhamento (mesmo com fallback).")
            st.stop()

        # Relatório de mapeamento (transparência)
        if mapping:
            st.info("Mapeamento de features aplicado (modelo → coluna do dataset):\n"
                    + "\n".join([f"- {k} → {v}" for k,v in mapping.items()]))
        if missing_train:
            st.warning("Colunas esperadas ausentes no treino: " + ", ".join(missing_train))
        if missing_pred:
            st.warning("Colunas esperadas ausentes na predição: " + ", ".join(missing_pred))

        # 6) Escalonar (se houver scaler)
        X_train_mat, X_pred_mat = _maybe_scale_fit_transform(scaler, X_train_df, X_pred_df)

        # 7) y de treino
        y = pd.to_numeric(gdf_train["maduro_kg"], errors="coerce").values

        # 8) Ajuste e predição
        modelo.fit(X_train_mat, y)
        yhat = modelo.predict(X_pred_mat)

        # 9) Salvar resultados — conversão igual à usada anteriormente
        gdf_pred["produtividade_kg"] = yhat
        gdf_pred["produtividade_sc/ha"] = gdf_pred["produtividade_kg"] * (1/60) * (1/0.0016)

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
            "missing_train_cols": missing_train,
            "missing_pred_cols": missing_pred,
        }
        out_meta = os.path.join(save_dir, f"predicao_params_{ts}.json")
        with open(out_meta, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    st.success("✅ Predição concluída e arquivos salvos!")
    st.caption(f"CSV: `{out_csv}`")
    st.caption(f"JSON (parâmetros): `{out_meta}`")
    st.download_button("📥 Baixar CSV de predição", data=open(out_csv,"rb").read(),
                       file_name=os.path.basename(out_csv), mime="text/csv")
    st.download_button("📥 Baixar JSON de parâmetros", data=open(out_meta,"rb").read(),
                       file_name=os.path.basename(out_meta), mime="application/json")
