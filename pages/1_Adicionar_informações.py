import json
import re
import time
import random
import string
import os
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape
from shapely.prepared import prep

import streamlit as st
import folium
from streamlit_folium import st_folium
import speech_recognition as sr


# =======================
# CSS compacto e z-index dos controles do Leaflet
# =======================
st.markdown("""
<style>
.block-container { padding-top: 0.2rem !important; padding-bottom: 0.2rem !important; }
header, footer {visibility: hidden;}
div[data-testid="stVerticalBlock"] { gap: 0.2rem !important; }
.leaflet-control { z-index: 1000 !important; }
.leaflet-top.leaflet-right { top: 12px !important; right: 12px !important; }
.leaflet-top.leaflet-left { top: 12px !important; left: 12px !important; }
.leaflet-control-layers-expanded { max-height: 260px; overflow:auto; }

/* Upload compacto */
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed #999 !important;
  background: #fafafa !important;
  padding: 6px 8px !important;
  min-height: 60px !important;
  margin: 0.2rem 0 !important;
}
div[data-testid="stFileUploaderDropzone"] small,
div[data-testid="stFileUploaderDropzone"] span { display: none !important; }
div[data-testid="stFileUploaderDropzone"]::after {
  content: "Arraste e solte o arquivo aqui ou clique para selecionar";
  display: block; color: #444; font-size: 11px; text-align: center; padding-top: 4px;
}
.controls-title { font-size: 12px !important; font-weight: 700; margin: 2px 0 2px 0 !important; }
.sub-mini { font-size: 11px !important; font-weight: 600; margin: 2px 0 1px 0 !important; }
</style>
""", unsafe_allow_html=True)


# =======================
# Estado
# =======================
DEFAULT_KEYS = [
    'gdf_poligono', 'gdf_pontos', 'unidade_selecionada',
    'densidade_plantas', 'produtividade_media', 'map_fit_bounds',
    # novos flags
    'add_mode', 'voice_mode', 'voice_value', 'last_click_token'
]
for k in DEFAULT_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

st.session_state.unidade_selecionada = st.session_state.unidade_selecionada or 'kg'
st.session_state.add_mode = bool(st.session_state.add_mode)
st.session_state.voice_mode = bool(st.session_state.voice_mode)
st.session_state.voice_value = st.session_state.voice_value if st.session_state.voice_value is not None else 0.0


# =======================
# Utilitários
# =======================
def gerar_codigo():
    letras = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    numeros = ''.join(random.choices(string.digits, k=2))
    return f"{letras}-{numeros}-{''.join(random.choices(string.ascii_uppercase + string.digits, k=4))}"


def converter_para_kg(valor, unidade):
    if pd.isna(valor):
        return 0.0
    try:
        v = float(valor)
    except Exception:
        return 0.0
    if unidade == 'kg':
        return v
    if unidade == 'latas':
        return v * 1.8
    if unidade == 'litros':
        return v * 0.09
    return v


def _fit_bounds_from_gdf(gdf):
    b = gdf.total_bounds  
    return [[b[1], b[0]], [b[3], b[2]]]


def _point_inside_area(lat, lon) -> bool:
    if st.session_state.gdf_poligono is None:
        return True  
    poly = st.session_state.gdf_poligono.geometry.unary_union
    return prep(poly).contains(Point(lon, lat))

# Mapa
def create_map():
    if st.session_state.gdf_poligono is not None:
        m = folium.Map(location=[0, 0], zoom_start=2, tiles=None, control_scale=True)
        bounds = _fit_bounds_from_gdf(st.session_state.gdf_poligono)
        st.session_state.map_fit_bounds = bounds
        m.fit_bounds(bounds, padding=(20, 20))
    else:
        m = folium.Map(location=[-15, -55], zoom_start=4, tiles=None, control_scale=True)

    # bases
    folium.TileLayer('OpenStreetMap', name='Mapa (ruas)', control=True, show=True).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satélite', control=True, show=False
    ).add_to(m)

    # desenho da área (polígono; sem marker)
    folium.plugins.Draw(
        draw_options={
            'polyline': False, 'rectangle': True, 'circle': False,
            'circlemarker': False, 'marker': False,
            'polygon': {'allowIntersection': False, 'showArea': True, 'repeatMode': False}
        },
        export=False, position='topleft'
    ).add_to(m)

    # área existente
    if st.session_state.gdf_poligono is not None:
        folium.GeoJson(
            st.session_state.gdf_poligono,
            name="Área amostral",
            style_function=lambda x: {"color": "blue", "fillColor": "blue", "fillOpacity": 0.2, "weight": 2}
        ).add_to(m)

    # pontos existentes
    if st.session_state.gdf_pontos is not None and not st.session_state.gdf_pontos.empty:
        for _, row in st.session_state.gdf_pontos.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5, color="green", fill=True, fill_color="green", fill_opacity=0.8,
                popup=f"Ponto: {row['Code']}<br>Produtividade (kg): {row['maduro_kg']:.2f}"
            ).add_to(m)

    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    return m

# Upload & IO
def processar_arquivo_carregado(uploaded_file, tipo='amostral'):
    try:
        if uploaded_file is None:
            return None
        if not uploaded_file.name.lower().endswith('.gpkg'):
            st.error("❌ O arquivo deve ter extensão .gpkg")
            return None

        temp_file = f"/tmp/{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        gdf = gpd.read_file(temp_file)
        os.remove(temp_file)

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        if tipo == 'amostral':
            if gdf.empty or not gdf.geom_type.isin(['Polygon', 'MultiPolygon']).any():
                st.error("❌ O arquivo da área amostral deve conter polígonos.")
                return None
            st.session_state.gdf_poligono = gdf[['geometry']]
            st.session_state.map_fit_bounds = _fit_bounds_from_gdf(st.session_state.gdf_poligono)
            st.success("✅ Área amostral carregada com sucesso!")
            return gdf

        elif tipo == 'pontos':
            if gdf.empty or not gdf.geom_type.isin(['Point', 'MultiPoint']).any():
                st.error("❌ O arquivo de pontos deve conter geometrias do tipo Ponto.")
                return None
            required_cols = ['Code', 'maduro_kg', 'latitude', 'longitude', 'geometry']
            faltando = [c for c in required_cols if c not in gdf.columns]
            if faltando:
                st.error("❌ Faltam as colunas: " + ", ".join(faltando))
                st.info("Necessário: Code, maduro_kg, latitude, longitude e geometry (pontos).")
                return None
            # coerções
            for c in ['maduro_kg', 'latitude', 'longitude']:
                gdf[c] = pd.to_numeric(gdf[c], errors='coerce')
            gdf['latitude'] = gdf.geometry.y
            gdf['longitude'] = gdf.geometry.x
            if not ((gdf['latitude'].between(-90, 90)) & (gdf['longitude'].between(-180, 180))).all():
                st.error("❌ Latitude/Longitude fora do intervalo esperado.")
                return None
            st.session_state.gdf_pontos = gdf
            st.success(f"✅ {len(gdf)} pontos carregados com sucesso!")
            return gdf

    except Exception as e:
        st.error("❌ Erro ao processar arquivo.")
        st.exception(e)
        return None


def salvar_no_streamlit_cloud():
    import os, json, time

    if st.session_state.get("gdf_poligono") is None:
        st.warning("⚠️ Defina a área amostral antes de salvar!")
        return
    if st.session_state.get("densidade_plantas") is None or \
       st.session_state.get("produtividade_media") is None:
        st.warning("⚠️ Parâmetros de densidade e produtividade não definidos!")
        return
 
    base_dir = "/tmp/streamlit_dados"
    os.makedirs(base_dir, exist_ok=True)

    carimbo = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(base_dir, f"salvamento-{carimbo}")
    os.makedirs(save_dir, exist_ok=True)

    area_path   = os.path.join(save_dir, "area_amostral.gpkg")
    pontos_path = os.path.join(save_dir, "pontos_produtividade.gpkg")
    params_path = os.path.join(save_dir, "parametros_area.json")

    st.session_state.gdf_poligono.to_file(area_path, driver="GPKG")
  
    if st.session_state.get("gdf_pontos") is not None and not st.session_state.gdf_pontos.empty:
        st.session_state.gdf_pontos.to_file(pontos_path, driver="GPKG")

    parametros = {
        "densidade_pes_ha": st.session_state.densidade_plantas,
        "produtividade_media_sacas_ha": st.session_state.produtividade_media,
    }
    with open(params_path, "w") as f:
        json.dump(parametros, f)
 
    st.session_state["tmp_save_dir"]    = save_dir
    st.session_state["tmp_area_path"]   = area_path
    st.session_state["tmp_params_path"] = params_path
    if os.path.exists(pontos_path):
        st.session_state["tmp_pontos_path"] = pontos_path

    st.success("✅ Dados salvos Vá para monitoramento.")

def exportar_dados():
    if st.session_state.gdf_poligono is None:
        st.warning("⚠️ Defina a área amostral antes de exportar!")
        return

    tmp_dir = "/tmp/export_zip"
    os.makedirs(tmp_dir, exist_ok=True)

    poligono_path = os.path.join(tmp_dir, "area_amostral.gpkg")
    pontos_path   = os.path.join(tmp_dir, "pontos_produtividade.gpkg")
    params_path   = os.path.join(tmp_dir, "parametros_area.json")
    zip_path      = os.path.join(tmp_dir, "dados_produtividade.zip")

    st.session_state.gdf_poligono.to_file(poligono_path, driver="GPKG")
    if st.session_state.gdf_pontos is not None and not st.session_state.gdf_pontos.empty:
        st.session_state.gdf_pontos.to_file(pontos_path, driver="GPKG")

    parametros = {
        'densidade_pes_ha': st.session_state.densidade_plantas,
        'produtividade_media_sacas_ha': st.session_state.produtividade_media
    }
    with open(params_path, "w") as f:
        json.dump(parametros, f)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(poligono_path, arcname="area_amostral.gpkg")
        if st.session_state.gdf_pontos is not None and not st.session_state.gdf_pontos.empty:
            zipf.write(pontos_path, arcname="pontos_produtividade.gpkg")
        zipf.write(params_path, arcname="parametros_area.json")

    with open(zip_path, "rb") as f:
        st.download_button(
            "💾 Exportar dados (ZIP)",
            data=f.read(),
            file_name="dados_produtividade.zip",
            mime="application/zip"
        )

# Inserção/edição de pontos
def _ensure_points_df():
    if st.session_state.gdf_pontos is None:
        st.session_state.gdf_pontos = gpd.GeoDataFrame(
            columns=['geometry', 'Code', 'valor', 'unidade', 'maduro_kg',
                     'coletado', 'latitude', 'longitude', 'metodo'],
            geometry='geometry', crs="EPSG:4326"
        )


def _add_point(lat, lon, metodo, valor=None):
    _ensure_points_df()
    if not _point_inside_area(lat, lon):
        st.warning("⚠️ Clique fora da área amostral. Ponto ignorado.")
        return
    valor = float(valor) if (valor is not None and str(valor).strip() != "") else 0.0
    novo = {
        'geometry': Point(lon, lat),
        'Code': gerar_codigo(),
        'valor': valor,
        'unidade': st.session_state.unidade_selecionada,
        'maduro_kg': converter_para_kg(valor, st.session_state.unidade_selecionada),
        'coletado': False,
        'latitude': lat,
        'longitude': lon,
        'metodo': metodo
    }
    st.session_state.gdf_pontos = gpd.GeoDataFrame(
        pd.concat([st.session_state.gdf_pontos, pd.DataFrame([novo])], ignore_index=True),
        geometry='geometry', crs="EPSG:4326"
    )


def inserir_produtividade():
    gdf = st.session_state.get("gdf_pontos")
    if gdf is None or gdf.empty:
        st.warning("Nenhum ponto disponível!")
        return
    unidade = st.session_state.unidade_selecionada or "kg"
    st.markdown(f"**Inserir/editar produtividade** — unidade atual: `{unidade}`")

    # inicializa valores no estado
    for idx, row in gdf.iterrows():
        key = f"valor_pt_{idx}"
        if key not in st.session_state:
            st.session_state[key] = float(row.get("valor", 0.0) or 0.0)

    with st.form("form_produtividade", clear_on_submit=False):
        cols = st.columns(3)
        for idx, _ in gdf.iterrows():
            col = cols[idx % 3]
            with col:
                st.number_input(
                    f"Ponto {idx+1}",
                    key=f"valor_pt_{idx}",
                    min_value=0.0,
                    step=0.01,
                    format="%.2f"
                )
        submitted = st.form_submit_button("Salvar produtividade", type="primary")

    if submitted:
        for idx in gdf.index:
            v = float(st.session_state.get(f"valor_pt_{idx}", 0.0))
            gdf.at[idx, "valor"] = v
            gdf.at[idx, "maduro_kg"] = converter_para_kg(v, unidade)
        st.session_state.gdf_pontos = gdf
        st.success("Produtividades salvas e convertidas para kg.")

# Layout
st.subheader("Adicionar informações: área amostral e pontos de produtividade")

mapa = create_map()
mapa_data = st_folium(mapa, use_container_width=True, height=520, key='mapa_principal')

# 1) captura polígono 
if mapa_data and mapa_data.get('last_active_drawing'):
    geometry = mapa_data['last_active_drawing']['geometry']
    gdf = gpd.GeoDataFrame(geometry=[shape(geometry)], crs="EPSG:4326")
    st.session_state.gdf_poligono = gdf
    st.session_state.map_fit_bounds = _fit_bounds_from_gdf(gdf)
    st.success("Área amostral definida!")
    time.sleep(0.2)
    st.rerun()

# 2) clique para criar ponto
if st.session_state.add_mode and mapa_data and mapa_data.get("last_clicked"):
    lat = mapa_data["last_clicked"]["lat"]
    lon = mapa_data["last_clicked"]["lng"]
    token = f"{lat:.6f},{lon:.6f}"
    if token != st.session_state.get("last_click_token", None):
        valor = st.session_state.voice_value if st.session_state.voice_mode else 0.0
        _add_point(
            lat, lon,
            metodo="clique" + ("+voz" if st.session_state.voice_mode else ""),
            valor=valor
        )
        st.session_state.last_click_token = token
        if st.session_state.voice_mode and st.session_state.voice_value:
            st.session_state.voice_value = 0.0
        st.success("Ponto adicionado!")
        time.sleep(0.15)
        st.rerun()

# --- 2) Uploads 
st.markdown('<div class="sub-mini">Uploads (opcional)</div>', unsafe_allow_html=True)
u1, u2 = st.columns(2)
with u1:
    uploaded_area = st.file_uploader("Área amostral (.gpkg)", type=['gpkg'], key='upload_area')
    if uploaded_area:
        processar_arquivo_carregado(uploaded_area, tipo='amostral')
with u2:
    uploaded_pontos = st.file_uploader("Pontos de produtividade (.gpkg)", type=['gpkg'], key='upload_pontos')
    if uploaded_pontos:
        processar_arquivo_carregado(uploaded_pontos, tipo='pontos')

# --- 3) Controles 
a1, a2, a3, a4 = st.columns([1, 1, 1, 1])
with a1:
    if st.button("🧭 Definir área amostral"):
        st.success("Use a barra de desenho (▭ polígono) no mapa para delimitar a área.")
with a2:
    st.session_state.add_mode = st.toggle(
        "➕ Adicionar pontos no clique",
        value=st.session_state.add_mode,
        help="Ative e clique no mapa para criar pontos."
    )
with a3:
    st.session_state.voice_mode = st.toggle("🎙️ Modo voz (falar produtividade)", value=st.session_state.voice_mode)
with a4:
    if st.button("🗑️ Limpar área e pontos"):
        st.session_state.gdf_poligono = None
        st.session_state.gdf_pontos = None
        st.session_state.map_fit_bounds = None
        st.session_state.voice_value = 0.0
        st.session_state.last_click_token = None
        st.session_state.add_mode = False
        st.session_state.voice_mode = False
        st.success("Área e pontos limpos!")

# Linha 2: unidade e voz + parâmetros
b1, b2, b3 = st.columns([1, 2, 2])
with b1:
    st.session_state.unidade_selecionada = st.selectbox(
        "Unidade", ['kg', 'latas', 'litros'],
        index=['kg', 'latas', 'litros'].index(st.session_state.unidade_selecionada)
    )
with b2:
    st.markdown("**Produtividade por voz (opcional)**")
    if st.session_state.voice_mode:
        audio = st.audio_input("Gravar número da produtividade (ex.: '12 vírgula 5')", key="voz_input")
        if audio is not None:
            wav_path = "/tmp/voz_produtividade.wav"
            with open(wav_path, "wb") as f:
                f.write(audio.getvalue())
            rec = sr.Recognizer()
            try:
                with sr.AudioFile(wav_path) as source:
                    data = rec.record(source)
                txt = rec.recognize_google(data, language="pt-BR")
                m = re.search(r"(\d+[.,]?\d*)", txt.replace("vírgula", ","))
                if m:
                    val = float(m.group(1).replace(",", "."))
                    st.session_state.voice_value = val
                    st.success(f"Valor reconhecido: {val:.2f} {st.session_state.unidade_selecionada}. Clique no mapa para criar o ponto.")
                else:
                    st.warning(f"Não consegui extrair número de: “{txt}”.")
            except Exception as e:
                st.warning(f"Não foi possível transcrever o áudio: {e}")
    else:
        st.caption("Ative o modo voz para gravar o número; no próximo clique no mapa o ponto é criado com esse valor.")

with b3:
    st.markdown("**Parâmetros da área**")
    st.session_state.densidade_plantas = st.number_input(
        "Densidade (plantas/ha)",
        value=float(st.session_state.densidade_plantas or 0)
    )
    st.session_state.produtividade_media = st.number_input(
        "Produtividade média última safra (sacas/ha)",
        value=float(st.session_state.produtividade_media or 0)
    )

# Painel inferior (inalterado)
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    if st.button("🔢 Gerar pontos automáticos (2/ha)"):
        if st.session_state.gdf_poligono is None:
            st.warning("Defina a área amostral primeiro.")
        else:
            gdf = st.session_state.gdf_poligono
            centroid = gdf.geometry.centroid.iloc[0]
            utm_zone = int((centroid.x + 180) / 6) + 1
            epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
            gdf_utm = gdf.to_crs(epsg=epsg)
            lado = np.sqrt(5000)  # ~2 pontos/ha
            b = gdf_utm.total_bounds
            xs = np.arange(b[0], b[2], lado)
            ys = np.arange(b[1], b[3], lado)
            pontos = [Point(x, y) for x in xs for y in ys if gdf_utm.geometry.iloc[0].contains(Point(x, y))]
            gdf_p = gpd.GeoDataFrame(geometry=pontos, crs=gdf_utm.crs).to_crs(4326)
            gdf_p['Code'] = [gerar_codigo() for _ in range(len(gdf_p))]
            gdf_p['valor'] = 0.0
            gdf_p['unidade'] = st.session_state.unidade_selecionada
            gdf_p['maduro_kg'] = 0.0
            gdf_p['coletado'] = False
            gdf_p['latitude'] = gdf_p.geometry.y
            gdf_p['longitude'] = gdf_p.geometry.x
            gdf_p['metodo'] = 'auto'
            st.session_state.gdf_pontos = gdf_p
            st.success(f"{len(gdf_p)} pontos gerados automaticamente!")
with c2:
    if st.button("📝 Inserir/editar produtividade"):
        inserir_produtividade()
with c3:
    if st.button("💾 Exportar dados"):
        exportar_dados()
with c4:
    if st.button("☁️ Salvar dados na nuvem"):
        salvar_no_streamlit_cloud()
