import json
import streamlit as st
import geemap
import time
import random
import string
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape
import os
import folium
from streamlit_folium import st_folium
import zipfile
from io import BytesIO

# ---- Estilo / CSS ----
st.markdown("""
    <style>
    .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; }
    header, footer {visibility: hidden;}

    /* Reduz espaçamentos verticais gerais */
    .stMarkdown, .stButton, .stNumberInput, .stSelectbox, .stFileUploader { margin: 0.2rem 0 !important; }

    /* Reduz o espaço logo abaixo do iframe do mapa */
    .streamlit-folium, .streamlit-folium iframe { margin-bottom: 0.2rem !important; }

    /* File uploader PT-BR e compacto */
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

    .sub-mini { font-size: 11px !important; font-weight: 600; margin: 4px 0 2px 0 !important; padding: 0 !important; }
    .controls-title { font-size: 12px !important; font-weight: 700; margin: 6px 0 4px 0 !important; padding: 0 !important; }
    .controls-group label { font-size: 11px !important; }
    .controls-group .stButton>button { padding: 2px 8px !important; font-size: 11px !important; margin: 1px 0 !important; }
    .stNumberInput input, .stSelectbox select { padding: 2px 8px !important; min-height: 30px !important; }
    </style>
""", unsafe_allow_html=True)

# ---- Estado inicial ----
for key in [
    'gdf_poligono', 'gdf_pontos', 'unidade_selecionada',
    'densidade_plantas', 'produtividade_media', 'mapa_data', 'modo_insercao',
    'drawing_mode', 'map_fit_bounds'
]:
    if key not in st.session_state:
        st.session_state[key] = 'kg' if key == 'unidade_selecionada' else None

# ---- Funções auxiliares ----
def gerar_codigo():
    letras = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    numeros = ''.join(random.choices(string.digits, k=2))
    return f"{letras}-{numeros}-{''.join(random.choices(string.ascii_uppercase + string.digits, k=4))}"

def converter_para_kg(valor, unidade):
    if pd.isna(valor):
        return 0
    try:
        valor = float(valor)
    except:
        return 0
    if unidade == 'kg':
        return valor
    elif unidade == 'latas':
        return valor * 1.8
    elif unidade == 'litros':
        return valor * 0.09
    return valor

def get_utm_epsg(lon, lat):
    utm_zone = int((lon + 180) / 6) + 1
    return 32600 + utm_zone if lat >= 0 else 32700 + utm_zone

def _fit_bounds_from_gdf(gdf):
    b = gdf.total_bounds  # [minx,miny,maxx,maxy]
    return [[b[1], b[0]], [b[3], b[2]]]

def create_map():
    """
    Mantém o enquadramento estável na área amostral e exibe as opções
    de base 'Mapa (ruas)' e 'Satélite' no controle de camadas.
    """
    # cria o mapa sem camada base padrão para podermos controlar as bases
    if st.session_state.gdf_poligono is not None:
        m = folium.Map(location=[0, 0], zoom_start=2, tiles=None, control_scale=True)
        bounds = _fit_bounds_from_gdf(st.session_state.gdf_poligono)
        st.session_state.map_fit_bounds = bounds
        m.fit_bounds(bounds, padding=(20, 20))
    else:
        m = folium.Map(location=[-15, -55], zoom_start=4, tiles=None, control_scale=True)

    # bases: ruas e satélite
    folium.TileLayer('OpenStreetMap', name='Mapa (ruas)', control=True).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satélite', control=True
    ).add_to(m)

    # desenho
    draw = folium.plugins.Draw(
        draw_options={
            'polyline': False, 'rectangle': True, 'circle': False,
            'circlemarker': False, 'marker': False,
            'polygon': {'allowIntersection': False, 'showArea': True, 'repeatMode': False}
        },
        export=False, position='topleft'
    )
    draw.add_to(m)

    # área amostral
    if st.session_state.gdf_poligono is not None:
        folium.GeoJson(
            st.session_state.gdf_poligono,
            name="Área amostral",
            style_function=lambda x: {"color": "blue", "fillColor": "blue", "fillOpacity": 0.3}
        ).add_to(m)
        try:
            m.fit_bounds(st.session_state.map_fit_bounds, padding=(20, 20))
        except Exception:
            pass

    # pontos
    if st.session_state.gdf_pontos is not None and not st.session_state.gdf_pontos.empty:
        for _, row in st.session_state.gdf_pontos.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5, color="green", fill=True, fill_color="green", fill_opacity=0.7,
                popup=f"Ponto: {row['Code']}<br>Produtividade: {row['maduro_kg']}"
            ).add_to(m)

    # controle de camadas visível
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    return m

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

        # Lê o GPKG (primeira camada por padrão)
        gdf = gpd.read_file(temp_file)
        os.remove(temp_file)

        # Normaliza CRS para WGS84 (EPSG:4326)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        if tipo == 'amostral':
            # Deve ser polígono
            if gdf.empty or not gdf.geom_type.isin(['Polygon', 'MultiPolygon']).any():
                st.error("❌ O arquivo da área amostral deve conter polígonos.")
                return None

            st.session_state.gdf_poligono = gdf[['geometry']]
            # Armazena bounds para manter enquadramento estável
            st.session_state.map_fit_bounds = _fit_bounds_from_gdf(st.session_state.gdf_poligono)
            st.success("✅ Área amostral carregada com sucesso!")
            return gdf

        elif tipo == 'pontos':
            # Deve ser ponto
            if gdf.empty or not gdf.geom_type.isin(['Point', 'MultiPoint']).any():
                st.error("❌ O arquivo de pontos deve conter geometrias do tipo Ponto.")
                return None

            required_cols = ['Code', 'maduro_kg', 'latitude', 'longitude', 'geometry']
            faltando = [c for c in required_cols if c not in gdf.columns]
            if faltando:
                st.error("❌ Arquivo inválido. Faltam as colunas obrigatórias: " + ", ".join(faltando))
                st.info("Necessário: Code, maduro_kg, latitude, longitude e geometry (pontos).")
                return None

            try:
                gdf['maduro_kg'] = pd.to_numeric(gdf['maduro_kg'])
                gdf['latitude']  = pd.to_numeric(gdf['latitude'])
                gdf['longitude'] = pd.to_numeric(gdf['longitude'])
            except Exception:
                st.error("❌ As colunas maduro_kg, latitude e longitude precisam ser numéricas.")
                return None

            # Recalcula latitude/longitude pela geometria
            gdf['latitude']  = gdf.geometry.y
            gdf['longitude'] = gdf.geometry.x

            # Checagem final
            if not ((gdf['latitude'].between(-90, 90)) & (gdf['longitude'].between(-180, 180))).all():
                st.error("❌ Latitude/Longitude fora do intervalo esperado (-90..90 / -180..180).")
                return None

            st.session_state.gdf_pontos = gdf
            st.success(f"✅ {len(gdf)} pontos carregados com sucesso!")
            return gdf

    except Exception as e:
        st.error("❌ Erro ao processar arquivo (detalhes abaixo).")
        st.exception(e)
        return None

def gerar_pontos_automaticos():
    if st.session_state.gdf_poligono is None:
        st.warning("Defina a área amostral")
        return
    centroid = st.session_state.gdf_poligono.geometry.centroid.iloc[0]
    epsg = get_utm_epsg(centroid.x, centroid.y)
    gdf_utm = st.session_state.gdf_poligono.to_crs(epsg=epsg)
    area_ha = gdf_utm.geometry.area.sum() / 10000
    lado = np.sqrt(5000)  # ~2 pontos/ha
    bounds = gdf_utm.total_bounds
    x_coords = np.arange(bounds[0], bounds[2], lado)
    y_coords = np.arange(bounds[1], bounds[3], lado)
    pontos = [Point(x, y) for x in x_coords for y in y_coords if gdf_utm.geometry.iloc[0].contains(Point(x, y))]
    gdf_pontos = gpd.GeoDataFrame(geometry=pontos, crs=gdf_utm.crs).to_crs("EPSG:4326")
    gdf_pontos['Code'] = [gerar_codigo() for _ in range(len(gdf_pontos))]
    gdf_pontos['valor'] = 0
    gdf_pontos['unidade'] = 'kg'
    gdf_pontos['maduro_kg'] = 0
    gdf_pontos['coletado'] = False
    gdf_pontos['latitude'] = gdf_pontos.geometry.y
    gdf_pontos['longitude'] = gdf_pontos.geometry.x
    gdf_pontos['metodo'] = 'auto'
    st.session_state.gdf_pontos = gdf_pontos
    st.success(f"{len(gdf_pontos)} pontos gerados automaticamente! Área: {area_ha:.2f} ha")

def salvar_pontos():
    if st.session_state.gdf_pontos is None or st.session_state.gdf_pontos.empty:
        st.warning("⚠️ Nenhum ponto para salvar!")
        return
    st.success("✅ Dados dos pontos preparados para exportação!")

def exportar_dados():
    if st.session_state.gdf_poligono is None:
        st.warning("⚠️ Defina a área amostral antes de exportar!")
        return

    tmp_dir = "/tmp/export_zip"
    os.makedirs(tmp_dir, exist_ok=True)

    poligono_path = os.path.join(tmp_dir, "area_amostral.gpkg")
    pontos_path = os.path.join(tmp_dir, "pontos_produtividade.gpkg")
    params_path = os.path.join(tmp_dir, "parametros_area.json")
    zip_path = os.path.join(tmp_dir, "dados_produtividade.zip")

    # Área amostral
    st.session_state.gdf_poligono.to_file(poligono_path, driver="GPKG")

    # Pontos (se houver)
    if st.session_state.gdf_pontos is not None and not st.session_state.gdf_pontos.empty:
        st.session_state.gdf_pontos.to_file(pontos_path, driver="GPKG")

    # Parâmetros
    parametros = {
        'densidade_pes_ha': st.session_state.densidade_plantas,
        'produtividade_media_sacas_ha': st.session_state.produtividade_media
    }
    with open(params_path, "w") as f:
        json.dump(parametros, f)

    # Monta o ZIP
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(poligono_path, arcname="area_amostral.gpkg")
        if st.session_state.gdf_pontos is not None and not st.session_state.gdf_pontos.empty:
            zipf.write(pontos_path, arcname="pontos_produtividade.gpkg")
        zipf.write(params_path, arcname="parametros_area.json")

    with open(zip_path, "rb") as f:
        st.download_button(
            label="💾 Exportar dados (ZIP)",
            data=f.read(),
            file_name="dados_produtividade.zip",
            mime="application/zip"
        )

def inserir_ponto_manual():
    with st.form("Inserir ponto (manual)"):
        lat = st.number_input("Latitude:", value=-15.0)
        lon = st.number_input("Longitude:", value=-55.0)
        if st.form_submit_button("Adicionar Ponto"):
            adicionar_ponto(lat, lon, "manual")
            st.rerun()

def adicionar_ponto(lat, lon, metodo):
    ponto = Point(lon, lat)
    if st.session_state.gdf_pontos is None:
        gdf_pontos = gpd.GeoDataFrame(
            columns=['geometry', 'Code', 'valor', 'unidade', 'maduro_kg', 'coletado', 'latitude', 'longitude', 'metodo'],
            geometry='geometry', crs="EPSG:4326"
        )
    else:
        gdf_pontos = st.session_state.gdf_pontos

    novo_ponto = {
        'geometry': ponto,
        'Code': gerar_codigo(),
        'valor': 0,
        'unidade': st.session_state.unidade_selecionada,
        'maduro_kg': 0,
        'coletado': False,
        'latitude': lat,
        'longitude': lon,
        'metodo': metodo
    }

    st.session_state.gdf_pontos = gpd.GeoDataFrame(
        pd.concat([gdf_pontos, pd.DataFrame([novo_ponto])]),
        crs="EPSG:4326"
    )
    st.success(f"Ponto {len(st.session_state.gdf_pontos)} adicionado ({metodo})")

def inserir_produtividade():
    gdf = st.session_state.get("gdf_pontos")
    if gdf is None or gdf.empty:
        st.warning("Nenhum ponto disponível!")
        return

    unidade = st.session_state.unidade_selecionada or "kg"

    st.markdown(f"**Inserir produtividade** — unidade atual: `{unidade}`")

    # Inicializa valores no session_state para manter edição estável
    for idx, row in gdf.iterrows():
        key = f"valor_pt_{idx}"
        if key not in st.session_state:
            st.session_state[key] = float(row.get("valor", 0.0) or 0.0)

    # Formulário evita recarregamento prematuro e salva com Enter
    with st.form("form_produtividade", clear_on_submit=False):
        # grade simples para reduzir altura: 3 colunas
        ncols = 3
        cols = st.columns(ncols)
        for idx, _ in gdf.iterrows():
            col = cols[idx % ncols]
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
        # Atualiza a coluna 'valor' a partir do estado do formulário
        for idx in gdf.index:
            gdf.at[idx, "valor"] = float(st.session_state.get(f"valor_pt_{idx}", 0.0))

        # Converte para kg conforme a unidade global
        gdf["maduro_kg"] = pd.to_numeric(
            gdf["valor"].apply(lambda v: converter_para_kg(v, unidade)),
            errors="coerce"
        ).fillna(0.0)

        st.session_state.gdf_pontos = gdf
        st.success("Produtividades salvas e convertidas para kg.")

def salvar_no_streamlit_cloud():
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

    area_path = os.path.join(save_dir, "area_amostral.gpkg")
    st.session_state.gdf_poligono.to_file(area_path, driver="GPKG")

    pontos_path = None
    if st.session_state.get("gdf_pontos") is not None and not st.session_state.gdf_pontos.empty:
        pontos_path = os.path.join(save_dir, "pontos_produtividade.gpkg")
        st.session_state.gdf_pontos.to_file(pontos_path, driver="GPKG")

    params_path = os.path.join(save_dir, "parametros_area.json")
    parametros = {
        "densidade_pes_ha": st.session_state.densidade_plantas,
        "produtividade_media_sacas_ha": st.session_state.produtividade_media,
    }
    with open(params_path, "w") as f:
        json.dump(parametros, f)

    st.session_state["tmp_save_dir"] = save_dir
    st.session_state["tmp_area_path"] = area_path
    if pontos_path:
        st.session_state["tmp_pontos_path"] = pontos_path
    st.session_state["tmp_params_path"] = params_path

    # Salvar silenciosamente (sem exibir links/caminhos)
    st.success("✅ Dados salvos na nuvem para uso nas próximas etapas.")

# =======================
# ---- LAYOUT EM UMA COLUNA: MAPA primeiro, CONTROLES depois
# =======================

# Mapa de visualização (estável na área amostral)
st.markdown("<h4 style='margin-bottom:4px'>Mapa de visualização</h4>", unsafe_allow_html=True)
mapa = create_map()
# a classe 'streamlit-folium' é injetada pelo pacote; o CSS acima reduz o espaço depois do mapa
mapa_data = st_folium(mapa, width=900, height=520, key='mapa_principal')

# sem linhas em branco entre o mapa e os controles:
st.markdown('<div class="controls-title">Controles</div>', unsafe_allow_html=True)

# Captura desenho no mapa
if mapa_data and mapa_data.get('last_active_drawing'):
    geometry = mapa_data['last_active_drawing']['geometry']
    gdf = gpd.GeoDataFrame(geometry=[shape(geometry)], crs="EPSG:4326")
    if st.session_state.get('drawing_mode') == 'amostral':
        st.session_state.gdf_poligono = gdf
        st.session_state.map_fit_bounds = _fit_bounds_from_gdf(st.session_state.gdf_poligono)
        st.session_state.drawing_mode = None
        st.success("Área amostral definida!")
        time.sleep(0.3)
        st.rerun()

# Controles (abaixo do mapa), compactos
st.markdown('<div class="controls-title">Controles</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="controls-group">', unsafe_allow_html=True)

    uploaded_area = st.file_uploader("Área amostral (.gpkg)", type=['gpkg'], key='upload_area')
    if uploaded_area:
        processar_arquivo_carregado(uploaded_area, tipo='amostral')

    uploaded_pontos = st.file_uploader("Pontos de produtividade (.gpkg)", type=['gpkg'], key='upload_pontos')
    if uploaded_pontos:
        processar_arquivo_carregado(uploaded_pontos, tipo='pontos')

    cbtn1, cbtn2, cbtn3, cbtn4 = st.columns([1,1,1,1])
    with cbtn1:
        if st.button("▶️ Área amostral"):
            st.session_state.drawing_mode = 'amostral'
            st.session_state.modo_insercao = None
            st.success("Modo desenho ativado: Área Amostral - Desenhe no mapa")
            time.sleep(0.3)
            st.rerun()
    with cbtn2:
        if st.button("🔢 Gerar pontos automáticos (2/ha)"):
            if st.session_state.gdf_poligono is not None:
                gerar_pontos_automaticos()
            else:
                st.warning("Defina a área amostral primeiro.")
    with cbtn3:
        if st.button("✏️ Inserir pontos manualmente"):
            st.session_state.modo_insercao = 'manual'
            st.rerun()
    with cbtn4:
        if st.button("📝 Inserir produtividade"):
            inserir_produtividade()

    st.markdown('<div class="sub-mini">Dados da área amostral</div>', unsafe_allow_html=True)
    st.session_state.densidade_plantas = st.number_input("Densidade (plantas/ha):", value=float(st.session_state.densidade_plantas or 0))
    st.session_state.produtividade_media = st.number_input("Produtividade média última safra (sacas/ha):", value=float(st.session_state.produtividade_media or 0))

    cact1, cact2, cact3 = st.columns([1,1,1])
    with cact1:
        if st.button("💾 Salvar pontos"):
            salvar_pontos()
    with cact2:
        if st.button("💾 Exportar dados"):
            exportar_dados()
    with cact3:
        if st.button("☁️ Salvar dados na nuvem"):
            salvar_no_streamlit_cloud()

    if st.session_state.get('modo_insercao') == 'manual':
        inserir_ponto_manual()

    st.markdown('<div class="sub-mini">Produtividade</div>', unsafe_allow_html=True)
    st.session_state.unidade_selecionada = st.selectbox(
        "Unidade:", ['kg', 'latas', 'litros'],
        index=['kg','latas','litros'].index(st.session_state.unidade_selecionada or 'kg')
    )

    if st.button("🗑️ Limpar área"):
        st.session_state.gdf_poligono = None        
        st.session_state.gdf_pontos = None
        st.session_state.map_fit_bounds = None
        st.success("Áreas limpas!")

    st.markdown('</div>', unsafe_allow_html=True)
