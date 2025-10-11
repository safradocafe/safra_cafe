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

# ---- Estilo (sem set_page_config aqui) ----
st.markdown("""
    <style>
    .block-container { padding-top: 0rem !important; padding-bottom: 1rem; }
    header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---- Estado inicial ----
for key in [
    'gdf_poligono', 'gdf_pontos', 'gdf_poligono_total', 'unidade_selecionada',
    'densidade_plantas', 'produtividade_media', 'mapa_data', 'modo_insercao',
    'drawing_mode'  # usar sempre esta chave
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

def create_map():
    m = folium.Map(location=[-15, -55], zoom_start=4, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satélite',
        overlay=False,
        control=True
    ).add_to(m)

    draw = folium.plugins.Draw(
        draw_options={
            'polyline': False,
            'rectangle': True,
            'circle': False,
            'circlemarker': False,
            'marker': False,
            'polygon': {
                'allowIntersection': False,
                'showArea': True,
                'repeatMode': False
            }
        },
        export=False,
        position='topleft'
    )
    draw.add_to(m)

    if st.session_state.gdf_poligono is not None:
        folium.GeoJson(
            st.session_state.gdf_poligono,
            name="Área Amostral",
            style_function=lambda x: {"color": "blue", "fillColor": "blue", "fillOpacity": 0.3}
        ).add_to(m)

    if st.session_state.gdf_poligono_total is not None:
        folium.GeoJson(
            st.session_state.gdf_poligono_total,
            name="Área Total",
            style_function=lambda x: {"color": "green", "fillColor": "green", "fillOpacity": 0.3}
        ).add_to(m)

    if st.session_state.gdf_pontos is not None and not st.session_state.gdf_pontos.empty:
        for _, row in st.session_state.gdf_pontos.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=0.7,
                popup=f"Ponto: {row['Code']}<br>Produtividade: {row['maduro_kg']}"
            ).add_to(m)

    folium.LayerControl().add_to(m)
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

        gdf = gpd.read_file(temp_file)
        os.remove(temp_file)

        if tipo == 'amostral':
            if gdf.empty or not any(gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])):
                st.error("❌ O arquivo da área amostral deve conter polígonos")
                return None
            st.session_state.gdf_poligono = gdf
            st.success("✅ Área amostral carregada com sucesso!")

        elif tipo == 'pontos':
            required_cols = ['Code', 'maduro_kg', 'latitude', 'longitude', 'geometry']
            faltando = [col for col in required_cols if col not in gdf.columns]
            if faltando:
                st.error(f"❌ Faltam colunas obrigatórias: {', '.join(faltando)}")
                return None
            if not any(gdf.geometry.type.isin(['Point', 'MultiPoint'])):
                st.error("❌ O arquivo de pontos deve conter geometrias do tipo Ponto")
                return None
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
                gdf['latitude'] = gdf.geometry.y
                gdf['longitude'] = gdf.geometry.x
            st.session_state.gdf_pontos = gdf
            st.success(f"✅ {len(gdf)} pontos carregados com sucesso!")
            st.info(f"Colunas disponíveis: {', '.join(gdf.columns)}")

        return gdf

    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {str(e)}")
        return None

def gerar_pontos_automaticos():
    if st.session_state.gdf_poligono is None:
        st.warning("Defina a área amostral primeiro!")
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
    if st.session_state.gdf_poligono is None or st.session_state.gdf_poligono_total is None:
        st.warning("⚠️ É necessário definir ambas as áreas (amostral e total) antes de exportar!")
        return

    tmp_dir = "/tmp/export_zip"
    os.makedirs(tmp_dir, exist_ok=True)

    poligono_path = os.path.join(tmp_dir, "area_poligono.gpkg")
    poligono_total_path = os.path.join(tmp_dir, "area_total_poligono.gpkg")
    pontos_path = os.path.join(tmp_dir, "pontos_produtividade.gpkg")
    params_path = os.path.join(tmp_dir, "parametros_area.json")
    zip_path = os.path.join(tmp_dir, "dados_produtividade.zip")

    st.session_state.gdf_poligono.to_file(poligono_path, driver="GPKG")
    st.session_state.gdf_poligono_total.to_file(poligono_total_path, driver="GPKG")
    if st.session_state.gdf_pontos is not None and not st.session_state.gdf_pontos.empty:
        st.session_state.gdf_pontos.to_file(pontos_path, driver="GPKG")

    parametros = {
        'densidade_pes_ha': st.session_state.densidade_plantas,
        'produtividade_media_sacas_ha': st.session_state.produtividade_media
    }
    with open(params_path, "w") as f:
        json.dump(parametros, f)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(poligono_path, arcname="area_poligono.gpkg")
        zipf.write(poligono_total_path, arcname="area_total_poligono.gpkg")
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
    if st.session_state.gdf_pontos is None or st.session_state.gdf_pontos.empty:
        st.warning("Nenhum ponto disponível!")
        return

    with st.expander("Editar dados de produtividade"):
        for idx, row in st.session_state.gdf_pontos.iterrows():
            cols = st.columns([1, 3, 3, 2])
            with cols[0]:
                st.write(f"**Ponto {idx+1}**")
                st.write(f"Lat: {row['latitude']:.5f}")
                st.write(f"Lon: {row['longitude']:.5f}")
            with cols[1]:
                novo_valor = st.number_input("Valor", value=float(row['valor']), key=f"valor_{idx}")
            with cols[2]:
                nova_unidade = st.selectbox("Unidade", ['kg', 'latas', 'litros'],
                                            index=['kg', 'latas', 'litros'].index(row['unidade']),
                                            key=f"unidade_{idx}")
            with cols[3]:
                coletado = st.checkbox("Coletado", value=row['coletado'], key=f"coletado_{idx}")

            st.session_state.gdf_pontos.at[idx, 'valor'] = novo_valor
            st.session_state.gdf_pontos.at[idx, 'unidade'] = nova_unidade
            st.session_state.gdf_pontos.at[idx, 'coletado'] = coletado
            st.session_state.gdf_pontos.at[idx, 'maduro_kg'] = converter_para_kg(novo_valor, nova_unidade)

        if st.button("Salvar alterações"):
            st.success("Dados de produtividade atualizados.")
            st.rerun()

# =======================
# ---- LAYOUT PRINCIPAL (fora de funções!) ----
# =======================

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<h4>Controles</h4>", unsafe_allow_html=True)

    uploaded_area = st.file_uploader("1. Área amostral (.gpkg)", type=['gpkg'], key='upload_area')
    if uploaded_area:
        processar_arquivo_carregado(uploaded_area, tipo='amostral')

    uploaded_pontos = st.file_uploader("2. Pontos de Produtividade (.gpkg)", type=['gpkg'], key='upload_pontos')
    if uploaded_pontos:
        processar_arquivo_carregado(uploaded_pontos, tipo='pontos')

    if st.button("▶️ Área amostral"):
        st.session_state.drawing_mode = 'amostral'
        st.session_state.modo_insercao = None
        st.success("Modo desenho ativado: Área Amostral - Desenhe no mapa")
        time.sleep(0.3)
        st.rerun()

    st.subheader("Dados da área amostral")
    st.session_state.densidade_plantas = st.number_input("Densidade (plantas/ha):", value=float(st.session_state.densidade_plantas or 0))
    st.session_state.produtividade_media = st.number_input("Produtividade média última safra (sacas/ha):", value=float(st.session_state.produtividade_media or 0))

    if st.button("🔢 Gerar pontos automáticos (2/ha)"):
        if st.session_state.gdf_poligono is not None:
            gerar_pontos_automaticos()
        else:
            st.warning("Defina a área amostral primeiro.")

    if st.button("✏️ Inserir pontos manualmente"):
        st.session_state.modo_insercao = 'manual'
        st.rerun()

    if st.button("📝 Inserir produtividade"):
        inserir_produtividade()

    if st.button("💾 Salvar pontos"):
        salvar_pontos()

    if st.button("▶️ Área total"):
        st.session_state.drawing_mode = 'total'
        st.session_state.modo_insercao = None
        st.success("Modo desenho ativado: Área Total - Desenhe no mapa")
        time.sleep(0.3)
        st.rerun()

    if st.button("💾 Exportar dados"):
        exportar_dados()

    if st.button("🗑️ Limpar área"):
        st.session_state.gdf_poligono = None
        st.session_state.gdf_poligono_total = None
        st.session_state.gdf_pontos = None
        st.success("Áreas limpas!")

    st.subheader("Produtividade")
    st.session_state.unidade_selecionada = st.selectbox("Unidade:", ['kg', 'latas', 'litros'],
                                                        index=['kg','latas','litros'].index(st.session_state.unidade_selecionada or 'kg'))

with col2:
    st.markdown("<h4>Mapa de visualização</h4>", unsafe_allow_html=True)
    mapa = create_map()
    mapa_data = st_folium(mapa, width=700, height=500, key='mapa_principal')

    # Se estiver no modo manual, mostra o formulário para inserir ponto
    if st.session_state.get('modo_insercao') == 'manual':
        inserir_ponto_manual()

    # Captura desenho no mapa
    if mapa_data and mapa_data.get('last_active_drawing'):
        geometry = mapa_data['last_active_drawing']['geometry']
        gdf = gpd.GeoDataFrame(geometry=[shape(geometry)], crs="EPSG:4326")

        if st.session_state.get('drawing_mode') == 'amostral':
            st.session_state.gdf_poligono = gdf
            st.session_state.drawing_mode = None
            st.success("Área amostral definida!")
            time.sleep(0.3)
            st.rerun()

        elif st.session_state.get('drawing_mode') == 'total':
            st.session_state.gdf_poligono_total = gdf
            st.session_state.drawing_mode = None
            st.success("Área total definida!")
            time.sleep(0.3)
            st.rerun()

# ---- Salvar no /tmp do container (opcional) ----
def salvar_no_streamlit_cloud():
    if st.session_state.get("gdf_poligono") is None or \
       st.session_state.get("gdf_poligono_total") is None or \
       st.session_state.get("gdf_pontos") is None or \
       st.session_state.gdf_pontos.empty:
        st.warning("⚠️ Certifique-se de que áreas e pontos foram definidos!")
        return

    if st.session_state.get("densidade_plantas") is None or \
       st.session_state.get("produtividade_media") is None:
        st.warning("⚠️ Parâmetros de densidade e produtividade não definidos!")
        return

    temp_dir = "/tmp/streamlit_dados"
    os.makedirs(temp_dir, exist_ok=True)

    st.session_state.gdf_poligono.to_file(f"{temp_dir}/area_poligono.gpkg", driver="GPKG")
    st.session_state.gdf_poligono_total.to_file(f"{temp_dir}/area_total_poligono.gpkg", driver="GPKG")
    st.session_state.gdf_pontos.to_file(f"{temp_dir}/pontos_produtividade.gpkg", driver="GPKG")

    parametros = {
        'densidade_pes_ha': st.session_state.densidade_plantas,
        'produtividade_media_sacas_ha': st.session_state.produtividade_media
    }
    with open(f"{temp_dir}/parametros_area.json", "w") as f:
        json.dump(parametros, f)

    st.success("✅ Arquivos salvos temporariamente em /tmp (container).")
    st.info("Esses arquivos podem ser usados pelas próximas páginas.")

if st.button("☁️ Salvar dados na nuvem"):
    salvar_no_streamlit_cloud()
