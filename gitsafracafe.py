import os
import json
import time
import random
import string
import numpy as np
import pandas as pd
import zipfile
import geemap
import ee
import geopandas as gpd
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium
import folium
from io import BytesIO

# Inicialização do Earth Engine
def init_gee():
    try:
        if "GEE_CREDENTIALS" not in st.secrets:
            st.error("❌ Credenciais do GEE não encontradas!")
            st.stop()
        
        creds = st.secrets["GEE_CREDENTIALS"]
        credentials = ee.ServiceAccountCredentials(
            creds["client_email"],
            key_data=json.dumps(dict(creds))
        )
        ee.Initialize(credentials)
    except Exception as e:
        st.error(f"🚨 Erro ao inicializar o GEE: {str(e)}")
        st.stop()

init_gee()

# Configuração da página
st.set_page_config(layout="wide")
st.title("Sistema de previsão avançada da produtividade do café")
st.markdown("""
    Este é um projeto de geotecnologia para previsão da produtividade do café,
    com o uso de imagens do sensor MSI/Sentinel-2A e algoritmos de machine learning.
""")

# Inicialização do estado da sessão
def init_session_state():
    defaults = {
        'gdf_poligono': None,
        'gdf_pontos': None,
        'gdf_poligono_total': None,
        'unidade_selecionada': 'kg',
        'densidade_plantas': None,
        'produtividade_media': None,
        'modo_desenho': None,
        'inserir_manual': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Funções auxiliares
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
    try:
        m = geemap.Map(center=[-15, -55], zoom=4)
        try:
            m.add_basemap('HYBRID')
        except Exception:
            m.add_basemap('OpenStreetMap')
        return m
    except Exception as e:
        st.error(f"Falha na criação do mapa: {str(e)}")
        st.stop()

def processar_arquivo(uploaded_file):
    try:
        temp_file = f"./temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name.endswith('.gpkg'):
            gdf = gpd.read_file(temp_file)
        elif uploaded_file.name.endswith('.shp'):
            gdf = gpd.read_file(temp_file)
        elif uploaded_file.name.endswith('.kml'):
            gdf = gpd.read_file(temp_file, driver='KML')
        elif uploaded_file.name.endswith('.kmz'):
            with zipfile.ZipFile(temp_file, 'r') as kmz:
                kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
                if kml_files:
                    with kmz.open(kml_files[0]) as kml:
                        gdf = gpd.read_file(kml, driver='KML')
        
        os.remove(temp_file)
        return gdf
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {str(e)}")
        return None

def gerar_pontos():
    if st.session_state.gdf_poligono is None:
        st.warning("Defina a área amostral primeiro!")
        return
    
    centroid = st.session_state.gdf_poligono.geometry.centroid.iloc[0]
    epsg = get_utm_epsg(centroid.x, centroid.y)
    gdf_utm = st.session_state.gdf_poligono.to_crs(epsg=epsg)
    area_ha = gdf_utm.geometry.area.sum() / 10000
    lado = np.sqrt(5000)

    bounds = gdf_utm.total_bounds
    x_coords = np.arange(bounds[0], bounds[2], lado)
    y_coords = np.arange(bounds[1], bounds[3], lado)

    pontos = [Point(x, y) for x in x_coords for y in y_coords 
              if gdf_utm.geometry.iloc[0].contains(Point(x, y))]
    
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
    st.success(f"{len(gdf_pontos)} pontos gerados! Área: {area_ha:.2f} ha")

def adicionar_ponto(lat, lon, metodo):
    ponto = Point(lon, lat)
    
    if st.session_state.gdf_pontos is None:
        gdf_pontos = gpd.GeoDataFrame(columns=[
            'geometry', 'Code', 'valor', 'unidade', 'maduro_kg',
            'coletado', 'latitude', 'longitude', 'metodo'
        ], geometry='geometry', crs="EPSG:4326")
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

def editar_produtividade():
    if st.session_state.gdf_pontos is None or st.session_state.gdf_pontos.empty:
        st.warning("Nenhum ponto disponível!")
        return
    
    with st.expander("Editar Produtividade dos Pontos"):
        for idx, row in st.session_state.gdf_pontos.iterrows():
            unique_key = f"prod_{idx}_{time.time_ns()}"
            cols = st.columns([1, 2, 2, 1])
            with cols[0]:
                st.write(f"**Ponto {idx+1}**")
                st.write(f"Lat: {row['latitude']:.5f}")
                st.write(f"Lon: {row['longitude']:.5f}")
            with cols[1]:
                novo_valor = st.number_input(
                    "Valor", 
                    value=float(row['valor']), 
                    key=f"valor_{unique_key}"
                )
            with cols[2]:
                nova_unidade = st.selectbox(
                    "Unidade", 
                    ['kg', 'latas', 'litros'],
                    index=['kg', 'latas', 'litros'].index(row['unidade']),
                    key=f"unidade_{unique_key}"
                )
            with cols[3]:
                coletado = st.checkbox(
                    "Coletado", 
                    value=row['coletado'], 
                    key=f"coletado_{unique_key}"
                )
            
            st.session_state.gdf_pontos.at[idx, 'valor'] = novo_valor
            st.session_state.gdf_pontos.at[idx, 'unidade'] = nova_unidade
            st.session_state.gdf_pontos.at[idx, 'coletado'] = coletado
            st.session_state.gdf_pontos.at[idx, 'maduro_kg'] = converter_para_kg(novo_valor, nova_unidade)
        
        if st.button("Salvar Alterações", key=f"salvar_{time.time_ns()}"):
            st.success("Dados atualizados!")
            st.rerun()

def exportar():
    if st.session_state.gdf_poligono is None:
        st.warning("Nenhuma área para exportar!")
        return
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        parametros = {
            'densidade_pes_ha': st.session_state.densidade_plantas,
            'produtividade_media_sacas_ha': st.session_state.produtividade_media
        }
        zip_file.writestr('parametros_area.json', json.dumps(parametros))
        
        if st.session_state.gdf_poligono is not None:
            poligono_buffer = BytesIO()
            st.session_state.gdf_poligono.to_file(poligono_buffer, driver='GPKG')
            zip_file.writestr('area_poligono.gpkg', poligono_buffer.getvalue())
        
        if st.session_state.gdf_poligono_total is not None:
            poligono_total_buffer = BytesIO()
            st.session_state.gdf_poligono_total.to_file(poligono_total_buffer, driver='GPKG')
            zip_file.writestr('area_total_poligono.gpkg', poligono_total_buffer.getvalue())
        
        if st.session_state.gdf_pontos is not None:
            pontos_buffer = BytesIO()
            st.session_state.gdf_pontos.to_file(pontos_buffer, driver='GPKG')
            zip_file.writestr('pontos_produtividade.gpkg', pontos_buffer.getvalue())
    
    st.download_button(
        label="⬇️ Baixar todos os dados",
        data=zip_buffer.getvalue(),
        file_name="dados_produtividade.zip",
        mime="application/zip",
        key=f"download_{time.time_ns()}"
    )
    st.success("Dados preparados para exportação!")

# Interface principal
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Controles")
    
    uploaded_file = st.file_uploader(
        "Carregar arquivo (.gpkg, .shp, .kml, .kmz)",
        type=['gpkg', 'shp', 'kml', 'kmz'],
        accept_multiple_files=False,
        key=f"uploader_{time.time_ns()}"
    )

    if uploaded_file:
        gdf = processar_arquivo(uploaded_file)
        if gdf is not None:
            if st.session_state.modo_desenho == 'amostral':
                st.session_state.gdf_poligono = gdf
                st.success("Área amostral carregada!")
            elif st.session_state.modo_desenho == 'total':
                st.session_state.gdf_poligono_total = gdf
                st.success("Área total carregada!")

    if st.button("▶️ Área Amostral", key=f"btn_amostral_{time.time_ns()}"):
        st.session_state.modo_desenho = 'amostral'
        st.success("Modo desenho ativado: Área Amostral")
    
    if st.button("▶️ Área Total", key=f"btn_total_{time.time_ns()}"):
        st.session_state.modo_desenho = 'total'
        st.success("Modo desenho ativado: Área Total")
    
    if st.button("🗑️ Limpar Área", key=f"btn_limpar_{time.time_ns()}"):
        st.session_state.gdf_poligono = None
        st.session_state.gdf_poligono_total = None
        st.session_state.gdf_pontos = None
        st.success("Áreas limpas!")
    
    st.subheader("Parâmetros da Área")
    st.session_state.densidade_plantas = st.number_input(
        "Plantas por hectare:", 
        value=0.0, 
        key=f"densidade_{time.time_ns()}"
    )
    st.session_state.produtividade_media = st.number_input(
        "Produtividade média (sacas/ha):", 
        value=0.0, 
        key=f"produtividade_{time.time_ns()}"
    )
    
    if st.button("🔢 Gerar pontos automaticamente", key=f"btn_gerar_{time.time_ns()}"):
        gerar_pontos()
    
    if st.button("✏️ Inserir pontos manualmente", key=f"btn_manual_{time.time_ns()}"):
        st.session_state.inserir_manual = True
        st.info("Clique no mapa para adicionar pontos")
    
    st.subheader("Produtividade")
    st.session_state.unidade_selecionada = st.selectbox(
        "Unidade:", 
        ['kg', 'latas', 'litros'], 
        key=f"unidade_{time.time_ns()}"
    )
    
    if st.button("📝 Inserir produtividade", key=f"btn_prod_{time.time_ns()}"):
        editar_produtividade()
    
    if st.button("💾 Exportar dados", key=f"btn_export_{time.time_ns()}"):
        exportar()

with col2:
    st.header("Mapa Interativo")
    m = create_map()
    
    if st.session_state.gdf_poligono is not None:
        m.add_gdf(st.session_state.gdf_poligono, layer_name="Área Amostral", style={'color': 'blue'})
    
    if st.session_state.gdf_pontos is not None:
        for idx, row in st.session_state.gdf_pontos.iterrows():
            color = 'green' if row['coletado'] else 'orange'
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"Ponto {idx+1}"
            ).add_to(m)
    
    map_output = st_folium(m, width=800, height=600, returned_objects=["last_clicked"])
    
    if isinstance(map_output, dict) and "last_clicked" in map_output and st.session_state.get('inserir_manual'):
        click_lat = map_output["last_clicked"]["lat"]
        click_lng = map_output["last_clicked"]["lng"]
        adicionar_ponto(click_lat, click_lng, "manual")
        st.session_state.inserir_manual = False
        st.rerun()
