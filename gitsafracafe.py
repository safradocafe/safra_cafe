import ee
import json
import streamlit as st
import geemap
# Configuração da página
st.set_page_config(layout="wide")
st.title("Google Earth Engine no Streamlit 🌍")

# Inicialização do GEE com tratamento de erro
try:
    # Verifica se as credenciais existem
    if "GEE_CREDENTIALS" not in st.secrets:
        st.error("❌ Credenciais do GEE não encontradas em secrets.toml!")
    else:
        # Carrega as credenciais
        credentials_dict = dict(st.secrets["GEE_CREDENTIALS"])
        credentials_json = json.dumps(credentials_dict)

        # Inicializa o GEE
        credentials = ee.ServiceAccountCredentials(
            email=credentials_dict["client_email"],
            key_data=credentials_json
        )
        ee.Initialize(credentials)
        st.success("✅ Google Earth Engine inicializado com sucesso!")

        # Exemplo: Carrega um mapa do GEE
        Map = geemap.Map(center=(40, -100), zoom=4)
        Map.add_basemap("SATELLITE")  # Adiciona imagem de satélite
        Map.to_streamlit()  # Renderiza o mapa no Streamlit

except Exception as e:
    st.error(f"🚨 Erro ao inicializar o GEE: {str(e)}")

################################
# INICIAR GERAÇÃO DO MAPA
################################

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
from shapely.geometry import Point, mapping, shape
from fiona.drvsupport import supported_drivers
from streamlit_folium import st_folium
import folium
from io import BytesIO
import base64

# Configuração inicial
st.set_page_config(layout="wide")
st.title("Sistema de Mapeamento de Produtividade de Café")

# Verificação segura se o GEE já foi inicializado
if not ee.data._initialized:  # Verifica o estado interno da API
    try:
        ee.Initialize()
        st.success("✅ Google Earth Engine inicializado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao inicializar GEE: {str(e)}")
        st.stop()
else:
    st.info("ℹ️ Google Earth Engine já estava inicializado")

# Variáveis de estado (substituem as variáveis globais)
if 'gdf_poligono' not in st.session_state:
    st.session_state.gdf_poligono = None
if 'gdf_pontos' not in st.session_state:
    st.session_state.gdf_pontos = None
if 'gdf_poligono_total' not in st.session_state:
    st.session_state.gdf_poligono_total = None
if 'unidade_selecionada' not in st.session_state:
    st.session_state.unidade_selecionada = 'kg'
if 'densidade_plantas' not in st.session_state:
    st.session_state.densidade_plantas = None
if 'produtividade_media' not in st.session_state:
    st.session_state.produtividade_media = None

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

def safe_st_folium(m, width=800, height=600):
    """Wrapper seguro para st_folium com fallback"""
    try:
        from streamlit_folium import st_folium
        return st_folium(
            m,
            width=width,
            height=height,
            returned_objects=["last_clicked"]
        )
    except Exception:
        # Fallback para html direto
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            m.save(f.name)
            html = open(f.name).read()
        return st.components.v1.html(html, width=width, height=height)

# Interface principal
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Controles")
    
    # Upload de arquivos
    uploaded_file = st.file_uploader("Carregar arquivo (.gpkg, .shp, .kml, .kmz)", 
                                   type=['gpkg', 'shp', 'kml', 'kmz'],
                                   accept_multiple_files=True)
    
    # Controles de área
    if st.button("▶️ Área Amostral"):
        st.session_state.modo_desenho = 'amostral'
        st.success("Modo desenho ativado: Área Amostral")
    
    if st.button("▶️ Área Total"):
        st.session_state.modo_desenho = 'total'
        st.success("Modo desenho ativado: Área Total")
    
    if st.button("🗑️ Limpar Área"):
        st.session_state.gdf_poligono = None
        st.session_state.gdf_poligono_total = None
        st.session_state.gdf_pontos = None
        st.success("Áreas limpas!")
    
    # Parâmetros da área
    st.subheader("Parâmetros da Área")
    st.session_state.densidade_plantas = st.number_input("Plantas por hectare:", value=0.0)
    st.session_state.produtividade_media = st.number_input("Produtividade média (sacas/ha):", value=0.0)
    
    # Controles de pontos
    if st.button("🔢 Gerar pontos automaticamente"):
        if st.session_state.gdf_poligono is not None:
            gerar_pontos_automaticos()
    
    if st.button("✏️ Inserir pontos manualmente"):
        st.session_state.inserir_manual = True
        st.info("Clique no mapa para adicionar pontos")
    
    # Produtividade
    st.subheader("Produtividade")
    st.session_state.unidade_selecionada = st.selectbox("Unidade:", ['kg', 'latas', 'litros'])
    
    if st.button("📝 Inserir produtividade"):
        if st.session_state.gdf_pontos is not None:
            inserir_produtividade()
    
    # Exportação
    if st.button("💾 Exportar dados"):
        exportar_dados()

with col2:
    st.header("Mapa Interativo")
    
    # Inicializa o mapa
    m = geemap.Map(center=[-15, -55], zoom=4)
    m.add_basemap('HYBRID')

# Verificação básica (logo após criar o mapa)
if not hasattr(m, '_id'):
    st.error("Erro na criação do mapa Folium")
    st.stop()  # Isso para a execução se houver erro

# Adicione aqui suas camadas/geometrias ao mapa
if st.session_state.gdf_poligono is not None:
    m.add_gdf(st.session_state.gdf_poligono, layer_name="Área Amostral", style={'color': 'blue'})
    m = geemap.Map(center=[-15, -55], zoom=4)
    m.add_basemap('HYBRID')
    
    # Adiciona áreas existentes
    if st.session_state.gdf_poligono is not None:
        m.add_gdf(st.session_state.gdf_poligono, layer_name="Área Amostral", style={'color': 'blue'})
            
    # Adiciona pontos existentes
    if st.session_state.gdf_pontos is not None:
        for idx, row in st.session_state.gdf_pontos.iterrows():
            color = 'green' if row['coletado'] else 'orange'
            marker = folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"Ponto {idx+1}"
            )
            marker.add_to(m)
    
    # Exibe o mapa
map_output = safe_st_folium(m)  # Substitua st_folium() por safe_st_folium()

# Processamento dos cliques (se necessário)
    if map_output and map_output.get("last_clicked"):
        click_lat = map_output["last_clicked"]["lat"]
        click_lng = map_output["last_clicked"]["lng"]
               
        if st.session_state.get('inserir_manual'):
            adicionar_ponto(click_lat, click_lng, "manual")
            st.session_state.inserir_manual = False
            st.rerun()

# Implementação das funções principais
def processar_arquivo_carregado(uploaded_file):
    try:
        # Cria um arquivo temporário
        temp_file = f"./temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Processa de acordo com o tipo de arquivo
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
        
        # Remove arquivo temporário
        os.remove(temp_file)
        
        # Atualiza estado conforme o modo
        if st.session_state.modo_desenho == 'amostral':
            st.session_state.gdf_poligono = gdf
            st.success("Área amostral carregada com sucesso!")
        elif st.session_state.modo_desenho == 'total':
            st.session_state.gdf_poligono_total = gdf
            st.success("Área total carregada com sucesso!")
        
        return gdf
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {str(e)}")
        return None

def gerar_pontos_automaticos():
    if st.session_state.gdf_poligono is None:
        st.warning("Defina a área amostral primeiro!")
        return
    
    centroid = st.session_state.gdf_poligono.geometry.centroid.iloc[0]
    epsg = get_utm_epsg(centroid.x, centroid.y)
    gdf_utm = st.session_state.gdf_poligono.to_crs(epsg=epsg)
    area_ha = gdf_utm.geometry.area.sum() / 10000
    lado = np.sqrt(5000)  # 2 pontos por hectare

    bounds = gdf_utm.total_bounds
    x_coords = np.arange(bounds[0], bounds[2], lado)
    y_coords = np.arange(bounds[1], bounds[3], lado)

    pontos = [Point(x, y) for x in x_coords for y in y_coords 
              if gdf_utm.geometry.iloc[0].contains(Point(x, y))]
    
    gdf_pontos = gpd.GeoDataFrame(geometry=pontos, crs=gdf_utm.crs).to_crs("EPSG:4326")
    
    # Adiciona metadados
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

def inserir_produtividade():
    if st.session_state.gdf_pontos is None or st.session_state.gdf_pontos.empty:
        st.warning("Nenhum ponto disponível!")
        return
    
    with st.expander("Editar Produtividade dos Pontos"):
        for idx, row in st.session_state.gdf_pontos.iterrows():
            cols = st.columns([1, 2, 2, 1])
            with cols[0]:
                st.write(f"**Ponto {idx+1}**")
                st.write(f"Lat: {row['latitude']:.5f}")
                st.write(f"Lon: {row['longitude']:.5f}")
            with cols[1]:
                novo_valor = st.number_input(
                    "Valor", 
                    value=float(row['valor']),
                    key=f"valor_{idx}"
                )
            with cols[2]:
                nova_unidade = st.selectbox(
                    "Unidade",
                    ['kg', 'latas', 'litros'],
                    index=['kg', 'latas', 'litros'].index(row['unidade']),
                    key=f"unidade_{idx}"
                )
            with cols[3]:
                coletado = st.checkbox(
                    "Coletado",
                    value=row['coletado'],
                    key=f"coletado_{idx}"
                )
            
            # Atualiza os dados
            st.session_state.gdf_pontos.at[idx, 'valor'] = novo_valor
            st.session_state.gdf_pontos.at[idx, 'unidade'] = nova_unidade
            st.session_state.gdf_pontos.at[idx, 'coletado'] = coletado
            st.session_state.gdf_pontos.at[idx, 'maduro_kg'] = converter_para_kg(novo_valor, nova_unidade)
        
        if st.button("Salvar Alterações"):
            st.success("Dados de produtividade atualizados!")
            st.rerun()

def exportar_dados():
    if st.session_state.gdf_poligono is None:
        st.warning("Nenhuma área para exportar!")
        return
    
    # Cria um arquivo ZIP na memória
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        # Adiciona parâmetros
        parametros = {
            'densidade_pes_ha': st.session_state.densidade_plantas,
            'produtividade_media_sacas_ha': st.session_state.produtividade_media
        }
        zip_file.writestr('parametros_area.json', json.dumps(parametros))
        
        # Adiciona polígonos
        if st.session_state.gdf_poligono is not None:
            poligono_buffer = BytesIO()
            st.session_state.gdf_poligono.to_file(poligono_buffer, driver='GPKG')
            zip_file.writestr('area_poligono.gpkg', poligono_buffer.getvalue())
        
        if st.session_state.gdf_poligono_total is not None:
            poligono_total_buffer = BytesIO()
            st.session_state.gdf_poligono_total.to_file(poligono_total_buffer, driver='GPKG')
            zip_file.writestr('area_total_poligono.gpkg', poligono_total_buffer.getvalue())
        
        # Adiciona pontos
        if st.session_state.gdf_pontos is not None:
            pontos_buffer = BytesIO()
            st.session_state.gdf_pontos.to_file(pontos_buffer, driver='GPKG')
            zip_file.writestr('pontos_produtividade.gpkg', pontos_buffer.getvalue())
    
    # Cria botão de download
    st.download_button(
        label="⬇️ Baixar todos os dados",
        data=zip_buffer.getvalue(),
        file_name="dados_produtividade.zip",
        mime="application/zip"
    )
    st.success("Dados preparados para exportação!")

# Processa arquivos carregados
if uploaded_file:
    processar_arquivo_carregado(uploaded_file[0])  # Pega o primeiro arquivo se múltiplos
