import json
import streamlit as st
import geemap
import ee 
# Configuração da página
#st.set_page_config(layout="wide")
#st.title("Google Earth Engine no Streamlit 🌍")

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
        #st.success("✅ Google Earth Engine inicializado com sucesso!")

        # Exemplo: Carrega um mapa do GEE
        Map = geemap.Map(center=(40, -100), zoom=4)
        Map.add_basemap("SATELLITE")  # Adiciona imagem de satélite
        Map.to_streamlit()  # Renderiza o mapa no Streamlit

except Exception as e:
    st.error(f"🚨 Erro ao inicializar o GEE: {str(e)}")

################################
# INICIAR GERAÇÃO DO MAPA
################################

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
import pydeck as pdk
from io import BytesIO
import base64
import os

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


def create_map():
    """Cria um mapa Pydeck com as camadas necessárias"""
    layers = []

    # Adiciona polígono se existir
    if st.session_state.gdf_poligono is not None:
        polygon_layer = pdk.Layer(
            "PolygonLayer",
            data=st.session_state.gdf_poligono,
            get_polygon="geometry.coordinates",
            get_fill_color=[0, 0, 255, 100],
            get_line_color=[0, 0, 255],
            pickable=True,
            auto_highlight=True,
        )
        layers.append(polygon_layer)

    # Adiciona pontos se existirem
    if st.session_state.gdf_pontos is not None:
        points_df = st.session_state.gdf_pontos[['longitude', 'latitude', 'coletado', 'Code', 'valor', 'unidade']].copy()
        points_df['color'] = points_df['coletado'].apply(lambda x: [0, 255, 0, 200] if x else [255, 165, 0, 200])
        point_layer = pdk.Layer(
            "ScatterplotLayer",
            data=points_df,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius=50,
            pickable=True,
        )
        layers.append(point_layer)

    # Configuração inicial do mapa
    view_state = pdk.ViewState(
        latitude=-15,
        longitude=-55,
        zoom=4,
        pitch=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/satellite-v9",
        tooltip={
            "html": """
                <b>Ponto:</b> {Code}<br/>
                <b>Produtividade:</b> {valor} {unidade}<br/>
                <b>Coletado:</b> {coletado}
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
    )


def main():
    st.set_page_config(layout="wide")
    st.title("Sistema de previsão avançada da produtividade do café")
    st.markdown("""
        Este é um projeto de geotecnologia para previsão da produtividade do café,
        com o uso de imagens do sensor MSI/Sentinel-2A e algoritmos de machine learning.
    """)
    st.subheader("Etapas do projeto e aplicações práticas")

    st.markdown("""
    - **Área produtiva:** delimitação das áreas de interesse (amostral e total) e geração de pontos amostrais (2 pontos/hectare).
    - **Coleta de dados:** inserção de informações de produtividade e seleção automática de imagens de satélite (sensor MSI/Sentinel-2A), com 5% de nuvens.
    - **Cálculo de índices espectrais**: NDVI, GNDVI, MSAVI2 (relação com o desenvolvimento vegetativo); NDRE e CCCI (conteúdo de clorofila); NDMI, NDWI e TWI2 (umidade do solo, conteúdo de água das folhas e umidade do ar) e NBR (estresse térmico).  
    - **Avaliação da correlação entre a produtividade e índices espectrais**: teste de Shapiro-Wilk para normalidade dos dados e correlação de Pearson (maioria normal) ou Spearman (não normal).
    - **Modelagem de produtividade:** treinamento com 11 algoritmos de machine learning, avaliação do desempenho (métricas R² e RMSE) e escolha do melhor modelo para previsão da produtividade.
    - **Geração de mapas interativos:** visualização da variabilidade espacial da produtividade e estimativa antecipada da colheita.
    - **Exportação de dados:** resultados em formato compatível com SIG, para integração com ferramentas de gestão agrícola.
    - **Comparação entre safras:** avaliação de padrões visuais e produtivos ao longo do tempo.
    - **Análise detalhada:** identificação de áreas promissoras ou com necessidade de atenção para o planejamento da próxima safra.
    """)

    # Inicialização do estado da sessão
    if 'gdf_poligono' not in st.session_state:
        st.session_state.gdf_poligono = None
    if 'gdf_pontos' not in st.session_state:
        st.session_state.gdf_pontos = None
    if 'gdf_poligono_total' not in st.session_state:
        st.session_state.gdf_poligono_total = None

    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("Controles")

        # Upload de arquivos
        uploaded_file = st.file_uploader("Carregar arquivo (.gpkg, .shp, .kml, .kmz)",
                                         type=['gpkg', 'shp', 'kml', 'kmz'],
                                         accept_multiple_files=True)

        if uploaded_file:
            processar_arquivo_carregado(uploaded_file[0])

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

        st.subheader("Parâmetros da Área")
        st.session_state.densidade_plantas = st.number_input("Plantas por hectare:", value=0.0)
        st.session_state.produtividade_media = st.number_input("Produtividade média (sacas/ha):", value=0.0)

        if st.button("🔢 Gerar pontos automaticamente"):
            if st.session_state.gdf_poligono is not None:
                gerar_pontos_automaticos()

        if st.button("✏️ Inserir pontos manualmente"):
            st.session_state.inserir_manual = True
            st.info("Clique no mapa para adicionar pontos")

        st.subheader("Produtividade")
        st.session_state.unidade_selecionada = st.selectbox("Unidade:", ['kg', 'latas', 'litros'])

        if st.button("📝 Inserir produtividade"):
            if st.session_state.gdf_pontos is not None:
                inserir_produtividade()

        if st.button("💾 Exportar dados"):
            exportar_dados()

    with col2:
        st.header("Mapa Interativo")
        deck = create_map()
        st.pydeck_chart(deck)

        if hasattr(deck, 'last_clicked'):
            if deck.last_clicked and st.session_state.get('inserir_manual'):
                click_data = deck.last_clicked
                if isinstance(click_data, dict) and 'latitude' in click_data and 'longitude' in click_data:
                    adicionar_ponto(click_data['latitude'], click_data['longitude'], "manual")
                    st.session_state.inserir_manual = False
                    st.rerun()


if __name__ == "__main__":
    main()

