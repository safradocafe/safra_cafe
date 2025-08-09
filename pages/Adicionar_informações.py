import streamlit as st

st.title("Adicionar informações")
st.write("Insira dados da fazenda")

import json
import streamlit as st
import geemap
import time
import random
import string
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import shape
from io import BytesIO
import base64
import os
import folium
from streamlit_folium import st_folium
import zipfile
from io import BytesIO

# ✅ Configuração da página
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem;
    }
    header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ✅ Inicialização do estado
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
if 'modo_desenho' not in st.session_state:   # ✅ Adicionado
    st.session_state.modo_desenho = None
if 'mapa_data' not in st.session_state:
    st.session_state.mapa_data = None
    
# ✅ Funções auxiliares
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

# ✅ Substituição do mapa para Folium
def create_map():
    # Cria o mapa com OpenStreetMap como camada base
    m = folium.Map(location=[-15, -55], zoom_start=4, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satélite',
        overlay=False,
        control=True
    ).add_to(m)

    # Adiciona controle de desenho (para áreas e pontos)
    draw = folium.plugins.Draw(
        draw_options={
            'polyline': False,
            'rectangle': True,
            'circle': False,
            'circlemarker': False,
            'marker': False,
            'polygon': True
        },
        export=False,
        position='topleft'
    )
    draw.add_to(m)

    # Adiciona o polígono amostral (se existir no session_state)
    if st.session_state.gdf_poligono is not None:
        folium.GeoJson(
            st.session_state.gdf_poligono,
            name="Área Amostral",
            style_function=lambda x: {"color": "blue", "fillColor": "blue", "fillOpacity": 0.3}
        ).add_to(m)

    # Polígono total
    if st.session_state.gdf_poligono_total is not None:
        folium.GeoJson(
            st.session_state.gdf_poligono_total,
            name="Área Total",
            style_function=lambda x: {"color": "green", "fillColor": "green", "fillOpacity": 0.3}
        ).add_to(m)

    # Pontos
    if st.session_state.gdf_pontos is not None:
        for _, row in st.session_state.gdf_pontos.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=0.7,
                popup=f"Ponto: {row['Code']}<br>Produtividade: {row['maduro_kg']} {row['latitude']} {row['longitude']} {row['geometry']}"
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

# ✅ Função para processar arquivo GPKG
def processar_arquivo_carregado(uploaded_file, tipo='amostral'):
    try:
        if uploaded_file is None:
            return None
            
        if not uploaded_file.name.lower().endswith('.gpkg'):
            st.error("❌ O arquivo deve ter extensão .gpkg")
            return None

        temp_file = f"./temp_{uploaded_file.name}"
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
            # Colunas obrigatórias para pontos
            required_cols = ['Code', 'maduro_kg', 'latitude', 'longitude', 'geometry']
            
            # Verifica se todas as colunas obrigatórias existem
            colunas_faltantes = [col for col in required_cols if col not in gdf.columns]
            if colunas_faltantes:
                st.error(f"❌ Arquivo de pontos está faltando colunas obrigatórias: {', '.join(colunas_faltantes)}")
                return None
                
            # Verifica se a geometria é do tipo ponto
            if not any(gdf.geometry.type.isin(['Point', 'MultiPoint'])):
                st.error("❌ O arquivo de pontos deve conter geometrias do tipo Ponto")
                return None
            
            # Converte para o CRS padrão (EPSG:4326) se necessário
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

def processar_pontos_produtividade(uploaded_file):
    """Processa arquivo de pontos de produtividade"""
    try:
        gdf = processar_arquivo_carregado(uploaded_file)  # Usa a função principal sem tipo
        
        if gdf is not None:
                      
            st.session_state.gdf_pontos = gdf
            st.success(f"✅ {len(gdf)} pontos carregados com sucesso!")
            
    except Exception as e:
        st.error(f"Erro ao processar pontos: {str(e)}")

# ✅ Funções para pontos e produtividade (mantidas iguais)
def gerar_pontos_automaticos():
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
    """Prepara os dados para exportação (equivalente ao btn_salvar_pontos)"""
    if st.session_state.gdf_pontos is None or st.session_state.gdf_pontos.empty:
        st.warning("⚠️ Nenhum ponto para salvar!")
        return   
    st.success("✅ Dados dos pontos preparados para exportação!")

def exportar_dados():
    if st.session_state.gdf_poligono is None or st.session_state.gdf_poligono_total is None:
        st.warning("⚠️ É necessário definir ambas as áreas (amostral e total) antes de exportar!")
        return
  
    # Cria buffer ZIP
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Parâmetros da área (JSON)
        parametros = {
            'densidade_pes_ha': st.session_state.densidade_plantas,
            'produtividade_media_sacas_ha': st.session_state.produtividade_media
        }
        zipf.writestr('parametros_area.json', json.dumps(parametros))

        # Polígono amostral (GPKG)
        if st.session_state.gdf_poligono is not None:
            poligono_buffer = BytesIO()
            st.session_state.gdf_poligono.to_file(poligono_buffer, driver='GPKG')
            zipf.writestr('area_poligono.gpkg', poligono_buffer.getvalue())

        # Polígono total (GPKG)
        if st.session_state.gdf_poligono_total is not None:
            poligono_total_buffer = BytesIO()
            st.session_state.gdf_poligono_total.to_file(poligono_total_buffer, driver='GPKG')
            zipf.writestr('area_total_poligono.gpkg', poligono_total_buffer.getvalue())

        # Pontos (GPKG)
        if st.session_state.gdf_pontos is not None:
            pontos_buffer = BytesIO()
            st.session_state.gdf_pontos.to_file(pontos_buffer, driver='GPKG')
            zipf.writestr('pontos_produtividade.gpkg', pontos_buffer.getvalue())

    # Botão de download
    st.download_button(
        label="💾 Exportar dados (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="dados_produtividade.zip",
        mime="application/zip"
    )

# ✅ Função principal
def main():   

    if st.session_state.get('modo_insercao') == 'manual':
        inserir_ponto_manual()
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("Controles")

        # Upload da área amostral
        uploaded_area = st.file_uploader(
            "1. Área Amostral (.gpkg)",
            type=['gpkg'],
            key='upload_area'
        )
        if uploaded_area:
            processar_arquivo_carregado(uploaded_area, tipo='amostral')

        # Upload dos pontos de produtividade
        uploaded_pontos = st.file_uploader(
            "2. Pontos de Produtividade (.gpkg)",
            type=['gpkg'],
            key='upload_pontos'
        )
        if uploaded_pontos:
            processar_arquivo_carregado(uploaded_pontos, tipo='pontos')

            # Botões de controle
            if st.button("▶️ Área Amostral"):
                st.session_state.modo_desenho = 'amostral'
                st.session_state.modo_insercao = None
                st.success("Modo desenho ativado: Área Amostral - Desenhe no mapa")
                st.rerun()

            if st.button("▶️ Área Total"):
                st.session_state.modo_desenho = 'total'
                st.session_state.modo_insercao = None
                st.success("Modo desenho ativado: Área Total - Desenhe no mapa")
                st.rerun()

            if st.button("✏️ Inserir pontos manualmente"):
                st.session_state.modo_insercao = 'manual'

            if st.button("📝 Inserir produtividade"):
                inserir_produtividade()

            if st.button("💾 Salvar pontos"):
                salvar_pontos()

            if st.button("💾 Exportar dados"):
                exportar_dados()

            if st.button("🗑️ Limpar Área"):
                st.session_state.gdf_poligono = None
                st.session_state.gdf_poligono_total = None
                st.session_state.gdf_pontos = None
                st.success("Áreas limpas!")

            # Dados da área amostral
            st.subheader("Dados da área amostral")
            st.session_state.densidade_plantas = st.number_input("Densidade (plantas/ha):", value=0.0)
            st.session_state.produtividade_media = st.number_input("Produtividade média última safra (sacas/ha):", value=0.0)

            if st.button("🔢 Gerar pontos automáticos (2/ha)"):
                if st.session_state.get('gdf_poligono') is not None:
                    gerar_pontos_automaticos()

            # Unidade de produtividade
            st.subheader("Produtividade")
            st.session_state.unidade_selecionada = st.selectbox("Unidade:", ['kg', 'latas', 'litros'])
    with col2:
        st.header("Mapa de visualização")
        mapa = create_map()
    # Atualize esta linha para capturar o retorno do st_folium em uma variável do session_state
        st.session_state.mapa_data = st_folium(mapa, width=800, height=600, key='mapa_principal')
    
    # Captura o desenho feito
    if st.session_state.get('mapa_data') and st.session_state.mapa_data.get('last_active_drawing'):
        geometry = st.session_state.mapa_data['last_active_drawing']['geometry']
        gdf = gpd.GeoDataFrame(geometry=[shape(geometry)], crs="EPSG:4326")
        
        if st.session_state.modo_desenho == 'amostral':
            st.session_state.gdf_poligono = gdf
            st.success("Área amostral definida!")
        elif st.session_state.modo_desenho == 'total':
            st.session_state.gdf_poligono_total = gdf
            st.success("Área total definida!")
        
        # Limpa o estado para evitar reprocessamento
        st.session_state.modo_desenho = None
        st.rerun()
        
        # Limpa o estado para evitar reprocessamento
        st.session_state.modo_desenho = None
        st.rerun()

if __name__ == "__main__":
    main()

def inserir_ponto_manual():   
    with st.form("Inserir Ponto Manual"):
        lat = st.number_input("Latitude:", value=-15.0)
        lon = st.number_input("Longitude:", value=-55.0)
        if st.form_submit_button("Adicionar Ponto"):
            adicionar_ponto(lat, lon, "manual")
            st.rerun()

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
    
    with st.expander("Editar dados de produtividade"):
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
        
        if st.button("Salvar alterações"):
            st.success("Dados de produtividade atualizados.")
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

# Função para salvar dados exportados no diretório temporário da nuvem Streamlit
def salvar_no_streamlit_cloud():
    if st.session_state.get("gdf_poligono") is None or \
       st.session_state.get("gdf_poligono_total") is None or \
       st.session_state.get("gdf_pontos") is None:
        st.warning("⚠️ Certifique-se de que todas as áreas e pontos foram definidos!")
        return

    if st.session_state.get("densidade_plantas") is None or \
       st.session_state.get("produtividade_media") is None:
        st.warning("⚠️ Parâmetros de densidade e produtividade não definidos!")
        return

    # Diretório temporário na nuvem
    temp_dir = "/tmp/streamlit_dados"
    os.makedirs(temp_dir, exist_ok=True)

    # Salvar os arquivos individualmente
    st.session_state.gdf_poligono.to_file(f"{temp_dir}/area_poligono.gpkg", driver="GPKG")
    st.session_state.gdf_poligono_total.to_file(f"{temp_dir}/area_total_poligono.gpkg", driver="GPKG")
    st.session_state.gdf_pontos.to_file(f"{temp_dir}/pontos_produtividade.gpkg", driver="GPKG")

    parametros = {
        'densidade_pes_ha': st.session_state.densidade_plantas,
        'produtividade_media_sacas_ha': st.session_state.produtividade_media
    }
    with open(f"{temp_dir}/parametros_area.json", "w") as f:
        json.dump(parametros, f)

    st.success("✅ Arquivos salvos temporariamente na nuvem do Streamlit!")
    st.info("➡️ Eles poderão ser carregados no próximo módulo do projeto.")

    # Retornar o caminho temporário para reuso (opcional)
    return temp_dir

# Se necessário, criar botão para executar
if st.button("☁️ Salvar dados na nuvem"):
    salvar_no_streamlit_cloud()
