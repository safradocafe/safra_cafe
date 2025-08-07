import json
import streamlit as st
import geemap
import ee
import time
import random
import string
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from io import BytesIO
import base64
import os
import folium
from streamlit_folium import st_folium

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

# ✅ Inicialização do GEE
try:
    if "GEE_CREDENTIALS" not in st.secrets:
        st.error("❌ Credenciais do GEE não encontradas em secrets.toml!")
    else:
        credentials_dict = dict(st.secrets["GEE_CREDENTIALS"])
        credentials_json = json.dumps(credentials_dict)
        credentials = ee.ServiceAccountCredentials(
            email=credentials_dict["client_email"],
            key_data=credentials_json
        )
        ee.Initialize(credentials)
except Exception as e:
    st.error(f"🚨 Erro ao inicializar o GEE: {str(e)}")

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
    
    # Adiciona a camada de satélite (Esri World Imagery)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satélite',
        overlay=False,
        control=True
    ).add_to(m)

    # Adiciona controle de desenho (para áreas e pontos)
    draw_options = {
        'polyline': False,
        'rectangle': False,
        'circle': False,
        'circlemarker': False,
        'marker': st.session_state.get('modo_desenho') == 'amostral'  # Habilita marcadores apenas no modo área amostral
    }
    draw = folium.plugins.Draw(
        draw_options=draw_options,
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
                color="green" if row['coletado'] else "red",
                fill=True,
                fill_color="green" if row['coletado'] else "red",
                fill_opacity=0.7,
                popup=f"Ponto: {row['Code']}<br>Produtividade: {row['valor']} {row['unidade']}"
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

# ✅ Função para processar arquivo GPKG
def processar_arquivo_carregado(uploaded_file):
    try:
        temp_file = f"./temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        gdf = gpd.read_file(temp_file)
        os.remove(temp_file)

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

    # Garante que maduro_kg está calculado
    st.session_state.gdf_pontos['maduro_kg'] = st.session_state.gdf_pontos.apply(
        lambda row: converter_para_kg(row['valor'], row['unidade']),
        axis=1
    )
    st.success("✅ Dados dos pontos preparados para exportação!")

def exportar_dados():
    """Exporta dados no formato ZIP (equivalente ao exportar_btn)"""
    if st.session_state.gdf_poligono is None:
        st.warning("⚠️ Nenhuma área desenhada para exportar")
        return

    import zipfile
    from io import BytesIO

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
    st.title("SAFRA DO CAFÉ")
    st.subheader("Sistema avançado de previsão da produtividade do café com imagens de satélite (sensor MSI/Sentintel-2A) e algoritmos de Machine Learning")

    # Upload de arquivo apenas GPKG
    uploaded_file = st.file_uploader("Carregar arquivo (.gpkg)", type=['gpkg'])
    if uploaded_file:
        processar_arquivo_carregado(uploaded_file)
    if st.session_state.get('modo_insercao') == 'manual':
        inserir_ponto_manual()
        return
    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("Controles")

        if st.button("▶️ Área Amostral"):
            st.session_state.modo_desenho = 'amostral'
            st.success("Modo desenho ativado: Área Amostral")     
        if st.button("✏️ Inserir pontos manualmente"):
        # Implementar lógica de inserção manual (similar ao código 1)
            st.session_state.modo_insercao = 'manual'
        if st.button("📝 Inserir produtividade"):
            inserir_produtividade()  # Função já existente
        if st.button("💾 Salvar pontos"):
            salvar_pontos()  # Função adicionada acima
        if st.button("💾 Exportar dados"):
            exportar_dados()  # Função adicionada acima
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

        st.subheader("Produtividade")
        st.session_state.unidade_selecionada = st.selectbox("Unidade:", ['kg', 'latas', 'litros'])

    with col2:
        st.header("Mapa de visualização")
        mapa = create_map()
        st_folium(mapa, width=800, height=600)

if __name__ == "__main__":
    main()

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

def inserir_ponto_manual():
    """Implementa a lógica do btn_inserir_manual (coordenadas via input)"""
    with st.form("Inserir Ponto Manual"):
        lat = st.number_input("Latitude:", value=-15.0)
        lon = st.number_input("Longitude:", value=-55.0)
        if st.form_submit_button("Adicionar Ponto"):
            adicionar_ponto(lat, lon, "manual")
            st.rerun()
