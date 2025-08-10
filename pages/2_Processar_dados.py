import ee
import geemap
import pandas as pd
import geopandas as gpd
from datetime import datetime
import streamlit as st
import tempfile
import json
import os

# Configuração da página do Streamlit
st.set_page_config(layout="wide")
st.title("Processamento dos dados")
st.title("Seleção das imagens do sensor MSI/Sentinel-2A, cálculo dos índices espectrais e criação do banco de dados")
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Funções auxiliares
def verificar_resultados_salvos():
    """Verifica se existem resultados salvos na nuvem."""
    temp_dir = "/tmp/streamlit_dados"
    return os.path.exists(f"{temp_dir}/resultados_analise.gpkg")

def carregar_resultados_da_nuvem():
    """Carrega resultados previamente salvos na nuvem."""
    temp_dir = "/tmp/streamlit_dados"
    try:
        gdf_resultado = gpd.read_file(f"{temp_dir}/resultados_analise.gpkg")
        with open(f"{temp_dir}/parametros_analise.json", "r") as f:
            parametros_analise = json.load(f)
        return gdf_resultado, parametros_analise
    except Exception as e:
        st.error(f"Erro ao carregar resultados: {str(e)}")
        return None, None

def salvar_resultados_na_nuvem(gdf_resultado, parametros_analise):
    """Salva os resultados da análise na nuvem do Streamlit para uso posterior."""
    temp_dir = "/tmp/streamlit_dados"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Salvar GeoDataFrame com os resultados
        gdf_resultado.to_file(f"{temp_dir}/resultados_analise.gpkg", driver="GPKG")
        
        # Salvar parâmetros da análise como JSON
        with open(f"{temp_dir}/parametros_analise.json", "w") as f:
            json.dump(parametros_analise, f)
            
        # Salvar também como CSV (sem geometria)
        df_resultado = pd.DataFrame(gdf_resultado.drop(columns='geometry'))
        df_resultado.to_csv(f"{temp_dir}/resultados_analise.csv", index=False)
        
        st.success("✅ Resultados salvos na nuvem com sucesso!")
        return True
    except Exception as e:
        st.error(f"❌ Erro ao salvar resultados na nuvem: {str(e)}")
        return False

def carregar_arquivos_da_nuvem():
    """Carrega arquivos salvos na nuvem do Streamlit."""
    temp_dir = "/tmp/streamlit_dados"
    try:
        # Carregar parâmetros
        with open(f"{temp_dir}/parametros_area.json", "r") as f:
            parametros = json.load(f)
        
        # Carregar polígonos e pontos
        gdf_poligono = gpd.read_file(f"{temp_dir}/area_poligono.gpkg")
        gdf_poligono_total = gpd.read_file(f"{temp_dir}/area_total_poligono.gpkg")
        gdf_pontos = gpd.read_file(f"{temp_dir}/pontos_produtividade.gpkg")
        
        return gdf_poligono, gdf_poligono_total, gdf_pontos, parametros
    except Exception as e:
        st.error(f"Erro ao carregar arquivos da nuvem: {str(e)}")
        return None, None, None, None

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
    st.error(f"Erro ao inicializar o GEE: {str(e)}")
    st.stop()

# Barra lateral - Gerenciamento de resultados
st.sidebar.header("Gerenciamento de resultados")
if st.sidebar.button("↩️ Carregar resultados existentes"):
    gdf_resultado, parametros = carregar_resultados_da_nuvem()
    if gdf_resultado is not None:
        st.session_state['gdf_resultado'] = gdf_resultado
        st.experimental_rerun()

# Interface principal
st.sidebar.header("Configurações")
opcao_carregamento = st.sidebar.radio(
    "Fonte dos dados:",
    ["Carregar da nuvem (salvo pelo código 2)", "Fazer upload manual"]
)

if opcao_carregamento == "Carregar da nuvem (salvo pelo código 2)":
    gdf_poligono, gdf_poligono_total, gdf_pontos, parametros = carregar_arquivos_da_nuvem()
    
    if gdf_poligono is not None:
        st.sidebar.success("✅ Dados carregados da nuvem com sucesso!")
        st.sidebar.write(f"Densidade: {parametros['densidade_pes_ha']} plantas/ha")
        st.sidebar.write(f"Produtividade média: {parametros['produtividade_media_sacas_ha']} sacas/ha")
else:
    # Upload manual de arquivos
    uploaded_poligono = st.sidebar.file_uploader("Área Polígono (GPKG)", type=['gpkg'])
    uploaded_pontos = st.sidebar.file_uploader("Pontos de Produtividade (GPKG)", type=['gpkg'])
    
    if uploaded_poligono and uploaded_pontos:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmp_poligono:
            tmp_poligono.write(uploaded_poligono.getvalue())
            gdf_poligono = gpd.read_file(tmp_poligono.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmp_pontos:
            tmp_pontos.write(uploaded_pontos.getvalue())
            gdf_pontos = gpd.read_file(tmp_pontos.name)
        
        st.sidebar.success("✅ Arquivos carregados com sucesso!")

# Verificar se temos dados para processar
if 'gdf_poligono' not in locals() or gdf_poligono is None:
    st.warning("Por favor, carregue os dados primeiro.")
    st.stop()

# Mostrar informações básicas
with st.expander("📊 Informações dos dados carregados"):
    st.write("**Área do polígono:**")
    st.write(gdf_poligono)
    st.write("**Pontos de produtividade:**")
    st.write(gdf_pontos.head())

# Converter para objetos do Earth Engine
try:
    poligono = geemap.gdf_to_ee(gdf_poligono)
    pontos = geemap.gdf_to_ee(gdf_pontos)
except Exception as e:
    st.error(f"Erro ao converter para Earth Engine: {str(e)}")
    st.stop()

# Configuração da análise
st.sidebar.header("Parâmetros da análise")
data_inicio = st.sidebar.text_input("Data de início", "2023-08-01")
data_fim = st.sidebar.text_input("Data de fim", "2024-05-31")
indices_selecionados = st.sidebar.multiselect(
    "Índices espectrais para calcular",
    ['NDVI', 'NDRE', 'CCCI', 'GNDVI', 'NDMI', 'MSAVI2', 'NBR', 'TWI2', 'NDWI'],
    default=['NDVI', 'NDRE', 'CCCI']
)

# Botão para executar a análise
if st.sidebar.button("▶️ Executar análise"):
    with st.spinner("Processando dados..."):
        try:
            # Intervalo de datas
            data_inicio_ee = ee.Date(data_inicio)
            data_fim_ee = ee.Date(data_fim)

            # Calcular índices espectrais
            def calcular_indices(image):
                bands = {}
                if 'NDVI' in indices_selecionados:
                    bands['NDVI'] = image.normalizedDifference(['B8', 'B4'])
                if 'NDRE' in indices_selecionados:
                    bands['NDRE'] = image.normalizedDifference(['B8', 'B5'])
                if 'CCCI' in indices_selecionados and 'NDVI' in indices_selecionados and 'NDRE' in indices_selecionados:
                    bands['CCCI'] = bands['NDRE'].divide(bands['NDVI'])
                if 'GNDVI' in indices_selecionados:
                    bands['GNDVI'] = image.normalizedDifference(['B8', 'B3'])
                if 'NDMI' in indices_selecionados:
                    bands['NDMI'] = image.normalizedDifference(['B8', 'B11'])
                if 'MSAVI2' in indices_selecionados:
                    msavi2 = image.expression(
                        '(2 * NIR + 1 - sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - RED))) / 2', {
                            'NIR': image.select('B8'),
                            'RED': image.select('B4')
                        })
                    bands['MSAVI2'] = msavi2
                if 'NBR' in indices_selecionados:
                    bands['NBR'] = image.normalizedDifference(['B8', 'B12'])
                if 'TWI2' in indices_selecionados:
                    bands['TWI2'] = image.normalizedDifference(['B9', 'B8'])
                if 'NDWI' in indices_selecionados:
                    bands['NDWI'] = image.normalizedDifference(['B3', 'B8'])
                
                # Renomear bandas e adicionar à imagem
                for name in bands:
                    bands[name] = bands[name].rename(name)
                
                return image.addBands(list(bands.values()))

            # Filtrar coleção Sentinel-2
            colecao = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(poligono.geometry()) \
                .filterDate(data_inicio_ee, data_fim_ee) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
                .map(calcular_indices)

            # Validar imagens com dados
            def imagem_valida(img):
                count = img.select(0).reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=poligono,
                    scale=10,
                    maxPixels=1e9
                ).values().get(0)
                return img.set('valida', ee.Number(count).gt(0))

            colecao_com_valida = colecao.map(imagem_valida)
            imagens_validas = colecao_com_valida.filter(ee.Filter.eq('valida', 1))

            def add_formatted_date(img):
                date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
                return img.set('formatted_date', date_str)

            imagens_validas = imagens_validas.map(add_formatted_date)
            distinct_dates = imagens_validas.distinct(['formatted_date'])
            imagens_validas = ee.ImageCollection(distinct_dates)

            datas = imagens_validas.aggregate_array('formatted_date').getInfo()

            if datas:
                st.success(f"{len(datas)} imagens válidas encontradas:")
                for data in sorted(datas):
                    st.write(f"- {data}")
            else:
                st.warning("Nenhuma imagem válida encontrada com dados na área especificada.")
                st.stop()

            # Função para extrair estatísticas por ponto
            def extrair_estatisticas_ponto_imagem(imagem, feature_ponto, nomes_indices):
                buffer = feature_ponto.geometry().buffer(5)  # Buffer de 5 metros
                data_img = ee.Date(imagem.get('system:time_start')).format('yyyyMMdd')
                resultado_img = ee.Dictionary({})

                for indice in nomes_indices:
                    banda = imagem.select(indice)
                    reducao = banda.reduceRegion(
                        reducer=ee.Reducer.min().combine(
                            reducer2=ee.Reducer.mean(), sharedInputs=True
                        ).combine(
                            reducer2=ee.Reducer.max(), sharedInputs=True
                        ),
                        geometry=buffer,
                        scale=10,
                        maxPixels=1e8
                    )

                    valor_min = reducao.get(indice + '_min')
                    valor_mean = reducao.get(indice + '_mean')
                    valor_max = reducao.get(indice + '_max')

                    chave_min = ee.String(data_img).cat('_').cat(indice).cat('_min')
                    chave_mean = ee.String(data_img).cat('_').cat(indice).cat('_mean')
                    chave_max = ee.String(data_img).cat('_').cat(indice).cat('_max')

                    resultado_img = ee.Dictionary(ee.Algorithms.If(valor_min, resultado_img.set(chave_min, valor_min), resultado_img))
                    resultado_img = ee.Dictionary(ee.Algorithms.If(valor_mean, resultado_img.set(chave_mean, valor_mean), resultado_img))
                    resultado_img = ee.Dictionary(ee.Algorithms.If(valor_max, resultado_img.set(chave_max, valor_max), resultado_img))

                return ee.Feature(feature_ponto.geometry(), resultado_img)

            # Processar todos os pontos
            def processar_ponto(ponto, colecao_imagens, lista_indices):
                def extrair_para_cada_imagem(imagem):
                    return extrair_estatisticas_ponto_imagem(imagem, ponto, lista_indices)

                resultados_por_imagem = colecao_imagens.map(extrair_para_cada_imagem)

                def combinar_props(feature, acumulador):
                    return ee.Dictionary(acumulador).combine(ee.Feature(feature).toDictionary(), overwrite=True)

                propriedades_combinadas = ee.Dictionary(
                    resultados_por_imagem.iterate(combinar_props, ee.Dictionary({}))
                )

                return ee.Feature(ponto.geometry(), propriedades_combinadas)

            # Executar processamento
            pontos_com_indices = pontos.map(lambda pt: processar_ponto(pt, imagens_validas, indices_selecionados))

            # Converter para GeoDataFrame em memória
            gdf_resultado = geemap.ee_to_gdf(pontos_com_indices)

            # Adicionar coluna de produtividade observada
            if 'maduro_kg' in gdf_pontos.columns:
                gdf_resultado['maduro_kg'] = gdf_pontos['maduro_kg'].values

  # Mostrar resultados
            st.subheader("Resultados da análise")
            df_sem_geometria = gdf_resultado.drop(columns=['geometry'] if 'geometry' in gdf_resultado.columns else [])
            st.dataframe(df_sem_geometria)
            
            # Opção para download
            csv = df_sem_geometria.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar resultados como CSV",
                data=csv,
                file_name="indices_vegetacao.csv",
                mime="text/csv"
            )

            # ✅ BOTÃO DE SALVAMENTO DENTRO DO BLOCO DE PROCESSAMENTO
            parametros_analise = {
                "data_inicio": data_inicio,
                "data_fim": data_fim,
                "indices_calculados": indices_selecionados,
                "num_imagens_processadas": len(datas),
                "num_pontos_analisados": len(gdf_resultado)
            }
            
            if st.button("💾 Salvar Resultados na Nuvem"):
                if salvar_resultados_na_nuvem(gdf_resultado, parametros_analise):
                    st.success("Dados salvos com sucesso!")
                    st.session_state['resultados_salvos'] = True

        except Exception as e:
            st.error(f"Erro durante o processamento: {str(e)}")
