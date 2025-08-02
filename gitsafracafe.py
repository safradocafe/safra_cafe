import ee
import json
import streamlit as st
import geemap  # Para visualização de mapas

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
