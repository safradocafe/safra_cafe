# app.py
import streamlit as st

st.set_page_config(
    page_title="☕ SAFRA DO CAFÉ - Sistema para Cafeicultura de Precisão",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Safra do Café")
st.caption("Sistema avançado de geotecnologias para Cafeicultura de Precisão")

st.markdown(
    """
    Este protótipo integra **amostras geolocalizadas de produtividade**, 
    **imagens Sentinel-2 (MSI)** via Google Earth Engine e **Machine Learning** para:
    - Mapa de variabilidade espacial da produtividade  
    - Monitoramento por índices espectrais (NDVI, NDRE, CCCI, GNDVI, NDMI, MSAVI2, NBR, TWI2, NDWI)  
    - Análise temporal e previsão de safra
    """
)

st.divider()
st.subheader("Navegação rápida")

# Requer Streamlit >= 1.25 (você está em 1.36.0, OK)
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_Adicionar_informações.py", label="➕ Adicionar informações", icon="🗺️")
    st.page_link("pages/2_Processar_dados.py", label="⚙️ Processar dados", icon="🧩")
with col2:
    st.page_link("pages/3_Análise_de_correlação.py", label="📈 Análise de correlação", icon="📊")
    st.page_link("pages/4_Treinamento_com_Machine_Learning.py", label="🤖 Treinamento com ML", icon="🚀")

st.divider()
st.info("Dica: utilize as páginas na ordem para um fluxo completo, começando por **Adicionar informações**.")
