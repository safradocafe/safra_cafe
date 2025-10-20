# app.py
import streamlit as st

st.set_page_config(
    page_title="☕ SAFRA DO CAFÉ | Sistema para Cafeicultura de Precisão",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Safra do Café")
st.caption("Sistema avançado de geotecnologias para Cafeicultura de Precisão")

st.markdown(
    """
**Protótipo que integra amostras geolocalizadas de produtividade, imagens Sentinel-2 (MSI) via Google Earth Engine e Machine Learning para:**
- gerar mapas de variabilidade espacial da produtividade;  
- monitorar áreas de produção com índices espectrais (NDVI, NDRE, CCCI, GNDVI, NDMI, MSAVI2, NBR, TWI2, NDWI);  
- processar dados, analisar correlação e prever a safra.
"""
)

st.divider()
st.subheader("Navegação rápida")

# Links para as páginas (garanta que os arquivos existam exatamente com esses nomes em /pages)
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_Adicionar_informações.py", label="➕ Adicionar informações", icon="🗺️")
    st.page_link("pages/2_Mapa_de_produtividade.py", label="🗂️ Mapa de produtividade", icon="🧭")
    st.page_link("pages/3_Monitoramento.py", label="📡 Monitoramento", icon="📆")
with col2:
    st.page_link("pages/4_Processamento_de_dados.py", label="⚙️ Processamento de dados", icon="🧩")
    st.page_link("pages/5_Análise_de_correlação.py", label="📈 Análise de correlação", icon="📊")
    st.page_link("pages/6_Treinamento_com_Machine_Learning.py", label="🤖 Treinamento com Machine Learning", icon="🚀")
    st.page_link("pages/7_Previsão_da_safra.py", label="☕ Previsão da safra", icon="📅")

st.divider()
st.info("Dica: siga a ordem das abas para um fluxo completo, iniciando por **Adicionar informações** e finalizando em **Previsão da safra**.")
