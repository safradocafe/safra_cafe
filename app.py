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
**Este é um protótipo que integra amostras geolocalizadas de produtividade do café, imagens do sensor MSI/Sentinel-2AB via Google Earth Engine (GEE) e Machine Learning para:**
- gerar mapas de variabilidade espacial da produtividade;  
- monitorar áreas de produção com índices espectrais (NDVI, NDRE, CCCI, GNDVI, NDMI, MSAVI2, NBR, TWI2 e NDWI);  
- processar dados, analisar a correlação da produtividade com os índices espectrais, treinar com ML e prever a própxima safra.

O projeto é resultado de pesquisas no Mestrado Profissional em Agricultura de Precisão da Universidade Federal de Santa Maria (UFSM).

**Autoria:** Rozymario Bittencourt Fagundes  
**Contatos:** rozymariofagundes@gmail.com | +55 77 98849 1600

**Leia a dissertação:**

Sensoriamento remoto e redes neurais na estimativa da produtividade do café arábica na Bahia

[https://repositorio.ufsm.br/handle/1/35196](https://repositorio.ufsm.br/handle/1/35196)

Conheça outras pesquisas do autor sobre **Cafeicultura de Precisão**:

**Coffee Science:**

Diagnosis about the perspectives of precision applications of coffee growing technologies in municipalities of Bahia, Brazil:

[https://coffeescience.ufla.br/index.php/Coffeescience/article/view/2034](https://coffeescience.ufla.br/index.php/Coffeescience/article/view/2034)

**Geomatics/MPDI:**

Analysis of Resampling Methods for the Red Edge Band of MSI/Sentinel-2A for Coffee Cultivation Monitoring:

[https://www.mdpi.com/2673-7418/5/2/19](https://www.mdpi.com/2673-7418/5/2/19)
 
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
st.markdown(
    """
**A base de dados para funcionamento deste código são arquivos da área de produção e amostras da produtividade de uma safra.** 

Caso não tenha esses dados para teste do sistema, você pode baixá-los neste link: [https://drive.google.com/drive/folders/1h5hSdq3PXXxKoX8NEpqH1qGoT59RjoG6?usp=drive_link](https://drive.google.com/drive/folders/1h5hSdq3PXXxKoX8NEpqH1qGoT59RjoG6?usp=drive_link)

**Observação:** esses dados são da pesquisa no mestrado e devem ser utilizados conforme a metodologia proposta. Eles são relativos à safra de café de 2024. A seleção de imagens de satélite para processamento dos dados deve ter como início 01/08/2023 e fim 31/05/2024. Caso utilize dados da sua fazenda, os arquivos devem estar no formato .gpkg. Você pode nomeá-los como quiser. É possível também inserir os dados de forma manual no sistema.

**Dica:** siga a ordem das abas para um fluxo completo, iniciando por **Adicionar informações** e finalizando em **Previsão da safra**.
"""
)
