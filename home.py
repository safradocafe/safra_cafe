import streamlit as st
from PIL import Image
import os

port = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Safra do café", layout="wide")
st.set_page_config(
    page_title="Safra do café",
    layout="wide"
)

st.title("☕ Safra do café - Sistema avançado de previsão da produtividade")

st.markdown("""
Este é um protótipo de **geotecnologia para previsão da produtividade do café** com o uso de imagens de 
satélite (sensor MSI/Sentinel-2A) e algoritmos de Machine Learning, sendo este trabalho resultado de 
pesquisa acadêmica de Mestrado Profissional em Agricultura de Precisão pelo Colégio Politécnico da 
Universidade Federal de Santa Maria (UFSM), de autoria de **Rozymario Bittencourt Fagundes (MSc)**. O sistema utiliza técnicas
de Cafeicultura de Precisão para gerar mapas da variabilidade espacial da produtividade e identificar zonas de manejo. 
""")

st.markdown("""
**Para executar o sistema:**  
Navegue pelas páginas laterais (menu à esquerda) e siga as instruções em cada seção.
""")

st.info("""
ℹ️ **Sobre o sistema:**  
O protótipo utiliza técnicas de Agricultura de Precisão, processamento de imagens de satélite e algoritmos de Machine Learning para 
estimar a produtividade em lavouras de café, com validação científica conforme metodologia desenvolvida na pesquisa.
""")
