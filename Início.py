import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Safra do Café",
    layout="wide"
)

st.title("☕ Safra do Café")
st.markdown("""
Sistema avançado de previsão da produtividade do café com imagens de satélite (sensor MSI/Sentinel-2A) e algoritmos de Machine Learning**.
st.title(##Navegue pelas páginas laterais##")
""")

# Carregando as imagens
img1 = Image.open(r"D:\Arquivos_R\sr\sr_cafe\Santa_Vera\teste app\indices_capa.jpeg")
img2 = Image.open(r"D:\Arquivos_R\sr\sr_cafe\Santa_Vera\teste app\img_capa_prod.jpeg")

# Criando duas colunas para as imagens
col1, col2 = st.columns(2)

st.markdown("""
<p style='text-align: center; font-size: 16px; margin-top: 10px;'>
    Combinamos índices espectrais e Machine Learning para gerar o seu mapa de produtividade
</p>
""", unsafe_allow_html=True)
