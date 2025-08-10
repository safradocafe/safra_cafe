import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Safra do Café",
    layout="wide"
)

st.title("☕ Safra do Café")
st.markdown("""
Sistema avançado de previsão da produtividade do café com imagens de satélite (MSI/Sentinel-2A) e algoritmos de Machine Learning.
""")
st.markdown("## *Acesse o sistema pelas barras laterais*")  # Corrigido: st.markdown em vez de st.title

# Carregando as imagens
img1 = Image.open(r"D:\Arquivos_R\sr\sr_cafe\Santa_Vera\teste app\indices_capa.jpeg")
img2 = Image.open(r"D:\Arquivos_R\sr\sr_cafe\Santa_Vera\teste app\img_capa_prod.jpeg")

# Criando duas colunas para as imagens
col1, col2 = st.columns(2)

with col1:
    st.image(img1, caption="Combinamos índices espectrais e Machine Learning para...", use_column_width=True)

with col2:
    st.image(img2, caption="... gerar o seu mapa de produtividade", use_column_width=True)

