import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Safra do café",
    layout="wide"
)

st.title("☕ Safra do café")
st.markdown("""
### 1️⃣ Área Amostral
- **Opção 1:** Faça upload de arquivo `.gpkg` com **polígono da área**.
- **Opção 2:** Desenhe diretamente no mapa:
    1. Clique em **"Área amostral"**.
    2. Clique no ícone de **pentágono** no mapa.
    3. Desenhe a áera total seguindo o mesmo procedimento
    4. Para reiniciar o desenho, clique em **"Apagar"**.
""")

st.markdown("""
### 2️⃣ Dados de Produtividade
- **Opção 1:** Faça upload de arquivo `.gpkg` com **pontos de produtividade** (2 pontos/ha).
- **Opção 2:** Insira manualmente no mapa.
- Caso **não tenha** a grade amostral de pontos, clique em **"Gerar pontos automaticamente"**.
""")

st.warning("""
⚠️ **Atenção:**  
O sistema funciona apenas com **2 pontos por hectare**, valor mínimo recomendado por pesquisas científicas para a Cafeicultura de Precisão.
""")

st.info("""
ℹ️ **Observação:**  
Os valores de produtividade podem ser inseridos em **Latas**, **Litros** ou **Quilos (Kg)**.  
Se forem inseridos em *Latas* ou *Litros*, o sistema converte automaticamente para **Kg**, conforme a literatura científica.
""")

st.markdown("""
### 3️⃣ Finalizar
Após inserir **todos os dados**, clique em **"Salvar dados"**.
""")
