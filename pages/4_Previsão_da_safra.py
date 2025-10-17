import streamlit as st
st.set_page_config(page_title="Previsão da safra", layout="wide")
st.markdown("# Previsão da safra")

st.page_link("pages/4_1_Analise_de_correlacao.py", label="1. Análise de correlação")
st.page_link("pages/4_2_Treinamento_com_Machine_Learning.py", label="2. Treinamento com Machine Learning")
st.page_link("pages/4_3_Prev_produtividade.py", label="3. Previsão da produtividade")

