# pages/8_Gerenciar_acessos.py
import streamlit as st
import uuid
import datetime
import json
import os

TOKEN_FILE = "tokens.json"
URL_BASE_APP = "https://safradocafe-959156646826.europe-west1.run.app"

def carregar_tokens():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    return {}

def salvar_tokens(tokens):
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=4, ensure_ascii=False)

st.set_page_config(
    page_title="Admin | Safra do Café",
    page_icon="🔑",
    layout="wide"
)

st.title("🔑 Gerenciar acessos de 48h")

# --- Proteção simples com senha ---
senha = st.text_input("Senha de administrador", type="password")
SENHA_CORRETA = "cafeprecisao2025"  # TODO: troque por outra e guarde em st.secrets depois

if senha != SENHA_CORRETA:
    st.warning("Informe a senha de administrador para continuar.")
    st.stop()

st.success("Acesso de administrador liberado.")

tokens = carregar_tokens()

st.subheader("Gerar novo link de acesso de 48h")
email = st.text_input("E-mail ou identificação do cliente (opcional)")

if st.button("Gerar link de acesso"):
    token = uuid.uuid4().hex
    expira_em = datetime.datetime.utcnow() + datetime.timedelta(hours=48)

    tokens[token] = {
        "email": email,
        "criado_em": datetime.datetime.utcnow().isoformat(),
        "expira_em": expira_em.isoformat(),
        "ativo": True
    }
    salvar_tokens(tokens)

    link = f"{URL_BASE_APP}/?token={token}"
    st.success("Link gerado com sucesso! Envie este link para o cliente:")
    st.code(link, language="text")

st.divider()
st.subheader("Tokens existentes")

if tokens:
    linhas = []
    agora = datetime.datetime.utcnow()
    for t, dados in tokens.items():
        expira_em = datetime.datetime.fromisoformat(dados["expira_em"])
        expirado = agora > expira_em
        linhas.append({
            "token": t,
            "email": dados.get("email", ""),
            "criado_em": dados["criado_em"],
            "expira_em": dados["expira_em"],
            "expirado": "Sim" if expirado else "Não"
        })
    st.dataframe(linhas)
else:
    st.info("Nenhum token gerado ainda.")
