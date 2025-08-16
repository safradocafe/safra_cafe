from flask import Flask, render_template_string
import os

app = Flask(__name__)

# Rota dinâmica que carrega o conteúdo do arquivo correspondente em pages/
@app.route('/<pagina>')
def carregar_pagina(pagina):
    modulo = __import__(f"pages.{pagina}", fromlist=["conteudo"])
    dados = modulo.conteudo()  # Cada script deve ter uma função 'conteudo()'
    return render_template_string(dados["html"], **dados["contexto"])

@app.route('/')
def home():
    return carregar_pagina("home")  # Assume que pages/home.py existe

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
