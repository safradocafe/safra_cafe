# Usa uma imagem base Python oficial
FROM python:3.9

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de requisitos para o contêiner
COPY requirements.txt ./

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da sua aplicação
COPY . .

# Expõe a porta que o Streamlit usará
EXPOSE 8501

# Comando para rodar a aplicação Streamlit
CMD ["streamlit", "run", "home.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
