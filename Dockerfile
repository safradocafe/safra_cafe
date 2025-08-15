# Usa uma imagem base Python leve
FROM python:3.9-slim

# Instala dependências de sistema necessárias para pacotes GIS
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de requisitos
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Expõe a porta padrão do Cloud Run
EXPOSE 8080

# Inicia o Streamlit usando a variável $PORT do Cloud Run
CMD streamlit run home.py --server.port=${PORT} --server.address=0.0.0.0
