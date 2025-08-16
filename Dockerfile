FROM python:3.10-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia os requirements primeiro para aproveitar cache do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto da aplicação
COPY . .

# Porta que o Cloud Run espera
EXPOSE 8080

# Comando para rodar o Streamlit no Cloud Run
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
