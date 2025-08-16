FROM python:3.10-slim

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

# Comando para iniciar o aplicativo Flask
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
