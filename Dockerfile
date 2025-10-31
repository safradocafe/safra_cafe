FROM python:3.10-slim

# Sistema (GDAL/GEOS/PROJ) — mantém o que você já usa
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Boas práticas e PORT padrão
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj

WORKDIR /app

# Melhor cache: copia requirements antes do código
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Agora o código
COPY . /app

# Exponha a porta (só documentação de build)
EXPOSE 8080

# IMPORTANTE: use a $PORT do Cloud Run (não fixar 8080)
# Usamos 'bash -lc' para expandir ${PORT}. Mantém Streamlit headless.
CMD ["bash","-lc","streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]

