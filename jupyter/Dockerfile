FROM python:3.11-slim

LABEL maintainer="MLOps Penguins"

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /work

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Puerto JupyterLab
EXPOSE 8888
