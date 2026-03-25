FROM python:3.11-slim

LABEL maintainer="MLOps Penguins"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

# --allow-root requerido porque el contenedor corre como root
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/work", \
     "--IdentityProvider.token=", \
     "--ServerApp.password="]