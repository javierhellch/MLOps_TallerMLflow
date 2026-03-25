FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

# ENTRYPOINT con lista JSON garantiza que los argumentos
# lleguen exactamente como se definen, sin interpretación de shell
ENTRYPOINT ["mlflow", "server", \
    "--backend-store-uri", "postgresql://mlops:mlops_secret@postgres:5432/mlops_db", \
    "--default-artifact-root", "s3://mlflows3/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5001", \
    "--serve-artifacts"]