FROM python:3.12-slim

# libgomp: OpenMP usado pelo LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# gera a carteira sintética e roda o backtest; os relatórios ficam em /app/relatorios
CMD ["sh", "-c", "python dados/gerar_carteira.py && python -m src.treinar"]
