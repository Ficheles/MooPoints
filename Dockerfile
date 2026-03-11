FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Instalar dependências do sistema incluindo git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    git \
    git-lfs \
    lsof \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos os arquivos necessários
COPY src ./src
COPY models ./models
COPY data ./data
COPY app.py ./
COPY .gitattributes ./

# Garantir que arquivos LFS estão disponíveis
RUN if [ -f models/yolo/best.pt ]; then \
        echo "✓ best.pt encontrado ($(du -h models/yolo/best.pt | cut -f1))"; \
    else \
        echo "✗ best.pt NÃO encontrado!"; \
        exit 1; \
    fi

# Expor porta 7860 (padrão do HF Spaces)
EXPOSE 7860

CMD ["python", "app.py"]
