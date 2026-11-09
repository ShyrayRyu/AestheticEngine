# =============================================================================
# Dockerfile Backend - AestheticEngine (Python 3.11 + FFmpeg + Librosa + MoviePy)
# =============================================================================
FROM python:3.11-slim-bookworm

# Variables d'environnement Python pour éviter les fichiers .pyc et bufferiser stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Installation des dépendances système indispensables :
# - ffmpeg       : Décodage/encodage vidéo, speed ramping et application de LUT 3D (.cube)
# - libsndfile1  : Lecture des formats audio par Librosa / SoundFile
# - libgl1/glib2 : Bibliothèques requises par OpenCV / MoviePy en environnement headless
# - curl         : Utilisé par le healthcheck Docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python en tirant parti du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source et des assets (LUTs)
COPY . .

# Création des répertoires de stockage s'ils n'existent pas
RUN mkdir -p /app/storage/uploads /app/storage/temp /app/storage/exports /app/assets/luts

EXPOSE 8000

# Lancement du serveur ASGI Uvicorn avec rechargement à chaud
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
