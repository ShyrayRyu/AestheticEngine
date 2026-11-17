#!/usr/bin/env bash
# =============================================================================
# AestheticEngine — Lanceur Ubuntu / Debian / Linux / WSL
# =============================================================================

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${CYAN}${BOLD}"
echo "   _   ___ ___ _____ _  _ ___ _____ ___ ___   ___ _  _  ___ ___ _  _ ___ "
echo "  /_\ | __/ __|_   _| || | __|_   _|_ _/ __| | __| \| |/ __|_ _| \| | __|"
echo " / _ \| _|\__ \ | | | __ | _|  | |  | | (__  | _|| .` | (_ || || .` | _| "
echo "/_/ \_\___|___/ |_| |_||_|___| |_| |___|___| |___|_|\_|\___|___|_|\_|___|"
echo -e "${NC}   ${GREEN}Montage Vidéo TikTok Musculation Automatisé — Linux Runner${NC}\n"

# 1. Vérification de Python 3
echo -e "${CYAN}==> Vérification de Python 3...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[ERREUR] Python 3 n'est pas installé.${NC}"
    echo "Installez-le avec : sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    exit 1
fi
echo -e "${GREEN}[INFO] Python détecté : $($PYTHON_CMD --version)${NC}"

# 2. Vérification du gestionnaire de paquets Frontend (bun > pnpm > npm)
echo -e "${CYAN}==> Vérification du gestionnaire de paquets Frontend...${NC}"
if command -v bun &>/dev/null; then
    PKG_MGR="bun"
elif command -v pnpm &>/dev/null; then
    PKG_MGR="pnpm"
elif command -v npm &>/dev/null; then
    PKG_MGR="npm"
else
    echo -e "${RED}[ERREUR] Aucun gestionnaire JS (bun, pnpm, npm) n'est installé.${NC}"
    echo "Recommandé (ultra-rapide) : curl -fsSL https://bun.sh/install | bash"
    echo "Ou via APT : sudo apt install -y nodejs npm"
    exit 1
fi
echo -e "${GREEN}[INFO] Gestionnaire frontend : $PKG_MGR ($($PKG_MGR --version))${NC}"

# 3. Vérification de FFmpeg
echo -e "${CYAN}==> Vérification de FFmpeg...${NC}"
if command -v ffmpeg &>/dev/null; then
    echo -e "${GREEN}[INFO] FFmpeg système détecté : $(ffmpeg -version | head -n 1)${NC}"
else
    echo -e "${YELLOW}[ATTENTION] FFmpeg n'est pas installé sur le système.${NC}"
    echo "Conseil Ubuntu/Debian : sudo apt install -y ffmpeg"
fi

# 4. Vérification d'Ollama
echo -e "${CYAN}==> Vérification d'Ollama...${NC}"
if curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    OLLAMA_MODELS=$(curl -s http://127.0.0.1:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')
    echo -e "${GREEN}[INFO] Ollama est en ligne. Modèles : ${OLLAMA_MODELS:-aucun}${NC}"
    if echo "$OLLAMA_MODELS" | grep -q "llava"; then
        echo -e "${GREEN}[INFO] Modèle LLaVA détecté pour l'IA Vision.${NC}"
    else
        echo -e "${YELLOW}[ATTENTION] Le modèle 'llava:latest' n'est pas encore téléchargé.${NC}"
        echo "Pour l'analyse des poses : ollama pull llava:latest"
    fi
else
    echo -e "${YELLOW}[ATTENTION] Ollama n'est pas accessible sur http://localhost:11434.${NC}"
    echo "(Optionnel pour le scoring IA. Démarrez-le avec : ollama serve)"
fi

# 5. Fichiers .env
if [ ! -f ".env" ]; then
    cat <<EOF > .env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VISION_MODEL=llava:latest
MIN_EFFORT_SCORE=7.0
STORAGE_DIR=storage
LUT_DIR=assets/luts
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
EOF
    echo -e "${GREEN}[INFO] Fichier .env créé.${NC}"
fi

cp -n .env backend/.env 2>/dev/null || true
if [ ! -f "frontend/.env" ]; then
    echo -e "VITE_API_BASE_URL=http://localhost:8000\nVITE_WS_BASE_URL=ws://localhost:8000" > frontend/.env
fi

# 6. Environnement virtuel Backend
echo -e "\n${CYAN}==> Préparation de l'environnement virtuel Backend...${NC}"
VENV_DIR="backend/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}[INFO] Création du venv Python dans $VENV_DIR...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR" || {
        echo -e "${RED}[ERREUR] Échec de création du venv. Exécutez : sudo apt install -y python3-venv${NC}"
        exit 1
    }
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Vérification des paquets Python
if ! $VENV_PY -c "import fastapi, librosa, moviepy" >/dev/null 2>&1; then
    echo -e "${GREEN}[INFO] Installation des dépendances Backend (FastAPI, Librosa, MoviePy)...${NC}"
    $VENV_PIP install --upgrade pip
    $VENV_PIP install -r backend/requirements.txt
fi

# 7. Dépendances Frontend
echo -e "\n${CYAN}==> Préparation des dépendances Frontend...${NC}"
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${GREEN}[INFO] Installation des modules npm avec $PKG_MGR...${NC}"
    (cd frontend && $PKG_MGR install)
fi

# 8. Lancement des serveurs
echo -e "\n${GREEN}${BOLD}🚀 Démarrage d'AestheticEngine...${NC}"
echo -e "👉 Interface Web : ${BOLD}http://localhost:5173${NC}"
echo -e "👉 API Swagger   : ${BOLD}http://localhost:8000/docs${NC}"
echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter les serveurs.${NC}\n"

# Démarrage Backend
(cd backend && .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

# Démarrage Frontend
(cd frontend && $PKG_MGR run dev --host 0.0.0.0 --port 5173) &
FRONTEND_PID=$!

# Nettoyage à l'arrêt
cleanup() {
    echo -e "\n${YELLOW}Arrêt des serveurs AestheticEngine (PIDs: $BACKEND_PID, $FRONTEND_PID)...${NC}"
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
    echo -e "${GREEN}Application arrêtée proprement. À bientôt !${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Attente de la fin des processus
wait
