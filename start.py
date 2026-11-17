#!/usr/bin/env python3
"""
AestheticEngine — Lanceur Universel Cross-Platform (Windows 11 & Ubuntu/Linux).
==============================================================================
Ce script automatise à 100% le cycle de vie du projet :
  1. Vérification de l'environnement (Python, Bun/NPM, FFmpeg, Ollama)
  2. Création et synchronisation du Virtualenv Backend (.venv)
  3. Installation automatique des dépendances (FastAPI, Librosa, MoviePy, etc.)
  4. Configuration des fichiers .env pour l'environnement local
  5. Démarrage simultané du Backend FastAPI (port 8000) et du Frontend Vite (port 5173)
  6. Arrêt propre (Graceful Shutdown) des deux serveurs lors d'un Ctrl+C
"""

import os
import sys
import shutil
import signal
import subprocess
import threading
import time
import webbrowser
import urllib.request
import json
from pathlib import Path

# Couleurs ANSI pour le terminal
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

BANNER = (
    f"{CYAN}{BOLD}\n"
    r"   _   ___ ___ _____ _  _ ___ _____ ___ ___   ___ _  _  ___ ___ _  _ ___ " "\n"
    r"  /_\ | __/ __|_   _| || | __|_   _|_ _/ __| | __| \| |/ __|_ _| \| | __|" "\n"
    r" / _ \| _|\__ \ | | | __ | _|  | |  | | (__  | _|| .` | (_ || || .` | _| " "\n"
    r"/_/ \_\___|___/ |_| |_||_|___| |_| |___|___| |___|_|\_|\___|___|_|\_|___|" "\n"
    f"{RESET}   {GREEN}Montage Vidéo TikTok Musculation Automatisé — Full-Stack Runner{RESET}\n"
)

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
IS_WINDOWS = sys.platform.startswith("win")

def log_info(msg: str):
    print(f"{GREEN}[INFO]{RESET} {msg}")

def log_warn(msg: str):
    print(f"{YELLOW}[ATTENTION]{RESET} {msg}")

def log_error(msg: str):
    print(f"{RED}[ERREUR]{RESET} {msg}")

def log_step(msg: str):
    print(f"\n{CYAN}{BOLD}==> {msg}{RESET}")


def get_venv_paths():
    venv_dir = BACKEND_DIR / ".venv"
    if IS_WINDOWS:
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
        scripts_dir = venv_dir / "Scripts"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
        scripts_dir = venv_dir / "bin"
    return venv_dir, python_exe, pip_exe, scripts_dir


def check_python_version():
    log_step("Vérification de la version Python")
    v = sys.version_info
    log_info(f"Python détecté : {v.major}.{v.minor}.{v.micro} sur {sys.platform}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        log_error("Python 3.10 ou supérieur est requis.")
        sys.exit(1)


def find_frontend_pm():
    """Détecte le gestionnaire de paquets frontend (bun > pnpm > npm)."""
    for pm in ["bun", "pnpm", "npm"]:
        path = shutil.which(pm)
        if path:
            return pm
    return None


def ensure_env_files():
    log_step("Vérification des variables d'environnement (.env)")
    env_content = (
        "# Configuration locale générée automatiquement\n"
        "OLLAMA_BASE_URL=http://localhost:11434\n"
        "OLLAMA_VISION_MODEL=llava:latest\n"
        "MIN_EFFORT_SCORE=7.0\n"
        "STORAGE_DIR=storage\n"
        "LUT_DIR=assets/luts\n"
        "VITE_API_BASE_URL=http://localhost:8000\n"
        "VITE_WS_BASE_URL=ws://localhost:8000\n"
    )

    # Root .env
    root_env = ROOT_DIR / ".env"
    if not root_env.exists():
        root_env.write_text(env_content, encoding="utf-8")
        log_info("Fichier .env racine créé.")

    # Backend .env
    backend_env = BACKEND_DIR / ".env"
    if not backend_env.exists():
        backend_env.write_text(env_content, encoding="utf-8")
        log_info("Fichier backend/.env créé.")

    # Frontend .env
    frontend_env = FRONTEND_DIR / ".env"
    if not frontend_env.exists():
        frontend_env.write_text(
            "VITE_API_BASE_URL=http://localhost:8000\nVITE_WS_BASE_URL=ws://localhost:8000\n",
            encoding="utf-8",
        )
        log_info("Fichier frontend/.env créé.")


def check_ollama():
    log_step("Vérification d'Ollama")
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "AestheticEngine"})
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            log_info(f"Ollama est EN LIGNE. Modèles disponibles : {', '.join(models) if models else 'aucun'}")
            has_llava = any("llava" in m for m in models)
            if not has_llava:
                log_warn("Le modèle de vision LLaVA n'est pas encore téléchargé.")
                log_warn("Pour activer le scoring des poses par IA : exécutez 'ollama pull llava:latest'")
            else:
                log_info("Modèle LLaVA détecté pour l'analyse visuelle par IA.")
    except Exception:
        log_warn("Ollama n'est pas accessible sur http://localhost:11434.")
        log_warn("L'application fonctionnera, mais pour le scoring LLaVA, pensez à lancer Ollama ('ollama serve').")


def setup_backend():
    log_step("Configuration du Backend Python")
    venv_dir, python_exe, pip_exe, scripts_dir = get_venv_paths()

    if not python_exe.exists():
        log_info(f"Création de l'environnement virtuel dans {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    # Vérification des dépendances critiques
    check_code = "import fastapi, uvicorn, librosa, moviepy, cv2; print('OK')"
    check_res = subprocess.run([str(python_exe), "-c", check_code], capture_output=True, text=True)

    if check_res.returncode != 0:
        log_info("Installation des dépendances backend (FastAPI, Librosa, MoviePy, etc.)...")
        req_file = BACKEND_DIR / "requirements.txt"
        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(pip_exe), "install", "-r", str(req_file)], check=True)
        log_info("Dépendances backend installées avec succès.")
    else:
        log_info("Dépendances backend déjà prêtes.")

    # Assurer la présence de ffmpeg
    if not shutil.which("ffmpeg"):
        ffmpeg_script = (
            "import imageio_ffmpeg, shutil, pathlib, sys; "
            "src = imageio_ffmpeg.get_ffmpeg_exe(); "
            "dst_dir = pathlib.Path(sys.executable).parent; "
            "dst = dst_dir / ('ffmpeg.exe' if sys.platform.startswith('win') else 'ffmpeg'); "
            "shutil.copyfile(src, dst) if not dst.exists() else None; "
            "print(str(dst))"
        )
        try:
            res = subprocess.run([str(python_exe), "-c", ffmpeg_script], capture_output=True, text=True, check=True)
            log_info(f"FFmpeg configuré : {res.stdout.strip()}")
        except Exception as e:
            log_warn(f"Impossible de copier FFmpeg depuis imageio_ffmpeg : {e}")

    # Injection du dossier scripts dans le PATH pour que 'ffmpeg' soit reconnu
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")


def setup_frontend(pm: str):
    log_step(f"Configuration du Frontend ({pm})")
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        log_info(f"Installation des dépendances frontend via '{pm} install'...")
        subprocess.run([pm, "install"], cwd=str(FRONTEND_DIR), check=True)
        log_info("Dépendances frontend installées avec succès.")
    else:
        log_info("Dépendances frontend déjà installées.")


def stream_output(process: subprocess.Popen, prefix: str, color: str):
    try:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            print(f"{color}{prefix}{RESET} {line.rstrip()}")
    except Exception:
        pass


def main():
    print(BANNER)
    check_python_version()

    pm = find_frontend_pm()
    if not pm:
        log_error("Aucun gestionnaire de paquets frontend (bun, pnpm, npm) n'a été trouvé.")
        log_error("Sur Windows ou Ubuntu : installez Bun (https://bun.sh) ou NodeJS (https://nodejs.org).")
        sys.exit(1)
    log_info(f"Gestionnaire frontend détecté : {pm}")

    ensure_env_files()
    check_ollama()
    setup_backend()
    setup_frontend(pm)

    log_step("Démarrage des serveurs AestheticEngine")
    venv_dir, python_exe, pip_exe, scripts_dir = get_venv_paths()

    # Démarrage Backend FastAPI
    log_info(f"Lancement Backend sur {BOLD}http://localhost:8000{RESET}...")
    backend_cmd = [str(python_exe), "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Démarrage Frontend Vite
    log_info(f"Lancement Frontend sur {BOLD}http://localhost:5173{RESET}...")
    frontend_cmd = [pm, "run", "dev", "--host", "0.0.0.0", "--port", "5173"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Threads pour afficher les logs en direct
    t_backend = threading.Thread(target=stream_output, args=(backend_proc, "[BACKEND]", CYAN), daemon=True)
    t_frontend = threading.Thread(target=stream_output, args=(frontend_proc, "[FRONTEND]", MAGENTA), daemon=True)
    t_backend.start()
    t_frontend.start()

    print(f"\n{GREEN}{BOLD}✨ AestheticEngine est opérationnel !{RESET}")
    print(f"👉 Interface Web : {BOLD}http://localhost:5173{RESET}")
    print(f"👉 API Swagger   : {BOLD}http://localhost:8000/docs{RESET}")
    print(f"{YELLOW}Appuyez sur Ctrl+C pour arrêter l'application.{RESET}\n")

    # Ouvrir le navigateur après 2 secondes
    def open_browser():
        time.sleep(2.0)
        try:
            webbrowser.open("http://localhost:5173")
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    def signal_handler(sig, frame):
        print(f"\n{YELLOW}Arrêt en cours de tous les services...{RESET}")
        try:
            backend_proc.terminate()
            frontend_proc.terminate()
            backend_proc.wait(timeout=5)
            frontend_proc.wait(timeout=5)
        except Exception:
            backend_proc.kill()
            frontend_proc.kill()
        print(f"{GREEN}Services arrêtés avec succès. À bientôt !{RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            b_code = backend_proc.poll()
            f_code = frontend_proc.poll()
            if b_code is not None:
                log_error(f"Le backend s'est arrêté avec le code {b_code}")
                break
            if f_code is not None:
                log_error(f"Le frontend s'est arrêté avec le code {f_code}")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
