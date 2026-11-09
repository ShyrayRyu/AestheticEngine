# AestheticEngine — Montage Vidéo TikTok Musculation Automatisé ("Dark Aesthetic")

**AestheticEngine** est une application web locale Full-Stack permettant d'automatiser le montage de vidéos TikTok de musculation dans le style **"Dark Aesthetic"** (sélection des meilleures répétitions par IA Vision locale, synchronisation millimétrée sur les **Beat Drops**, **Speed Ramping** dynamique, étalonnage **LUT 3D `.cube`** via FFmpeg et export **XML EDL** pour Adobe Premiere Pro / DaVinci Resolve).

---

## Stack Technique

- **Frontend** : Vue 3 (Composition API, `<script setup>`), Vite, TypeScript, Tailwind CSS, composants UI de style Shadcn-vue.
- **Backend** : Python 3.11+, FastAPI, Pydantic v2, WebSockets.
- **Traitement Audio & Vidéo** :
  - **Librosa** : Détection du BPM, séparation harmonique/percussive (HPSS), énergie Sub-Bass (`20-150 Hz`) et détection des **Beat Drops** majeurs.
  - **MoviePy & OpenCV** : Découpage des rushs, recadrage vertical `9:16` (`1080x1920`) et **Speed Ramping** synchronisant le sommet de chaque répétition avec le timestamp exact d'un Beat Drop.
  - **FFmpeg** : Application d'une LUT 3D (`dark_aesthetic.cube`), micro-contraste musculaire (`unsharp`) et vignettage.
- **Intelligence Artificielle Locale** : **Ollama** (`llava:latest`) évaluant l'intensité de l'effort physique (`1-10`) et la contraction maximale sur 1 frame extraite par seconde.
- **Infrastructure** : Docker & `docker-compose` prêts pour un serveur Linux ou un conteneur LXC (`host.docker.internal:host-gateway`).

---

## Démarrage Rapide (Docker / Linux / LXC)

1. **Vérifier qu'Ollama tourne sur la machine hôte** avec le modèle LLaVA :
   ```bash
   ollama pull llava:latest
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

2. **Lancer la stack complète avec Docker Compose** :
   ```bash
   cp .env.example .env
   docker compose up --build -d
   ```

3. **Accéder à l'application** :
   - **Studio Web Vue 3** : `http://localhost:5173`
   - **Documentation Swagger FastAPI** : `http://localhost:8000/docs`
