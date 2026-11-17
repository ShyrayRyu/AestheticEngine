# =============================================================================
# AestheticEngine — Lanceur Windows 11 (PowerShell)
# =============================================================================

$ErrorActionPreference = "Stop"

# Couleurs ANSI / PowerShell
function Write-Cyan($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Green($msg) { Write-Host $msg -ForegroundColor Green }
function Write-Yellow($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Red($msg) { Write-Host $msg -ForegroundColor Red }

$banner = @"
   _   ___ ___ _____ _  _ ___ _____ ___ ___   ___ _  _  ___ ___ _  _ ___ 
  /_\ | __/ __|_   _| || | __|_   _|_ _/ __| | __| \| |/ __|_ _| \| | __|
 / _ \| _|\__ \ | | | __ | _|  | |  | | (__  | _|| .` | (_ || || .` | _| 
/_/ \_\___|___/ |_| |_||_|___| |_| |___|___| |___|_|\_|\___|___|_|\_|___|
   Montage Video TikTok Musculation Automatise — Windows 11 Runner
"@

Write-Cyan $banner
Write-Host ""

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $rootDir

# 1. Verification de Python
Write-Cyan "==> Verification de Python..."
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
} else {
    Write-Red "[ERREUR] Python n'est pas installe ou n'est pas dans le PATH."
    Write-Yellow "Telechargez Python 3.11 ou superieur sur : https://www.python.org/downloads/"
    exit 1
}

$pyVersion = & $pythonCmd --version
Write-Green "[INFO] Python detecte : $pyVersion"

# 2. Verification du gestionnaire de paquets frontend (bun > pnpm > npm)
Write-Cyan "==> Verification du gestionnaire Frontend..."
$pkgMgr = $null
if (Get-Command bun -ErrorAction SilentlyContinue) {
    $pkgMgr = "bun"
} elseif (Get-Command pnpm -ErrorAction SilentlyContinue) {
    $pkgMgr = "pnpm"
} elseif (Get-Command npm -ErrorAction SilentlyContinue) {
    $pkgMgr = "npm"
} else {
    Write-Red "[ERREUR] Aucun gestionnaire JS (bun, pnpm, npm) trouve."
    Write-Yellow "Installez Bun (recommande, ultra rapide) : powershell -c `"irm bun.sh/install.ps1 | iex`""
    Write-Yellow "Ou NodeJS : https://nodejs.org/"
    exit 1
}
$pmVersion = & $pkgMgr --version
Write-Green "[INFO] Gestionnaire frontend detecte : $pkgMgr (v$pmVersion)"

# 3. Verification d'Ollama
Write-Cyan "==> Verification d'Ollama..."
try {
    $ollamaRes = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 2 -ErrorAction Stop
    $models = $ollamaRes.models.name
    Write-Green "[INFO] Ollama est en ligne sur le port 11434."
    if ($models) {
        Write-Green "[INFO] Modeles installes : $($models -join ', ')"
    }
    if ($models -contains "llava:latest" -or $models -contains "llava") {
        Write-Green "[INFO] Modele de vision LLaVA detecte pour l'analyse IA."
    } else {
        Write-Yellow "[ATTENTION] Le modele de vision 'llava:latest' n'est pas installe."
        Write-Yellow "Pour activer le scoring des poses : ollama pull llava:latest"
    }
} catch {
    Write-Yellow "[ATTENTION] Ollama n'est pas accessible sur http://localhost:11434."
    Write-Yellow "(Optionnel pour le scoring IA. Lancez Ollama si vous souhaitez l'analyse visuelle)."
}

# 4. Fichiers .env
$envTemplate = @"
# Configuration locale generee automatiquement
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VISION_MODEL=llava:latest
MIN_EFFORT_SCORE=7.0
STORAGE_DIR=storage
LUT_DIR=assets/luts
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
"@

if (-not (Test-Path ".env")) {
    Set-Content -Path ".env" -Value $envTemplate -Encoding UTF8
    Write-Green "[INFO] Fichier .env racine cree."
}
if (-not (Test-Path "backend\.env")) {
    Set-Content -Path "backend\.env" -Value $envTemplate -Encoding UTF8
    Write-Green "[INFO] Fichier backend\.env cree."
}
if (-not (Test-Path "frontend\.env")) {
    Set-Content -Path "frontend\.env" -Value "VITE_API_BASE_URL=http://localhost:8000`nVITE_WS_BASE_URL=ws://localhost:8000" -Encoding UTF8
    Write-Green "[INFO] Fichier frontend\.env cree."
}

# 5. Environnement virtuel Backend
Write-Cyan "`n==> Preparation de l'environnement virtuel Backend..."
$venvDir = Join-Path $rootDir "backend\.venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"
$venvScripts = Join-Path $venvDir "Scripts"

if (-not (Test-Path $venvPython)) {
    Write-Green "[INFO] Creation du virtualenv Python dans $venvDir..."
    & $pythonCmd -m venv $venvDir
}

# Verifier les paquets
$testImports = "import fastapi, librosa, moviepy; print('OK')"
$importOk = $false
try {
    $res = & $venvPython -c $testImports 2>$null
    if ($res -like "*OK*") { $importOk = $true }
} catch {}

if (-not $importOk) {
    Write-Green "[INFO] Installation des dependances backend (FastAPI, Librosa, MoviePy)..."
    & $venvPip install --upgrade pip
    & $venvPip install -r "backend\requirements.txt"
} else {
    Write-Green "[INFO] Dependances backend pretes."
}

# Assurer FFmpeg dans le venv
$ffmpegExe = Join-Path $venvScripts "ffmpeg.exe"
if (-not (Test-Path $ffmpegExe) -and -not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Green "[INFO] Extraction de FFmpeg embarque via imageio-ffmpeg..."
    $copyFfmpeg = "import imageio_ffmpeg, shutil, pathlib, sys; src = imageio_ffmpeg.get_ffmpeg_exe(); dst = pathlib.Path(sys.executable).parent / 'ffmpeg.exe'; shutil.copyfile(src, dst) if not dst.exists() else None; print('FFMPEG_OK')"
    & $venvPython -c $copyFfmpeg
}

# Ajouter Scripts au PATH de la session courante
$env:PATH = "$venvScripts;" + $env:PATH

# 6. Dependances Frontend
Write-Cyan "`n==> Preparation des dependances Frontend..."
if (-not (Test-Path "frontend\node_modules")) {
    Write-Green "[INFO] Installation des modules avec $pkgMgr..."
    Set-Location "frontend"
    & $pkgMgr install
    Set-Location $rootDir
} else {
    Write-Green "[INFO] Dependances frontend pretes."
}

# 7. Lancement des serveurs
Write-Host ""
Write-Green "=========================================================="
Write-Green "  AestheticEngine est pret a demarrer !"
Write-Green "  👉 Interface Web : http://localhost:5173"
Write-Green "  👉 API Swagger   : http://localhost:8000/docs"
Write-Yellow "  Appuyez sur Ctrl+C dans cette fenetre pour arreter."
Write-Green "=========================================================="
Write-Host ""

# Lancement du Backend en arriere-plan
$backendArgs = "-m uvicorn app.main:app --host 0.0.0.0 --port 8000"
$backendProc = Start-Process -FilePath $venvPython -ArgumentList $backendArgs -WorkingDirectory (Join-Path $rootDir "backend") -PassThru -NoNewWindow

Start-Sleep -Seconds 1

# Ouverture automatique du navigateur
Start-Process "http://localhost:5173"

try {
    # Lancement du Frontend au premier plan
    Set-Location "frontend"
    & $pkgMgr run dev --host 0.0.0.0 --port 5173
} finally {
    Write-Yellow "`nArrêt des services AestheticEngine..."
    if ($backendProc -and -not $backendProc.HasExited) {
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    }
    Set-Location $rootDir
    Write-Green "Services arretes proprement."
}
