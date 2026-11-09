"""
Configuration centrale du backend AestheticEngine via Pydantic Settings.
Permet de surcharger chaque paramètre par variable d'environnement ou fichier .env.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres globaux de l'application."""

    APP_NAME: str = "AestheticEngine API"
    APP_VERSION: str = "1.0.0"

    # Configuration réseau & CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

    # Configuration Ollama (Vision IA locale type LLaVA)
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_VISION_MODEL: str = "llava:latest"
    OLLAMA_TIMEOUT_SECONDS: float = 45.0

    # Paramètres par défaut du pipeline de montage TikTok
    MIN_EFFORT_SCORE: float = 7.0
    FRAME_SAMPLE_INTERVAL_SEC: float = 1.0
    TARGET_RESOLUTION_WIDTH: int = 1080
    TARGET_RESOLUTION_HEIGHT: int = 1920
    TARGET_FPS: int = 60

    # Répertoires de stockage
    STORAGE_DIR: Path = Path("/app/storage")
    LUT_DIR: Path = Path("/app/assets/luts")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def uploads_dir(self) -> Path:
        return self.STORAGE_DIR / "uploads"

    @property
    def temp_dir(self) -> Path:
        return self.STORAGE_DIR / "temp"

    @property
    def exports_dir(self) -> Path:
        return self.STORAGE_DIR / "exports"

    def ensure_directories(self) -> None:
        """Crée l'arborescence de stockage et de LUTs si elle n'existe pas."""
        for directory in (self.uploads_dir, self.temp_dir, self.exports_dir, self.LUT_DIR):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
