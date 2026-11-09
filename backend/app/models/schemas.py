"""
Schémas Pydantic v2 pour la validation des entrées/sorties de l'API AestheticEngine.
Optimisé pour les TikToks de Posing, Musculation & Aesthetic (coupes franches sur les beats,
respect strict des ratios 9:16 et 16:9 sans étirement, sélection multi-poses intelligente).
"""

from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    """Étapes séquentielles du pipeline de génération vidéo."""

    UPLOADED = "uploaded"
    ANALYZING_AUDIO = "analyzing_audio"
    ANALYZING_VISION = "analyzing_vision"
    SPEED_RAMPING = "speed_ramping"
    COLOR_GRADING = "color_grading"
    EXPORTING_EDL = "exporting_edl"
    COMPLETED = "completed"
    FAILED = "failed"


class BeatDropEvent(BaseModel):
    """Représente un point de coupe musical (Beat Drop ou accent de phrase) dans la piste audio."""

    timestamp: float = Field(..., description="Timestamp exact du cut/drop en secondes")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Intensité relative normalisée (0.0 à 1.0)")
    buildup_start: float = Field(..., description="Timestamp du point de coupe précédent en secondes")
    sub_bass_energy: float = Field(..., description="Énergie mesurée dans les basses fréquences (20-150 Hz)")


class AudioAnalysisResult(BaseModel):
    """Résultat complet de l'analyse musicale par Librosa."""

    bpm: float = Field(..., description="Tempo estimé en battements par minute (BPM)")
    duration: float = Field(..., description="Durée totale de la piste audio en secondes")
    effective_duration: float = Field(default=21.5, description="Durée retenue pour le montage TikTok")
    beat_timestamps: List[float] = Field(default_factory=list, description="Tous les temps forts détectés")
    beat_drops: List[BeatDropEvent] = Field(default_factory=list, description="Points de coupe (Beat Drops / Accents)")


class FrameEffortScore(BaseModel):
    """Évaluation IA / Computer Vision d'un instant ou d'une pose dans le rush vidéo."""

    video_filename: str = Field(..., description="Nom du fichier vidéo source")
    timestamp: float = Field(..., description="Position temporelle dans le rush (secondes)")
    effort_score: float = Field(..., ge=1.0, le=10.0, description="Note esthétique / contraction musculaire (1 à 10)")
    exercise_detected: str = Field(default="Pose Aesthetic", description="Pose ou mouvement identifié")
    is_peak_contraction: bool = Field(
        default=False,
        description="Indique si l'athlète verrouille une pose (contraction isométrique)",
    )
    reasoning: str = Field(default="", description="Détail de la détection")


class SelectedVideoSegment(BaseModel):
    """Segment de pose retenu et calé entre deux points de coupe musicaux (sans transition)."""

    source_video: str = Field(..., description="Chemin du rush vidéo d'origine")
    start_time: float = Field(..., description="Début du segment dans le rush source (secondes)")
    end_time: float = Field(..., description="Fin du segment dans le rush source (secondes)")
    peak_time_in_source: float = Field(
        ...,
        description="Timestamp du verrouillage de la pose dans le rush source",
    )
    timeline_start: float = Field(default=0.0, description="Début du plan sur la timeline finale (secondes)")
    timeline_end: float = Field(default=0.0, description="Fin du plan sur la timeline finale (secondes)")
    pose_label: str = Field(default="Pose", description="Nom ou type du plan (Intro, Pose #1, etc.)")
    max_effort_score: float = Field(..., description="Score esthétique / intensité du segment")
    target_beat_drop: Optional[float] = Field(
        default=None,
        description="Timestamp du beat sur lequel ce plan démarre (coupe franche)",
    )
    speed_fast_factor: float = Field(default=1.0, description="Facteur de vitesse d'entrée")
    speed_slow_factor: float = Field(default=1.0, description="Facteur de vitesse de maintien")


class ProcessProjectRequest(BaseModel):
    """Paramètres configurables par l'utilisateur lors du lancement du rendu."""

    aspect_ratio: Literal["9:16", "16:9"] = Field(
        default="9:16",
        description="Format de sortie : 9:16 (Vertical TikTok 1080x1920) ou 16:9 (Paysage 1920x1080), par rognage centré sans étirement.",
    )
    editing_mode: Literal["pose_cut", "speed_ramp"] = Field(
        default="pose_cut",
        description="'pose_cut' = Coupes franches 1.0x sur les beats (style exemple.mp4) | 'speed_ramp' = Speed Ramping sur les drops.",
    )
    target_duration: float = Field(
        default=21.5,
        ge=8.0,
        le=60.0,
        description="Durée cible du TikTok final en secondes (ex: 21.5s comme exemple.mp4).",
    )
    include_intro_shot: bool = Field(
        default=True,
        description="Inclure le plan d'introduction (mise en place avant le 1er drop à ~1.9s).",
    )
    min_effort_score: float = Field(
        default=7.0,
        ge=1.0,
        le=10.0,
        description="Seuil de qualité de pose (1-10)",
    )
    ollama_model: str = Field(
        default="llava:latest",
        description="Modèle de vision Ollama optionnel",
    )
    fast_speed_factor: float = Field(
        default=1.35,
        ge=1.0,
        le=2.5,
        description="Vitesse d'amorce (uniquement en mode speed_ramp)",
    )
    slow_speed_factor: float = Field(
        default=0.75,
        ge=0.4,
        le=1.0,
        description="Vitesse de ralenti (uniquement en mode speed_ramp)",
    )
    lut_preset: str = Field(
        default="natural_aesthetic",
        description="Preset colorimétrique : 'natural_aesthetic' (fidèle au rush comme exemple.mp4), 'dark_aesthetic', ou 'obsidian_monochrome'",
    )
    drop_sensitivity: float = Field(
        default=0.65,
        ge=0.1,
        le=0.95,
        description="Fréquence des coupes sur les beats (0.1 = plans longs ~4s, 0.95 = coupes rapides ~2s)",
    )


class UploadResponse(BaseModel):
    """Réponse renvoyée après l'upload des rushs vidéo et de la piste audio."""

    project_id: str
    message: str
    video_files: List[str]
    audio_file: str


class ProjectStatusResponse(BaseModel):
    """État temps-réel d'un projet de montage automatisé."""

    project_id: str
    stage: PipelineStage
    progress_percent: int = Field(..., ge=0, le=100)
    current_step_detail: str
    video_files: List[str] = Field(default_factory=list)
    audio_file: Optional[str] = None
    audio_analysis: Optional[AudioAnalysisResult] = None
    frame_scores: List[FrameEffortScore] = Field(default_factory=list)
    selected_segments: List[SelectedVideoSegment] = Field(default_factory=list)
    output_video_url: Optional[str] = None
    output_xml_url: Optional[str] = None
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
