"""
Point d'entrée principal du Backend FastAPI - AestheticEngine.
==============================================================
Orchestre le workflow de montage TikTok Musculation / Posing Aesthetic :
  1. Upload asynchrone des rushs vidéo (.mp4, .mov) et de la piste audio (.mp3/.wav)
  2. Analyse musicale via Librosa (BPM + extraction des points de coupe Beat Drops)
  3. Détection intelligente des séquences de posing (Intro + Pose Locks variés via CV + Ollama)
  4. Montage par coupes franches ("Hard Cuts", sans transition) en 9:16 ou 16:9 sans étirement
  5. Étalonnage ("Natural Aesthetic", "Dark Aesthetic" ou "Monochrome") via FFmpeg
  6. Génération de la vidéo .mp4 finale et du fichier XML (EDL) Premiere Pro / DaVinci Resolve
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import (
    PipelineStage,
    ProcessProjectRequest,
    ProjectStatusResponse,
    UploadResponse,
)
from app.services.audio_analyzer import AudioBeatDropAnalyzer
from app.services.color_grading import DarkAestheticColorGrader
from app.services.edl_exporter import PremiereResolveXMLExporter
from app.services.video_editor import SpeedRampVideoEditor
from app.services.vision_scorer import OllamaVisionScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("aesthetic_engine")

PROJECTS_STATE: Dict[str, ProjectStatusResponse] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialise les dossiers de stockage et la LUT 3D Dark Aesthetic au démarrage."""
    settings.ensure_directories()
    grader = DarkAestheticColorGrader(lut_dir=settings.LUT_DIR)
    grader.ensure_default_dark_lut("dark_aesthetic")
    logger.info(
        "AestheticEngine démarré | Ollama URL: %s | Modèle: %s",
        settings.OLLAMA_BASE_URL,
        settings.OLLAMA_VISION_MODEL,
    )
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "API de montage vidéo automatisé pour TikTok Posing & Musculation "
        "(Coupes franches sur les Beats, 9:16 ou 16:9 sans étirement, sans transition)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", tags=["Système"])
async def health_check() -> dict:
    """Vérifie l'état du backend et la connectivité avec le serveur Ollama local."""
    ollama_online = False
    available_models: List[str] = []

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                ollama_online = True
                data = resp.json()
                available_models = [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        ollama_online = False

    return {
        "status": "ok",
        "ollama_url": settings.OLLAMA_BASE_URL,
        "ollama_online": ollama_online,
        "configured_vision_model": settings.OLLAMA_VISION_MODEL,
        "available_models": available_models,
    }


@app.post(
    "/api/v1/projects/upload",
    response_model=UploadResponse,
    status_code=201,
    tags=["Pipeline Vidéo"],
)
async def upload_project_media(
    videos: List[UploadFile] = File(
        ..., description="Un ou plusieurs rushs vidéo de posing/musculation (.mp4, .mov)"
    ),
    audio: Optional[UploadFile] = File(
        default=None,
        description="Piste musicale optionnelle (.mp3, .wav, .m4a). Si omise, montage muet (son vidéo à 0).",
    ),
) -> UploadResponse:
    if not videos:
        raise HTTPException(status_code=400, detail="Veuillez fournir au moins un rush vidéo.")

    project_id = uuid.uuid4().hex[:10]
    project_upload_dir = settings.uploads_dir / project_id
    videos_dir = project_upload_dir / "videos"
    audio_dir = project_upload_dir / "audio"
    videos_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    saved_video_names: List[str] = []
    allowed_video_exts = {".mp4", ".mov", ".mkv", ".webm"}

    for idx, video_file in enumerate(videos):
        original_name = Path(video_file.filename or f"rush_{idx + 1}.mp4").name
        ext = Path(original_name).suffix.lower() or ".mp4"
        if ext not in allowed_video_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Format vidéo non supporté ({original_name}). Utilisez .mp4 ou .mov.",
            )

        safe_video_name = f"{idx + 1:02d}_{original_name}"
        dest_video_path = videos_dir / safe_video_name
        with dest_video_path.open("wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        saved_video_names.append(safe_video_name)

    audio_name: Optional[str] = None
    if audio is not None and audio.filename:
        audio_name = Path(audio.filename).name
        dest_audio_path = audio_dir / audio_name
        with dest_audio_path.open("wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

    audio_status_msg = (
        f"piste '{audio_name}'"
        if audio_name
        else "mode sans musique (son vidéo coupé à 0)"
    )

    PROJECTS_STATE[project_id] = ProjectStatusResponse(
        project_id=project_id,
        stage=PipelineStage.UPLOADED,
        progress_percent=5,
        current_step_detail=(
            f"{len(saved_video_names)} rush(s) vidéo et {audio_status_msg} prêts pour l'analyse."
        ),
        video_files=saved_video_names,
        audio_file=audio_name,
        logs=[
            f"[UPLOAD] Projet #{project_id} créé avec {len(saved_video_names)} rush(s) vidéo.",
            (
                f"[UPLOAD] Piste audio chargée : {audio_name}"
                if audio_name
                else "[UPLOAD] Aucune musique sélectionnée → Montage sans musique (son vidéo forcé à 0)."
            ),
        ],
    )

    return UploadResponse(
        project_id=project_id,
        message="Fichiers uploadés avec succès. Prêt à lancer le traitement.",
        video_files=saved_video_names,
        audio_file=audio_name,
    )


@app.post(
    "/api/v1/projects/{project_id}/process",
    response_model=ProjectStatusResponse,
    tags=["Pipeline Vidéo"],
)
async def start_project_processing(
    project_id: str,
    options: ProcessProjectRequest,
    background_tasks: BackgroundTasks,
) -> ProjectStatusResponse:
    project = PROJECTS_STATE.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable.")

    if project.stage not in (PipelineStage.UPLOADED, PipelineStage.COMPLETED, PipelineStage.FAILED):
        raise HTTPException(
            status_code=409,
            detail=f"Un traitement est déjà en cours pour le projet '{project_id}'.",
        )

    project.stage = PipelineStage.ANALYZING_AUDIO
    project.progress_percent = 8
    project.error_message = None
    project.current_step_detail = "Démarrage de l'analyse spectrale et détection des Beat Cuts..."
    project.logs.append(
        f"[START] Pipeline lancé (Format={options.aspect_ratio}, Mode={options.editing_mode}, "
        f"Durée cible={options.target_duration}s, Look={options.lut_preset})"
    )

    background_tasks.add_task(_run_full_editing_pipeline, project_id, options)
    return project


@app.get(
    "/api/v1/projects/{project_id}/status",
    response_model=ProjectStatusResponse,
    tags=["Pipeline Vidéo"],
)
async def get_project_status(project_id: str) -> ProjectStatusResponse:
    project = PROJECTS_STATE.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' introuvable.")
    return project


@app.websocket("/ws/projects/{project_id}")
async def project_progress_websocket(websocket: WebSocket, project_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            project = PROJECTS_STATE.get(project_id)
            if not project:
                await websocket.send_json({"error": "Projet introuvable"})
                break

            await websocket.send_json(project.model_dump())
            if project.stage in (PipelineStage.COMPLETED, PipelineStage.FAILED):
                break
            await asyncio.sleep(0.75)
    except WebSocketDisconnect:
        logger.info("Client WebSocket déconnecté du projet %s", project_id)


@app.get("/api/v1/projects/{project_id}/download/video", tags=["Exports"])
async def download_final_video(project_id: str) -> FileResponse:
    video_path = settings.exports_dir / project_id / f"tiktok_dark_aesthetic_{project_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="La vidéo finale n'est pas encore disponible.")
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=video_path.name,
    )


@app.get("/api/v1/projects/{project_id}/download/xml", tags=["Exports"])
async def download_premiere_xml(project_id: str) -> FileResponse:
    xml_path = settings.exports_dir / project_id / f"timeline_{project_id}.xml"
    if not xml_path.exists():
        raise HTTPException(status_code=404, detail="Le fichier XML EDL n'est pas encore disponible.")
    return FileResponse(
        path=xml_path,
        media_type="application/xml",
        filename=xml_path.name,
    )


def _update_project(
    project_id: str,
    stage: PipelineStage,
    progress: int,
    detail: str,
    log_entry: str | None = None,
) -> None:
    project = PROJECTS_STATE.get(project_id)
    if not project:
        return
    project.stage = stage
    project.progress_percent = int(max(0, min(100, progress)))
    project.current_step_detail = detail
    if log_entry:
        project.logs.append(log_entry)


async def _run_full_editing_pipeline(
    project_id: str,
    options: ProcessProjectRequest,
) -> None:
    project = PROJECTS_STATE[project_id]
    project_upload_dir = settings.uploads_dir / project_id
    videos_dir = project_upload_dir / "videos"
    audio_dir = project_upload_dir / "audio"
    temp_project_dir = settings.temp_dir / project_id
    export_project_dir = settings.exports_dir / project_id
    temp_project_dir.mkdir(parents=True, exist_ok=True)
    export_project_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(videos_dir.glob("*"))
    audio_path = next(audio_dir.glob("*"), None)

    if not video_paths:
        _update_project(
            project_id,
            PipelineStage.FAILED,
            0,
            "Aucun rush vidéo trouvé dans le dossier projet.",
            "[ERREUR] Rushs vidéo introuvables.",
        )
        project.error_message = "Rushs vidéo introuvables."
        return

    try:
        # ---------------------------------------------------------------------
        # ÉTAPE 1 : ANALYSE AUDIO (LIBROSA) OU GRILLE MUETTE (SANS MUSIQUE)
        # ---------------------------------------------------------------------
        audio_analyzer = AudioBeatDropAnalyzer()
        if audio_path is not None and audio_path.exists():
            _update_project(
                project_id,
                PipelineStage.ANALYZING_AUDIO,
                15,
                f"Analyse musicale Librosa sur '{audio_path.name}' (durée cible: {options.target_duration:.1f}s)...",
            )
            audio_result = await asyncio.to_thread(
                audio_analyzer.analyze_track,
                audio_path,
                options.drop_sensitivity,
                options.target_duration,
            )
            audio_log_prefix = f"[AUDIO] BPM={audio_result.bpm}"
        else:
            _update_project(
                project_id,
                PipelineStage.ANALYZING_AUDIO,
                15,
                f"Mode sans musique : génération de la grille de coupes ({options.target_duration:.1f}s, son vidéo à 0)...",
            )
            audio_result = audio_analyzer.generate_silent_cut_grid(
                sensitivity=options.drop_sensitivity,
                target_duration=options.target_duration,
            )
            audio_log_prefix = "[SANS MUSIQUE] Son vidéo coupé à 0"

        project.audio_analysis = audio_result
        drop_times_str = ", ".join(f"{d.timestamp:.2f}s" for d in audio_result.beat_drops) or "aucun"
        needed_shots = len(audio_result.beat_drops) + 1

        _update_project(
            project_id,
            PipelineStage.ANALYZING_VISION,
            28,
            f"{len(audio_result.beat_drops)} coupes programmées ({needed_shots} plans de posing à extraire).",
            f"{audio_log_prefix} | Durée={audio_result.effective_duration}s | Coupes à : [{drop_times_str}]",
        )

        # ---------------------------------------------------------------------
        # ÉTAPE 2 : DÉTECTION DES SÉQUENCES DE POSING & FLEX (CV + OLLAMA)
        # ---------------------------------------------------------------------
        vision_scorer = OllamaVisionScorer(
            ollama_base_url=settings.OLLAMA_BASE_URL,
            model_name=options.ollama_model,
        )

        def on_vision_progress(msg: str, sub_pct: int) -> None:
            mapped_pct = 28 + int(sub_pct * 0.30)
            _update_project(project_id, PipelineStage.ANALYZING_VISION, mapped_pct, msg)

        frame_scores, selected_segments = await vision_scorer.analyze_videos(
            video_paths=video_paths,
            min_effort_score=options.min_effort_score,
            target_segment_count=needed_shots,
            include_intro_shot=options.include_intro_shot,
            progress_callback=on_vision_progress,
        )
        project.frame_scores = frame_scores
        project.selected_segments = selected_segments

        pose_summary = ", ".join(
            f"{s.pose_label} (@{s.peak_time_in_source:.1f}s)" for s in selected_segments[:needed_shots]
        )
        _update_project(
            project_id,
            PipelineStage.SPEED_RAMPING,
            60,
            f"{len(selected_segments)} séquences de posing variées sélectionnées.",
            f"[POSING IA] Plans retenus : {pose_summary}",
        )

        # ---------------------------------------------------------------------
        # ÉTAPE 3 : ASSEMBLAGE BEAT-SYNC SANS TRANSITION & ROGNAGE (9:16 / 16:9)
        # ---------------------------------------------------------------------
        video_editor = SpeedRampVideoEditor()
        raw_assembled_path = temp_project_dir / f"raw_assembled_{project_id}.mp4"

        def on_ramp_progress(msg: str, sub_pct: int) -> None:
            mapped_pct = 60 + int(sub_pct * 0.22)
            _update_project(project_id, PipelineStage.SPEED_RAMPING, mapped_pct, msg)

        synced_segments = await asyncio.to_thread(
            video_editor.assemble_synced_timeline,
            selected_segments,
            audio_result.beat_drops,
            audio_path,
            raw_assembled_path,
            audio_result.effective_duration,
            options.aspect_ratio,
            options.editing_mode,
            options.fast_speed_factor,
            options.slow_speed_factor,
            on_ramp_progress,
        )
        project.selected_segments = synced_segments
        _update_project(
            project_id,
            PipelineStage.COLOR_GRADING,
            84,
            f"Application du profil colorimétrique '{options.lut_preset}' en {options.aspect_ratio}...",
            f"[MONTAGE] {len(synced_segments)} plans assemblés par coupes franches (sans transition, sans étirement).",
        )

        # ---------------------------------------------------------------------
        # ÉTAPE 4 : ÉTALONNAGE VIA FFMPEG
        # ---------------------------------------------------------------------
        color_grader = DarkAestheticColorGrader(lut_dir=settings.LUT_DIR)
        final_video_path = export_project_dir / f"tiktok_dark_aesthetic_{project_id}.mp4"
        await asyncio.to_thread(
            color_grader.apply_dark_aesthetic_grade,
            raw_assembled_path,
            final_video_path,
            options.lut_preset,
        )

        _update_project(
            project_id,
            PipelineStage.EXPORTING_EDL,
            94,
            "Génération du fichier XML (EDL) pour Premiere Pro & DaVinci Resolve...",
            f"[COLOR GRADING] Rendu '{options.lut_preset}' terminé.",
        )

        # ---------------------------------------------------------------------
        # ÉTAPE 5 : EXPORT DU FICHIER XML (EDL) PREMIERE PRO / DAVINCI RESOLVE
        # ---------------------------------------------------------------------
        xml_exporter = PremiereResolveXMLExporter()
        final_xml_path = export_project_dir / f"timeline_{project_id}.xml"
        await asyncio.to_thread(
            xml_exporter.export_timeline_xml,
            project_id,
            synced_segments,
            audio_result.beat_drops,
            audio_path,
            final_xml_path,
            options.aspect_ratio,
        )

        project.output_video_url = f"/api/v1/projects/{project_id}/download/video"
        project.output_xml_url = f"/api/v1/projects/{project_id}/download/xml"
        _update_project(
            project_id,
            PipelineStage.COMPLETED,
            100,
            f"Montage Posing ({options.aspect_ratio}, {audio_result.effective_duration:.1f}s) prêt !",
            "[EXPORT] Vidéo MP4 et fichier XML Premiere/Resolve générés avec succès.",
        )

    except Exception as exc:
        logger.exception("Erreur critique dans le pipeline du projet %s", project_id)
        project.error_message = str(exc)
        _update_project(
            project_id,
            PipelineStage.FAILED,
            project.progress_percent,
            f"Échec du traitement : {exc}",
            f"[ERREUR] {exc}",
        )
