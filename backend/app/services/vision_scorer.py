"""
Service d'Analyse de Posing, Esthétique Musculaire & Filtrage IA (Computer Vision + Ollama LLaVA).
--------------------------------------------------------------------------------------------------
Conçu spécifiquement pour les vidéos TikTok de Posing / Aesthetic / Bodybuilding :
  1. Détecte automatiquement si l'utilisateur a fourni un long rush continu contenant
     plusieurs poses successives (ex: 2 minutes face au miroir) ou plusieurs rushs courts.
  2. Isole le plan d'introduction ("Walk-in / Mise en place" sous la lumière avant le 1er drop).
  3. Identifie chaque fenêtre de "Pose Lock / Contraction Isométrique" (le moment où l'athlète
     prend sa pose, contracte et maintient la posture pendant 2 à 4 secondes).
  4. Calcule une empreinte visuelle (descripteur de silhouette) par pose pour maximiser la
     variété entre deux coupes consécutives (ex: Most Muscular -> Side Chest -> Double Biceps).
  5. Interroge Ollama (LLaVA) uniquement sur les meilleures poses candidates s'il est en ligne,
     sans jamais bloquer ou ralentir le pipeline si Ollama est éteint.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import cv2
import httpx
import numpy as np

from app.core.config import settings
from app.models.schemas import FrameEffortScore, SelectedVideoSegment

logger = logging.getLogger(__name__)

LLAVA_POSING_PROMPT = """Tu es un juge d'élite de bodybuilding et monteur TikTok "Dark Aesthetic Posing".
Analyse cette image extraite d'une séance de posing musculation.
Évalue la qualité esthétique et l'intensité de la contraction musculaire :
- Définition musculaire et striations (pectoraux, deltoïdes, dorsaux, bras, abdominaux, vascularité).
- Symétrie, posture de posing (Double Biceps, Most Muscular, Lat Spread, Side Chest/Triceps).
- Éclairage dramatique, V-Taper et présence charismatique sous la lumière.
Pénalise sévèrement les moments où l'athlète se repose, a les bras ballants le long du corps, ou ajuste ses vêtements.
Réponds UNIQUEMENT avec un objet JSON valide :
{
  "effort_score": 9.2,
  "exercise_detected": "Back Double Biceps / Most Muscular / Side Chest / Front Lat Spread",
  "is_peak_contraction": true,
  "reasoning": "Contraction maximale des dorsaux et deltoïdes avec V-taper prononcé sous éclairage zénithal"
}
"""


@dataclass
class PoseCandidateWindow:
    """Fenêtre candidate correspondant à une pose verrouillée dans un rush."""

    video_path: Path
    center_time: float
    start_time: float
    end_time: float
    score: float
    motion_val: float
    gradient_val: float
    descriptor: np.ndarray
    is_intro: bool = False
    pose_label: str = "Pose Aesthetic"
    rotation_correction: int = 0
    upright_aspect_ratio: Literal["9:16", "16:9"] = "9:16"


class OllamaVisionScorer:
    """
    Moteur hybride Computer Vision + Ollama LLaVA qui extrait les meilleures séquences
    de posing et de contraction musculaire à partir d'un ou plusieurs rushs vidéo.
    """

    def __init__(
        self,
        ollama_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        sample_interval_sec: float = 0.25,
    ) -> None:
        self.ollama_base_url = (ollama_base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.OLLAMA_VISION_MODEL
        self.sample_interval_sec = sample_interval_sec

    async def analyze_videos(
        self,
        video_paths: List[Path],
        min_effort_score: float = 7.0,
        target_segment_count: int = 8,
        include_intro_shot: bool = True,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Tuple[List[FrameEffortScore], List[SelectedVideoSegment]]:
        """
        Analyse l'ensemble des rushs pour détecter :
          - Le plan d'intro (mise en place sous la lumière) si `include_intro_shot=True`
          - Les `target_segment_count` meilleures poses distinctes et variées.
        """
        if progress_callback:
            progress_callback("Analyse biomécanique des poses et contractions musculaires...", 10)

        # 1. Extraction rapide par Computer Vision (OpenCV) des courbes de mouvement et de poses
        all_candidates: List[PoseCandidateWindow] = []
        intro_candidate: Optional[PoseCandidateWindow] = None

        for v_idx, video_path in enumerate(video_paths):
            cands, intro = await asyncio.to_thread(
                self._scan_video_pose_windows,
                video_path,
                max(16, target_segment_count * 2),
            )
            all_candidates.extend(cands)
            if intro_candidate is None and intro is not None:
                intro_candidate = intro

            if progress_callback:
                pct = int(((v_idx + 1) / max(1, len(video_paths))) * 60)
                progress_callback(
                    f"Rush analysé : {video_path.name} ({len(cands)} poses détectées)",
                    pct,
                )

        if not all_candidates:
            raise ValueError("Aucune séquence vidéo exploitable n'a pu être extraite des rushs.")

        # 2. Sélection diversifiée des poses (évite de répéter la même pose deux fois de suite)
        diversified_poses = self._select_diverse_poses(
            candidates=all_candidates,
            desired_count=max(6, target_segment_count),
        )

        # 3. Vérification unique et rapide (1.5s) de la disponibilité d'Ollama LLaVA
        ollama_available = await self._check_ollama_online()
        frame_scores: List[FrameEffortScore] = []

        async with httpx.AsyncClient(timeout=15.0) as client:
            for idx, pose_win in enumerate(diversified_poses):
                if ollama_available:
                    score_obj = await self._score_candidate_with_ollama(
                        client=client,
                        candidate=pose_win,
                        pose_index=idx + 1,
                    )
                else:
                    # Notation basée sur la stabilité isométrique et la netteté musculaire
                    mapped_score = round(float(np.clip(7.2 + pose_win.score * 2.6, 6.5, 9.9)), 1)
                    score_obj = FrameEffortScore(
                        video_filename=pose_win.video_path.name,
                        timestamp=round(pose_win.center_time, 2),
                        effort_score=mapped_score,
                        exercise_detected=pose_win.pose_label,
                        is_peak_contraction=True,
                        reasoning=(
                            f"Pose verrouillée (stabilité={pose_win.motion_val:.2f}, "
                            f"définition musculaire={pose_win.gradient_val:.1f})"
                        ),
                    )

                pose_win.score = score_obj.effort_score
                pose_win.pose_label = score_obj.exercise_detected
                frame_scores.append(score_obj)

                if progress_callback:
                    pct = 60 + int(((idx + 1) / max(1, len(diversified_poses))) * 40)
                    progress_callback(
                        f"Pose #{idx + 1} retenue @ {pose_win.center_time:.1f}s ({score_obj.effort_score:.1f}/10)",
                        pct,
                    )

        # 4. Construction de la liste finale de segments (Plan d'intro + Poses variées)
        selected_segments: List[SelectedVideoSegment] = []

        if include_intro_shot and intro_candidate is not None:
            selected_segments.append(
                SelectedVideoSegment(
                    source_video=str(intro_candidate.video_path),
                    start_time=round(intro_candidate.start_time, 3),
                    end_time=round(intro_candidate.end_time, 3),
                    peak_time_in_source=round(intro_candidate.center_time, 3),
                    pose_label="Intro / Mise en place",
                    max_effort_score=8.5,
                    rotation_correction=intro_candidate.rotation_correction,
                    upright_aspect_ratio=intro_candidate.upright_aspect_ratio,
                    hold_duration=round(intro_candidate.end_time - intro_candidate.start_time, 2),
                )
            )

        for idx, pose_win in enumerate(diversified_poses):
            if pose_win.score >= min_effort_score or len(selected_segments) < target_segment_count:
                selected_segments.append(
                    SelectedVideoSegment(
                        source_video=str(pose_win.video_path),
                        start_time=round(pose_win.start_time, 3),
                        end_time=round(pose_win.end_time, 3),
                        peak_time_in_source=round(pose_win.center_time, 3),
                        pose_label=pose_win.pose_label or f"Pose #{idx + 1}",
                        max_effort_score=round(pose_win.score, 1),
                        rotation_correction=pose_win.rotation_correction,
                        upright_aspect_ratio=pose_win.upright_aspect_ratio,
                        hold_duration=round(pose_win.end_time - pose_win.start_time, 2),
                    )
                )

        return frame_scores, selected_segments

    @staticmethod
    def resolve_optimal_aspect_ratio(
        segments: List[SelectedVideoSegment],
        requested_aspect_ratio: Literal["9:16", "16:9"],
        auto_orient_and_frame: bool = True,
    ) -> Tuple[Literal["9:16", "16:9"], bool, str]:
        """
        Vérifie si les rushs (une fois remis à l'endroit dans le bon sens) correspondent
        au format `9:16` ou `16:9` sélectionné.
        Si `auto_orient_and_frame=True` et que le sens réel des rushs est inversé par rapport
        au format choisi (ex: rush filmé en 16:9 alors que 9:16 est coché, ou inversement,
        ou rush tourné à 90°/270° qui change de ratio une fois redressé), l'IA interchange
        automatiquement entre `9:16` et `16:9` sans jamais rogner ni recadrer la vidéo.
        """
        if not segments:
            return requested_aspect_ratio, False, "Format conservé par défaut."

        rotated_count = sum(1 for s in segments if s.rotation_correction != 0)
        count_9_16 = sum(1 for s in segments if s.upright_aspect_ratio == "9:16")
        count_16_9 = sum(1 for s in segments if s.upright_aspect_ratio == "16:9")

        dominant_ratio: Literal["9:16", "16:9"] = (
            "16:9" if count_16_9 > count_9_16 else "9:16"
        )

        if not auto_orient_and_frame:
            msg = (
                f"Format {requested_aspect_ratio} forcé manuellement "
                f"(rotation IA appliquée sur {rotated_count} plan(s), sans rognage)."
            )
            return requested_aspect_ratio, rotated_count > 0, msg

        if dominant_ratio != requested_aspect_ratio:
            msg = (
                f"IA Sens & Format : vidéo détectée en {dominant_ratio} une fois dans le bon sens "
                f"→ interchange automatique {requested_aspect_ratio} ↔ {dominant_ratio} "
                f"(sans rognage, redressement {rotated_count} plan(s))."
            )
            logger.info(msg)
            return dominant_ratio, True, msg

        if rotated_count > 0:
            msg = (
                f"IA Sens & Format : vidéo remise dans le bon sens "
                f"({rotated_count} plan(s) redressé(s)) en {dominant_ratio} sans rognage."
            )
            logger.info(msg)
            return dominant_ratio, True, msg

        msg = f"IA Sens & Format : vidéo bien orientée dans le bon sens ({dominant_ratio}, sans rognage)."
        return dominant_ratio, False, msg

    @staticmethod
    def _rotate_frame_cw(frame: np.ndarray, angle: int) -> np.ndarray:
        """Applique une rotation horaire de 0°, 90°, 180° ou 270° à une image OpenCV."""
        norm_angle = angle % 360
        if norm_angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if norm_angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if norm_angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _detect_video_orientation_and_framing(
        self,
        video_path: Path,
    ) -> Tuple[int, Literal["9:16", "16:9"]]:
        """
        Reconnaît avec précision l'orientation réelle d'un rush vidéo :
          - Si le flux est déjà décodé en portrait vertical (h >= w, ex: 1080x1920 ou 720x1280),
            l'orientation est STRICTEMENT 0° (9:16). Toute rotation de 90°/270° coucherait
            inutilement l'athlète sur le flanc.
          - Si le flux est décodé en paysage horizontal (w > h), on vérifie si la vidéo a été
            filmée verticalement (téléphone orienté en mode portrait mais fichier encodé en paysage)
            en évaluant si une rotation de 90° ou 270° redresse une silhouette humaine debout.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0, "9:16"

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = n_frames / fps if fps > 0 else 0.0

        sample_times = np.linspace(
            max(0.2, duration * 0.12),
            max(0.5, duration * 0.88),
            num=min(6, max(3, int(duration))),
        )
        sampled_frames: List[np.ndarray] = []
        try:
            for t_sec in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec) * 1000.0)
                ok, frm = cap.read()
                if ok and frm is not None:
                    sampled_frames.append(frm)
        finally:
            cap.release()

        if not sampled_frames:
            return 0, "9:16"

        raw_h, raw_w = sampled_frames[0].shape[:2]

        # 1. RÈGLE MAJEURE : Si le flux est déjà vertical (hauteur >= largeur)
        # OpenCV et FFmpeg intègrent déjà automatiquement les métadonnées de rotation (displaymatrix).
        # Un rush vertical 9:16 est donc DÉJÀ dans le bon sens !
        if raw_h >= raw_w:
            logger.info(
                "Rush '%s' : format natif vertical portrait (%dx%d) -> rotation_correction=0° (9:16 garanti)",
                video_path.name,
                raw_w,
                raw_h,
            )
            return 0, "9:16"

        # 2. Cas d'un rush horizontal (raw_w > raw_h, ex: 1920x1080)
        # On vérifie si l'athlète a filmé à la verticale sans balise de rotation
        face_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
            except Exception as e:
                logger.warning("Impossible de charger le CascadeClassifier OpenCV : %s", e)
                face_cascade = None

        scores: Dict[int, float] = {0: 1.0, 90: 0.0, 270: 0.0}

        for cand_angle in (0, 90, 270):
            for raw_f in sampled_frames:
                rot_f = self._rotate_frame_cw(raw_f, cand_angle)
                rh, rw = rot_f.shape[:2]
                scale = 320.0 / float(max(rh, rw))
                small = cv2.resize(
                    rot_f,
                    (max(32, int(rw * scale)), max(32, int(rh * scale))),
                )
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                sh, sw = gray.shape[:2]

                if face_cascade is not None and not face_cascade.empty():
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
                    for fx, fy, fw, fh in faces:
                        cy = (fy + fh * 0.5) / float(sh)
                        cx = (fx + fw * 0.5) / float(sw)
                        if 0.05 <= cy <= 0.48 and 0.18 <= cx <= 0.82:
                            scores[cand_angle] += 4.0

        best_angle = 0
        if scores[90] > scores[0] * 2.0 and scores[90] > scores[270] + 2.0:
            best_angle = 90
        elif scores[270] > scores[0] * 2.0 and scores[270] > scores[90] + 2.0:
            best_angle = 270

        upright_aspect: Literal["9:16", "16:9"] = "9:16" if best_angle in (90, 270) else "16:9"

        logger.info(
            "Orientation rush paysage '%s' : rotation_correction=%d° | aspect=%s | scores=%s",
            video_path.name,
            best_angle,
            upright_aspect,
            {k: round(v, 2) for k, v in scores.items()},
        )
        return best_angle, upright_aspect

    def _scan_video_pose_windows(
        self,
        video_path: Path,
        max_candidates: int = 16,
    ) -> Tuple[List[PoseCandidateWindow], Optional[PoseCandidateWindow]]:
        """
        Analyse approfondie du rush vidéo pour détecter les véritables contractions musculaires :
          1. Détecte et garantit l'orientation exacte (0° pour 9:16 natif).
          2. Extrait à haute vitesse (FFmpeg rawvideo pipe) les flux de mouvement et de contour.
          3. Détecte les vrais plateaux isométriques ("Pose Locks") : les moments où l'athlète maintient
             une contraction ferme (1.6s à 4.5s) avec forte netteté musculaire et déploiement de la silhouette
             (bras levés, dos déployé, abdos contractés, deltoïdes sous éclairage zénithal).
          4. Élimine formellement les temps morts de repos (bras ballants le long du corps).
          5. Isole automatiquement le plan d'intro (mise en place sous le spot juste avant la 1re pose).
        """
        rot_corr, upright_aspect = self._detect_video_orientation_and_framing(video_path)

        # 1. Décodage ultra-rapide des frames basse résolution (90x160) via FFmpeg
        w_small, h_small = 90, 160
        fps_s = 2.0
        frames_array = self._extract_fast_grayscale_frames(
            video_path=video_path,
            rotation_correction=rot_corr,
            target_w=w_small,
            target_h=h_small,
            fps_sample=fps_s,
        )

        n_frames = len(frames_array)
        duration = n_frames / fps_s if n_frames > 0 else 0.0

        if n_frames < 8:
            return [], None

        # 2. Vitesse de mouvement inter-trames & lissage
        diffs = np.mean(
            np.abs(frames_array[1:].astype(np.float32) - frames_array[:-1].astype(np.float32)),
            axis=(1, 2),
        )
        kernel = np.ones(3, dtype=np.float32) / 3.0
        smooth_mot = np.convolve(diffs, kernel, mode="same")

        # 3. Zone corporelle supérieure (deltoïdes, trapèzes, pectoraux, dorsaux, bras)
        # y: 18%..55% de la hauteur, x: 10%..90% de la largeur
        y1, y2 = int(h_small * 0.18), int(h_small * 0.55)
        x1, x2 = int(w_small * 0.10), int(w_small * 0.90)
        upper_body = frames_array[:, y1:y2, x1:x2]

        # 3bis. Détection stricte de présence humaine dans le rectangle central du corps
        # En l'absence de l'athlète (ex: porte, casiers ou pièce vides), le rectangle central
        # est lisse et sans détails vivants : smooth_cgrad < 14.0 et smooth_cstd < 30.0.
        # Quand un athlète est présent (pectoraux, dorsaux, débardeur, veines, tête),
        # smooth_cgrad s'élève entre 17.5 et 26.5.
        cy1, cy2 = int(h_small * 0.22), int(h_small * 0.65)
        cx1, cx2 = int(w_small * 0.25), int(w_small * 0.75)
        center_body = frames_array[:, cy1:cy2, cx1:cx2]
        cgx = np.abs(center_body[:, :, 1:].astype(np.float32) - center_body[:, :, :-1].astype(np.float32))
        cgy = np.abs(center_body[:, 1:, :].astype(np.float32) - center_body[:, :-1, :].astype(np.float32))
        center_grad = cgx[:, :-1, :].mean(axis=(1, 2)) + cgy[:, :, :-1].mean(axis=(1, 2))
        smooth_cgrad = np.convolve(center_grad, kernel, mode="same")
        center_std = np.std(center_body, axis=(1, 2))
        smooth_cstd = np.convolve(center_std, kernel, mode="same")
        has_human_frame = (smooth_cgrad >= 14.5) & (smooth_cstd >= 30.0)

        # 4. Gradient musculaire et striations (Sobel) sur le haut du corps
        gx = np.abs(upper_body[:, :, 1:].astype(np.float32) - upper_body[:, :, :-1].astype(np.float32))
        gy = np.abs(upper_body[:, 1:, :].astype(np.float32) - upper_body[:, :-1, :].astype(np.float32))
        grad_energy = gx[:, :-1, :].mean(axis=(1, 2)) + gy[:, :, :-1].mean(axis=(1, 2))
        smooth_grad = np.convolve(grad_energy, kernel, mode="same")

        # 5. Déploiement des bras et V-Taper (activité latérale externe des coudes et dorsaux)
        flank_w = int(upper_body.shape[2] * 0.28)
        left_flank = upper_body[:, :, :flank_w]
        right_flank = upper_body[:, :, -flank_w:]
        outer_expansion = np.std(left_flank, axis=(1, 2)) + np.std(right_flank, axis=(1, 2))
        smooth_expansion = np.convolve(outer_expansion, kernel, mode="same")

        # 6. Densité musculaire / peau sous spot zénithal (55 à 220)
        skin_density = np.mean((upper_body >= 55) & (upper_body <= 220), axis=(1, 2))
        smooth_skin = np.convolve(skin_density, kernel, mode="same")

        # 7. Seuil de mouvement adaptatif : permet les contractions isométriques intenses
        # Lors d'une contraction maximale (Most Muscular, Crab Flex), les muscles tremblent
        # légèrement et la respiration s'intensifie (motion entre 3.0 et 6.5).
        p50 = float(np.percentile(smooth_mot, 50))
        mot_thresh = max(3.0, min(6.8, p50 * 1.35))

        # Fenêtre de sécurité : ignore le réglage de la caméra / téléphone au tout début
        start_guard = 4.0 if duration > 20.0 else 1.0
        end_guard = max(start_guard + 2.0, duration - 2.5)

        raw_holds: List[dict] = []
        in_hold = False
        s_idx = 0

        for i in range(len(smooth_mot)):
            t = i / fps_s
            if t < start_guard or t > end_guard:
                if in_hold:
                    in_hold = False
                continue

            # Maintien isométrique valide UNIQUEMENT si l'athlète est réellement présent dans le cadre
            is_still = bool(smooth_mot[i] < mot_thresh and has_human_frame[i])
            if is_still:
                if not in_hold:
                    in_hold = True
                    s_idx = i
            else:
                if in_hold:
                    in_hold = False
                    hold_len = (i - s_idx) / fps_s
                    if 1.4 <= hold_len <= 7.0:
                        rec = self._build_hold_record(
                            frames_array=frames_array,
                            smooth_mot=smooth_mot,
                            smooth_grad=smooth_grad,
                            smooth_expansion=smooth_expansion,
                            smooth_skin=smooth_skin,
                            smooth_cgrad=smooth_cgrad,
                            smooth_cstd=smooth_cstd,
                            s_idx=s_idx,
                            e_idx=i,
                            fps_s=fps_s,
                            w_small=w_small,
                        )
                        if rec is not None:
                            raw_holds.append(rec)

        if in_hold and 1.4 <= (len(smooth_mot) - s_idx) / fps_s <= 7.0:
            rec = self._build_hold_record(
                frames_array=frames_array,
                smooth_mot=smooth_mot,
                smooth_grad=smooth_grad,
                smooth_expansion=smooth_expansion,
                smooth_skin=smooth_skin,
                smooth_cgrad=smooth_cgrad,
                smooth_cstd=smooth_cstd,
                s_idx=s_idx,
                e_idx=len(smooth_mot),
                fps_s=fps_s,
                w_small=w_small,
            )
            if rec is not None:
                raw_holds.append(rec)

        # Si aucun hold strict n'a été trouvé, on élargit le seuil avec présence humaine obligatoire
        if not raw_holds:
            logger.warning("Aucun plateau isométrique strict détecté — recherche assouplie...")
            mot_thresh_relaxed = max(4.0, float(np.median(smooth_mot)) * 1.25)
            in_hold = False
            s_idx = 0
            for i in range(len(smooth_mot)):
                t = i / fps_s
                if t < 1.0 or t > duration - 1.0:
                    continue
                if smooth_mot[i] < mot_thresh_relaxed and has_human_frame[i]:
                    if not in_hold:
                        in_hold = True
                        s_idx = i
                else:
                    if in_hold:
                        in_hold = False
                        hold_len = (i - s_idx) / fps_s
                        if 1.2 <= hold_len <= 7.0:
                            rec = self._build_hold_record(
                                frames_array=frames_array,
                                smooth_mot=smooth_mot,
                                smooth_grad=smooth_grad,
                                smooth_expansion=smooth_expansion,
                                smooth_skin=smooth_skin,
                                smooth_cgrad=smooth_cgrad,
                                smooth_cstd=smooth_cstd,
                                s_idx=s_idx,
                                e_idx=i,
                                fps_s=fps_s,
                                w_small=w_small,
                            )
                            if rec is not None:
                                raw_holds.append(rec)

        # 8. Détection intelligente du plan d'Intro (la mise en place de l'athlète sous le spot)
        # On recherche la première fenêtre où l'athlète est réellement dans le cadre (has_human_frame)
        intro_dur = 2.4
        intro_found = False
        intro_start = 0.5
        intro_end = min(duration - 1.0, 2.9)

        for t_probe in np.arange(1.0, min(duration - intro_dur, 25.0), 0.5):
            s_probe = int(t_probe * fps_s)
            e_probe = int((t_probe + intro_dur) * fps_s)
            if s_probe < len(has_human_frame) and e_probe <= len(has_human_frame):
                if np.all(has_human_frame[s_probe:e_probe]):
                    intro_start = round(t_probe, 3)
                    intro_end = round(t_probe + intro_dur, 3)
                    intro_found = True
                    break

        if not intro_found and raw_holds:
            # Repli sécurisé juste avant le premier hold validé
            intro_start = max(0.5, raw_holds[0]["start"] - 2.8)
            intro_end = raw_holds[0]["start"] - 0.4

        intro_cand = PoseCandidateWindow(
            video_path=video_path,
            center_time=round((intro_start + intro_end) / 2.0, 3),
            start_time=intro_start,
            end_time=intro_end,
            score=8.5,
            motion_val=1.5,
            gradient_val=float(np.mean(smooth_grad)),
            descriptor=np.zeros(256, dtype=np.float32),
            is_intro=True,
            pose_label="Intro / Mise en place",
            rotation_correction=rot_corr,
            upright_aspect_ratio=upright_aspect,
        )

        # 9. Tri par score esthétique et élimination des doublons temporels
        raw_holds.sort(key=lambda h: float(h["score"]), reverse=True)
        selected_holds: List[dict] = []
        min_gap_sec = 3.5 if duration > 30.0 else 1.8

        for h in raw_holds:
            c_t = float(h["mid"])
            if all(abs(c_t - float(sel["mid"])) >= min_gap_sec for sel in selected_holds):
                selected_holds.append(h)
                if len(selected_holds) >= max_candidates:
                    break

        candidates: List[PoseCandidateWindow] = []
        for rank_idx, h in enumerate(selected_holds):
            candidates.append(
                PoseCandidateWindow(
                    video_path=video_path,
                    center_time=h["mid"],
                    start_time=h["start"],
                    end_time=h["end"],
                    score=h["score"],
                    motion_val=h["mot"],
                    gradient_val=h["grad"],
                    descriptor=h["desc"],
                    is_intro=False,
                    pose_label=h["label"],
                    rotation_correction=rot_corr,
                    upright_aspect_ratio=upright_aspect,
                )
            )

        logger.info(
            "Rush '%s' : %d vraies poses esthétiques sélectionnées (intro @ %.1f-%.1fs, 1re pose @ %.1fs, top score=%.1f)",
            video_path.name,
            len(candidates),
            intro_start,
            intro_end,
            candidates[0].start_time if candidates else 0.0,
            candidates[0].score if candidates else 0.0,
        )
        return candidates, intro_cand

    def _build_hold_record(
        self,
        frames_array: np.ndarray,
        smooth_mot: np.ndarray,
        smooth_grad: np.ndarray,
        smooth_expansion: np.ndarray,
        smooth_skin: np.ndarray,
        smooth_cgrad: np.ndarray,
        smooth_cstd: np.ndarray,
        s_idx: int,
        e_idx: int,
        fps_s: float,
        w_small: int,
    ) -> Optional[dict]:
        """Construit l'analyse détaillée et le score esthétique d'un plateau isométrique."""
        # 1. Vérification formelle de la présence humaine sur l'ensemble du maintien
        avg_cg = float(np.mean(smooth_cgrad[s_idx:e_idx]))
        avg_cs = float(np.mean(smooth_cstd[s_idx:e_idx]))
        if avg_cg < 14.5 or avg_cs < 30.0:
            # Rejette formellement toute pièce ou porte vide
            return None

        h_start = round(s_idx / fps_s, 3)
        h_end = round(e_idx / fps_s, 3)
        h_len = round(h_end - h_start, 3)
        avg_g = float(np.mean(smooth_grad[s_idx:e_idx]))
        avg_m = float(np.mean(smooth_mot[s_idx:e_idx]))
        avg_exp = float(np.mean(smooth_expansion[s_idx:e_idx]))
        avg_skin = float(np.mean(smooth_skin[s_idx:e_idx]))
        mid_idx = (s_idx + e_idx) // 2
        mid_t = round(mid_idx / fps_s, 3)

        # Descripteur de silhouette normalisé 16x16
        desc = cv2.resize(frames_array[mid_idx], (16, 16)).astype(np.float32).flatten()
        desc -= float(np.mean(desc))
        norm_d = float(np.linalg.norm(desc))
        if norm_d > 1e-5:
            desc /= norm_d

        # Composantes du score esthétique musculation :
        # 1. Définition et striations musculaires (Sobel)
        grad_pts = (avg_g / 20.0) * 30.0

        # 2. Déploiement des bras & V-taper (coudes/dorsaux écartés)
        expansion_pts = (min(55.0, avg_exp) / 55.0) * 25.0

        # 3. Présence musculaire / peau exposée sous le spot (pondération quadratique)
        skin_pts = (avg_skin ** 2) * 25.0

        # 4. Durée idéale de tenue (une vraie pose dure 1.8s à 4.5s)
        if 1.8 <= h_len <= 4.5:
            dur_pts = 15.0
        elif h_len < 1.8:
            dur_pts = 8.0
        else:
            # Pénalité pour les temps morts / repos prolongés (> 4.5s)
            dur_pts = max(0.0, 15.0 - (h_len - 4.5) * 4.0)

        # 5. Bonus esthétique majeur pour les poses torse nu / peau découverte
        shirtless_bonus = 20.0 if avg_skin >= 0.72 else (10.0 if avg_skin >= 0.65 else 0.0)

        # 6. Pénalité sévère si posture de repos (bras ballants le long du corps)
        resting_penalty = 30.0 if avg_exp < 28.0 else 0.0
        mot_penalty = float(avg_m * 1.8)

        raw_score = float(
            grad_pts + expansion_pts + skin_pts + dur_pts + shirtless_bonus - resting_penalty - mot_penalty
        )

        # Normalisation sur l'échelle standard 7.0 - 9.9
        normalized_score = float(np.clip(7.0 + (raw_score - 40.0) * (2.9 / 55.0), 7.0, 9.9))

        label = self._classify_pose_silhouette(
            expansion_val=avg_exp,
            grad_val=avg_g,
            skin_val=avg_skin,
        )

        return {
            "start": h_start,
            "end": h_end,
            "dur": h_len,
            "mid": mid_t,
            "grad": avg_g,
            "mot": avg_m,
            "expansion": avg_exp,
            "skin": avg_skin,
            "raw_score": raw_score,
            "score": round(normalized_score, 1),
            "desc": desc,
            "label": label,
        }

    @staticmethod
    def _classify_pose_silhouette(
        expansion_val: float,
        grad_val: float,
        skin_val: float,
    ) -> str:
        """Détermine le type de pose musculation selon le gabarit biomécanique et le relief."""
        if expansion_val >= 107.5:
            if grad_val >= 16.0:
                return "Back Double Biceps / Overhead Flex"
            return "Front Lat Spread (V-Taper)"
        if expansion_val >= 102.5:
            if skin_val >= 0.76:
                return "Most Muscular (Crab / Flex)"
            return "Front Double Biceps"
        if expansion_val < 102.5:
            if skin_val >= 0.75:
                return "Side Chest & Triceps Lock"
            return "Abdominals & Serratus Flex"
        return "Classic Aesthetic Flex"

    def _extract_fast_grayscale_frames(
        self,
        video_path: Path,
        rotation_correction: int,
        target_w: int,
        target_h: int,
        fps_sample: float = 2.0,
    ) -> np.ndarray:
        """
        Extrait rapidement les frames basse résolution via FFmpeg pipe.
        Échelonne d'abord en basse résolution avant d'appliquer la rotation
        pour une exécution ultra-rapide même sur les rushs de 10+ minutes.
        """
        rot = rotation_correction % 360
        if rot == 90:
            vf = f"fps={fps_sample},scale={target_h}:{target_w},transpose=1,format=gray"
        elif rot == 270:
            vf = f"fps={fps_sample},scale={target_h}:{target_w},transpose=2,format=gray"
        elif rot == 180:
            vf = f"fps={fps_sample},scale={target_w}:{target_h},hflip,vflip,format=gray"
        else:
            vf = f"fps={fps_sample},scale={target_w}:{target_h},format=gray"

        cmd = [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-",
        ]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw_bytes, _ = p.communicate(timeout=300.0)
            if p.returncode == 0 and len(raw_bytes) >= target_w * target_h * 4:
                frame_size = target_w * target_h
                n = len(raw_bytes) // frame_size
                return np.frombuffer(raw_bytes[: n * frame_size], dtype=np.uint8).reshape((n, target_h, target_w))
        except Exception as e:
            logger.warning("Échec de l'extraction FFmpeg rapide (%s), repli sur OpenCV.", e)

        # Repli OpenCV séquentiel
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return np.zeros((0, target_h, target_w), dtype=np.uint8)

        fps_vid = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        step = max(1, int(round(fps_vid / fps_sample)))
        frames_list: List[np.ndarray] = []
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    if rot != 0:
                        frame = self._rotate_frame_cw(frame, rot)
                    small = cv2.resize(frame, (target_w, target_h))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    frames_list.append(gray)
                idx += 1
        finally:
            cap.release()

        return np.array(frames_list, dtype=np.uint8) if frames_list else np.zeros((0, target_h, target_w), dtype=np.uint8)

    def _select_diverse_poses(
        self,
        candidates: List[PoseCandidateWindow],
        desired_count: int,
    ) -> List[PoseCandidateWindow]:
        """
        Sélectionne les poses candidates selon l'algorithme Maximal Marginal Relevance (MMR)
        pour garantir que chaque plan consécutif présente une pose visuellement très différente
        (ex: Most Muscular -> Side Chest -> Front Double Biceps -> Lat Spread) tout en privilégiant
        la qualité esthétique maximale sans pénaliser artificiellement les meilleures poses de la vidéo.
        """
        if len(candidates) <= 1:
            return candidates

        pool = list(candidates)
        pool.sort(key=lambda c: c.score, reverse=True)
        chosen: List[PoseCandidateWindow] = [pool.pop(0)]

        while pool and len(chosen) < desired_count:
            last_desc = chosen[-1].descriptor

            def mmr_score(cand: PoseCandidateWindow) -> float:
                # Différence visuelle par rapport à la pose précédente
                cos_sim_last = float(np.dot(cand.descriptor, last_desc))
                visual_diff = float(np.clip(1.0 - cos_sim_last, 0.0, 1.0))

                # Distance temporelle minimale par rapport à l'ensemble des poses déjà sélectionnées
                min_time_dist = min(abs(cand.center_time - c.center_time) for c in chosen)
                if min_time_dist < 3.2:
                    # Évite les doublons quasi-immédiats de la même contraction
                    return -999.0

                time_factor = min(1.0, min_time_dist / 14.0)

                # La note de qualité esthétique (cand.score) domine le classement (~75%)
                # La différence visuelle apporte une saine variété de poses (~20%)
                # L'espacement temporel apporte un léger bonus d'aération (~5%)
                return (cand.score * 8.5) + (visual_diff * 12.0) + (time_factor * 3.5)

            valid_candidates = [c for c in pool if mmr_score(c) > -900.0]
            if not valid_candidates:
                best_next = max(pool, key=lambda c: c.score)
            else:
                best_next = max(valid_candidates, key=mmr_score)

            pool.remove(best_next)
            chosen.append(best_next)

        return chosen

    async def _check_ollama_online(self) -> bool:
        """Vérifie en 1.5s max si le serveur Ollama local répond."""
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                resp = await client.get(f"{self.ollama_base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            logger.info(
                "Serveur Ollama non détecté sur %s — utilisation directe du moteur Computer Vision Posing.",
                self.ollama_base_url,
            )
            return False

    async def _score_candidate_with_ollama(
        self,
        client: httpx.AsyncClient,
        candidate: PoseCandidateWindow,
        pose_index: int,
    ) -> FrameEffortScore:
        """Interroge Ollama LLaVA sur la frame clé d'une pose candidate."""
        frame_b64 = self._extract_single_frame_b64(
            candidate.video_path,
            candidate.center_time,
            candidate.rotation_correction,
        )
        if not frame_b64:
            return FrameEffortScore(
                video_filename=candidate.video_path.name,
                timestamp=round(candidate.center_time, 2),
                effort_score=round(candidate.score, 1),
                exercise_detected=candidate.pose_label,
                is_peak_contraction=True,
                reasoning="Pose esthétique verrouillée détectée par analyse biomécanique",
            )

        payload = {
            "model": self.model_name,
            "prompt": LLAVA_POSING_PROMPT,
            "images": [frame_b64],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.15},
        }
        try:
            resp = await client.post(f"{self.ollama_base_url}/api/generate", json=payload)
            resp.raise_for_status()
            parsed = self._parse_llm_json(resp.json().get("response", "{}"))
            score = float(np.clip(float(parsed.get("effort_score", candidate.score)), 7.0, 10.0))
            pose_name = str(parsed.get("exercise_detected") or candidate.pose_label)
            reasoning = str(parsed.get("reasoning") or "Évaluation LLaVA")
            return FrameEffortScore(
                video_filename=candidate.video_path.name,
                timestamp=round(candidate.center_time, 2),
                effort_score=round(score, 1),
                exercise_detected=pose_name,
                is_peak_contraction=True,
                reasoning=reasoning,
            )
        except Exception:
            return FrameEffortScore(
                video_filename=candidate.video_path.name,
                timestamp=round(candidate.center_time, 2),
                effort_score=round(candidate.score, 1),
                exercise_detected=candidate.pose_label,
                is_peak_contraction=True,
                reasoning=f"Pose #{pose_index} sélectionnée par analyse esthétique & striations",
            )

    @classmethod
    def _extract_single_frame_b64(
        cls,
        video_path: Path,
        timestamp_sec: float,
        rotation_correction: int = 0,
    ) -> Optional[str]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            frame = cls._rotate_frame_cw(frame, rotation_correction)
            h, w = frame.shape[:2]
            scale = 512.0 / float(max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None
        finally:
            cap.release()

    @staticmethod
    def _parse_llm_json(raw_text: str) -> dict:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {}
