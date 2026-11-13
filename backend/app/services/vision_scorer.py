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

LLAVA_POSING_PROMPT = """Tu es un juge de bodybuilding et monteur TikTok "Aesthetic Posing".
Analyse cette image extraite d'une séance de posing musculation.
Évalue la qualité esthétique de la pose et de la contraction musculaire (définition, symétrie, éclairage zénithal, charisme).
Vérifie également si l'athlète est dans le bon sens (debout, tête en haut -> rotation_needed: 0) ou si l'image doit être tournée dans le sens horaire de 90, 180 ou 270 degrés pour être à l'endroit.
Réponds UNIQUEMENT avec un objet JSON valide :
{
  "effort_score": 8.8,
  "exercise_detected": "Most Muscular / Side Chest / Front Double Biceps / Back Pose",
  "is_peak_contraction": true,
  "rotation_needed": 0,
  "reasoning": "Pose verrouillée avec excellente définition sous éclairage dramatique"
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
                max(8, target_segment_count + 4),
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
        Reconnaît automatiquement si un rush vidéo est dans le bon sens (athlète debout tête en haut)
        ou s'il est tourné sur le côté (90° / 270°) ou à l'envers (180°), et détermine :
          1. `rotation_correction` : 0, 90, 180 ou 270 degrés horaires pour remettre l'athlète debout.
          2. `upright_aspect_ratio` : '9:16' ou '16:9' une fois le rush remis dans le bon sens.
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
            num=min(8, max(3, int(duration))),
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

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        upper_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_upperbody.xml"
        )
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        scores: Dict[int, float] = {0: 0.25, 90: 0.0, 180: 0.0, 270: 0.0}

        for cand_angle in (0, 90, 180, 270):
            grays: List[np.ndarray] = []
            for raw_f in sampled_frames:
                rot_f = self._rotate_frame_cw(raw_f, cand_angle)
                rh, rw = rot_f.shape[:2]
                scale = 320.0 / float(max(rh, rw))
                small = cv2.resize(
                    rot_f,
                    (max(32, int(rw * scale)), max(32, int(rh * scale))),
                )
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                grays.append(gray)
                sh, sw = gray.shape[:2]

                # 1. Détection de visage dans la moitié supérieure de l'image
                if not face_cascade.empty():
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(18, 18))
                    for fx, fy, fw, fh in faces:
                        cy = (fy + fh * 0.5) / float(sh)
                        cx = (fx + fw * 0.5) / float(sw)
                        if 0.04 <= cy <= 0.52 and 0.15 <= cx <= 0.85:
                            scores[cand_angle] += 3.5

                # 2. Détection du haut du corps (épaules / buste)
                if not upper_cascade.empty():
                    uppers = upper_cascade.detectMultiScale(gray, 1.1, 3, minSize=(32, 32))
                    for ux, uy, uw, uh in uppers:
                        cy = (uy + uh * 0.5) / float(sh)
                        cx = (ux + uw * 0.5) / float(sw)
                        if 0.08 <= cy <= 0.68 and 0.15 <= cx <= 0.85:
                            scores[cand_angle] += 2.0

                # 3. Détecteur de silhouette humaine debout (HOG)
                _, weights = hog.detectMultiScale(
                    gray, winStride=(8, 8), padding=(8, 8), scale=1.05
                )
                for w_val in weights:
                    scores[cand_angle] += float(w_val) * 1.5

            # 4. Analyse de la colonne vertébrale de mouvement & éclairage zénithal
            if len(grays) >= 2:
                diffs = [cv2.absdiff(grays[i], grays[i - 1]) for i in range(1, len(grays))]
                mean_mot = np.mean(diffs, axis=0)
                sh, sw = mean_mot.shape[:2]

                col_energy = np.mean(mean_mot, axis=0)
                row_energy = np.mean(mean_mot, axis=1)
                col_conc = float(np.std(col_energy) / (np.mean(col_energy) + 1e-5))
                row_conc = float(np.std(row_energy) / (np.mean(row_energy) + 1e-5))
                if col_conc > row_conc:
                    scores[cand_angle] += 1.5

                top_mot = float(np.mean(mean_mot[int(sh * 0.12) : int(sh * 0.55), :]))
                bot_mot = float(np.mean(mean_mot[int(sh * 0.55) : int(sh * 0.92), :]))
                if top_mot > bot_mot * 1.05:
                    scores[cand_angle] += 1.2

        best_angle = max(scores, key=lambda k: scores[k])
        if best_angle != 0 and scores[best_angle] < scores[0] + 1.2:
            best_angle = 0

        upright_sample = self._rotate_frame_cw(sampled_frames[0], best_angle)
        uh, uw = upright_sample.shape[:2]
        upright_aspect: Literal["9:16", "16:9"] = "9:16" if uh >= uw else "16:9"

        logger.info(
            "Orientation IA '%s' : rotation_correction=%d° | format_redressé=%s | scores=%s",
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
          1. Détecte et applique l'orientation réelle (remet l'athlète debout à l'endroit).
          2. Extrait à haute vitesse (FFmpeg rawvideo pipe à 4 Hz) les flux de mouvement et de contour.
          3. Détecte les vrais plateaux isométriques ("Pose Locks") : les moments où l'athlète maintient
             une contraction ferme (> 1.6s) après être entré dans la pose, avec forte netteté musculaire
             et déploiement de la silhouette (bras levés, dos déployé, abdos contractés).
          4. Isole automatiquement le plan d'intro réel (la mise en place sous le spot juste avant la 1re pose,
             en éliminant tout réglage de caméra / écran au début).
        """
        rot_corr, upright_aspect = self._detect_video_orientation_and_framing(video_path)

        # 1. Décodage ultra-rapide des frames basse résolution (90x160) à 2 Hz via FFmpeg
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

        # 3. Gradient musculaire (Sobel) sur la région du buste et des bras (y: 20%..80%, x: 10%..90%)
        torso = frames_array[:, int(h_small * 0.20) : int(h_small * 0.80), int(w_small * 0.10) : int(w_small * 0.90)]
        gx = np.abs(torso[:, :, 1:].astype(np.float32) - torso[:, :, :-1].astype(np.float32))
        gy = np.abs(torso[:, 1:, :].astype(np.float32) - torso[:, :-1, :].astype(np.float32))
        grad_energy = gx[:, :-1, :].mean(axis=(1, 2)) + gy[:, :, :-1].mean(axis=(1, 2))
        smooth_grad = np.convolve(grad_energy, kernel, mode="same")

        # 4. Envergure de la silhouette (largeur du corps lors des poses: lat spread, double biceps...)
        bg_est = np.median(frames_array[::max(1, int(fps_s * 2))], axis=0)
        fg_mask = np.abs(frames_array.astype(np.float32) - bg_est[np.newaxis, :, :]) > 18.0
        upper_fg = fg_mask[:, int(h_small * 0.18) : int(h_small * 0.72), :]
        body_width = np.sum(np.any(upper_fg, axis=1), axis=1)

        # 5. Seuil de mouvement adaptatif : un plateau isométrique a un mouvement sous le 35e centile
        p35 = float(np.percentile(smooth_mot, 35))
        mot_thresh = max(2.5, min(4.9, p35 * 1.30))

        # Fenêtre de sécurité : ignore le réglage de la caméra / téléphone au tout début
        start_guard = 12.0 if duration > 25.0 else (5.0 if duration > 14.0 else 0.5)
        end_guard = max(start_guard + 2.0, duration - 3.5)

        raw_holds: List[dict] = []
        in_hold = False
        s_idx = 0

        for i in range(len(smooth_mot)):
            t = i / fps_s
            if t < start_guard or t > end_guard:
                if in_hold:
                    in_hold = False
                continue

            is_still = bool(smooth_mot[i] < mot_thresh)
            if is_still:
                if not in_hold:
                    in_hold = True
                    s_idx = i
            else:
                if in_hold:
                    in_hold = False
                    hold_len = (i - s_idx) / fps_s
                    if hold_len >= 1.6:
                        raw_holds.append(
                            self._build_hold_record(
                                frames_array=frames_array,
                                smooth_mot=smooth_mot,
                                smooth_grad=smooth_grad,
                                body_width=body_width,
                                s_idx=s_idx,
                                e_idx=i,
                                fps_s=fps_s,
                                w_small=w_small,
                            )
                        )

        if in_hold and (len(smooth_mot) - s_idx) / fps_s >= 1.6:
            raw_holds.append(
                self._build_hold_record(
                    frames_array=frames_array,
                    smooth_mot=smooth_mot,
                    smooth_grad=smooth_grad,
                    body_width=body_width,
                    s_idx=s_idx,
                    e_idx=len(smooth_mot),
                    fps_s=fps_s,
                    w_small=w_small,
                )
            )

        # Si aucun hold strict n'a été trouvé (ex: rush court très actif), on élargit le seuil
        if not raw_holds:
            logger.warning("Aucun plateau isométrique strict détecté — recherche assouplie...")
            mot_thresh_relaxed = max(4.0, float(np.median(smooth_mot)))
            in_hold = False
            s_idx = 0
            for i in range(len(smooth_mot)):
                t = i / fps_s
                if t < 1.0 or t > duration - 1.0:
                    continue
                if smooth_mot[i] < mot_thresh_relaxed:
                    if not in_hold:
                        in_hold = True
                        s_idx = i
                else:
                    if in_hold:
                        in_hold = False
                        hold_len = (i - s_idx) / fps_s
                        if hold_len >= 1.2:
                            raw_holds.append(
                                self._build_hold_record(
                                    frames_array=frames_array,
                                    smooth_mot=smooth_mot,
                                    smooth_grad=smooth_grad,
                                    body_width=body_width,
                                    s_idx=s_idx,
                                    e_idx=i,
                                    fps_s=fps_s,
                                    w_small=w_small,
                                )
                            )

        # 6. Détection intelligente du plan d'Intro (la mise en place sous le spot)
        intro_cand: Optional[PoseCandidateWindow] = None
        if raw_holds and duration >= 14.0:
            first_hold_start = min(h["start"] for h in raw_holds)
            if first_hold_start >= 4.0:
                intro_end = round(max(2.0, first_hold_start - 0.35), 3)
                intro_start = round(max(0.5, intro_end - 2.5), 3)
            else:
                intro_start = 0.5
                intro_end = min(duration - 1.0, 2.8)
        else:
            intro_start = 0.5
            intro_end = min(duration - 1.0, 2.8)

        intro_cand = PoseCandidateWindow(
            video_path=video_path,
            center_time=round((intro_start + intro_end) / 2.0, 3),
            start_time=intro_start,
            end_time=intro_end,
            score=0.92,
            motion_val=1.5,
            gradient_val=float(np.mean(smooth_grad)),
            descriptor=np.zeros(256, dtype=np.float32),
            is_intro=True,
            pose_label="Intro / Mise en place",
            rotation_correction=rot_corr,
            upright_aspect_ratio=upright_aspect,
        )

        # 7. Tri par score biomécanique et élimination des doublons temporels
        raw_holds.sort(key=lambda h: float(h["score"]), reverse=True)
        selected_holds: List[dict] = []
        min_gap_sec = 4.0 if duration > 30.0 else 1.8

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
            "Rush '%s' : %d vraies poses verrouillées détectées (intro @ %.1f-%.1fs, 1re pose @ %.1fs)",
            video_path.name,
            len(candidates),
            intro_start,
            intro_end,
            candidates[0].start_time if candidates else 0.0,
        )
        return candidates, intro_cand

    def _build_hold_record(
        self,
        frames_array: np.ndarray,
        smooth_mot: np.ndarray,
        smooth_grad: np.ndarray,
        body_width: np.ndarray,
        s_idx: int,
        e_idx: int,
        fps_s: float,
        w_small: int,
    ) -> dict:
        """Construit l'analyse détaillée d'un plateau isométrique."""
        h_start = round(s_idx / fps_s, 3)
        h_end = round(e_idx / fps_s, 3)
        h_len = round(h_end - h_start, 3)
        avg_g = float(np.mean(smooth_grad[s_idx:e_idx]))
        avg_m = float(np.mean(smooth_mot[s_idx:e_idx]))
        avg_w = float(np.mean(body_width[s_idx:e_idx]))
        mid_idx = (s_idx + e_idx) // 2
        mid_t = round(mid_idx / fps_s, 3)

        # Descripteur de silhouette normalisé 16x16
        desc = cv2.resize(frames_array[mid_idx], (16, 16)).astype(np.float32).flatten()
        desc -= float(np.mean(desc))
        norm_d = float(np.linalg.norm(desc))
        if norm_d > 1e-5:
            desc /= norm_d

        # Score qualité de pose : longue tenue stable + forte définition musculaire + silhouette déployée
        score = float(
            (avg_g * 0.45)
            + (avg_w * 0.35)
            + (min(8.0, h_len) * 2.5)
            - (avg_m * 2.5)
        )

        # Classification morphologique de la pose selon le gabarit de la silhouette
        label = self._classify_pose_silhouette(
            width_ratio=avg_w / float(w_small),
            grad_val=avg_g,
        )

        return {
            "start": h_start,
            "end": h_end,
            "dur": h_len,
            "mid": mid_t,
            "grad": avg_g,
            "mot": avg_m,
            "width": avg_w,
            "score": score,
            "desc": desc,
            "label": label,
        }

    @staticmethod
    def _classify_pose_silhouette(width_ratio: float, grad_val: float) -> str:
        """Détermine le type de pose musculation selon le gabarit et les reliefs."""
        if width_ratio >= 0.72:
            return "Front Double Biceps / Lat Spread"
        if width_ratio >= 0.62:
            if grad_val >= 18.0:
                return "Most Muscular (Crab / Flex)"
            return "Front Lat Spread (V-Taper)"
        if width_ratio <= 0.48:
            return "Side Chest & Biceps Lock"
        if grad_val >= 17.0:
            return "Abdominals & Thighs Flex"
        return "Classic Aesthetic Pose"

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
        (ex: Most Muscular -> Side Chest -> Front Double Biceps -> Lat Spread).
        """
        if len(candidates) <= 1:
            return candidates

        pool = list(candidates)
        pool.sort(key=lambda c: c.score, reverse=True)
        chosen: List[PoseCandidateWindow] = [pool.pop(0)]

        while pool and len(chosen) < desired_count:
            last_desc = chosen[-1].descriptor

            def mmr_score(cand: PoseCandidateWindow) -> float:
                cos_sim = float(np.dot(cand.descriptor, last_desc))
                visual_diff = 1.0 - max(-1.0, min(1.0, cos_sim))
                # Écart temporel pour éviter de répéter deux prises de la même pose
                time_sep = min(30.0, abs(cand.center_time - chosen[-1].center_time)) / 30.0
                return 0.50 * cand.score + 22.0 * visual_diff + 6.0 * time_sep

            best_next = max(pool, key=mmr_score)
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
                effort_score=round(float(np.clip(7.5 + candidate.score * 2.2, 7.0, 9.8)), 1),
                exercise_detected=candidate.pose_label,
                is_peak_contraction=True,
                reasoning="Pose verrouillée détectée par analyse de stabilité",
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
            score = float(np.clip(float(parsed.get("effort_score", 8.5)), 1.0, 10.0))
            pose_name = str(parsed.get("exercise_detected") or candidate.pose_label)
            reasoning = str(parsed.get("reasoning") or "Évaluation LLaVA")
            rot_needed = int(parsed.get("rotation_needed", 0) or 0)
            if rot_needed in (90, 180, 270):
                candidate.rotation_correction = (candidate.rotation_correction + rot_needed) % 360
                if rot_needed in (90, 270):
                    candidate.upright_aspect_ratio = (
                        "16:9" if candidate.upright_aspect_ratio == "9:16" else "9:16"
                    )
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
                effort_score=round(float(np.clip(7.5 + candidate.score * 2.2, 7.0, 9.8)), 1),
                exercise_detected=candidate.pose_label,
                is_peak_contraction=True,
                reasoning=f"Pose #{pose_index} sélectionnée par stabilité isométrique & contraste",
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
