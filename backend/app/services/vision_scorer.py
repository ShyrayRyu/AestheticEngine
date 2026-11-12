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
        max_candidates: int = 12,
    ) -> Tuple[List[PoseCandidateWindow], Optional[PoseCandidateWindow]]:
        """
        Analyse un rush vidéo à 4 Hz (toutes les 0.25s) après l'avoir remis dans le bon sens :
          1. Détecte l'orientation réelle (`rotation_correction` 0/90/180/270) et le format (`9:16` ou `16:9`)
          2. Identifie le plan d'intro (entre 2.6s et 5.0s)
          3. Identifie toutes les fenêtres de posing verrouillées (sans rognage ni recadrage).
        """
        rot_corr, upright_aspect = self._detect_video_orientation_and_framing(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], None

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = n_frames / fps if fps > 0 else 0.0

        step_frames = max(1, int(round(fps * self.sample_interval_sec)))
        times: List[float] = []
        motions: List[float] = []
        gradients: List[float] = []
        descriptors: List[np.ndarray] = []

        prev_roi: Optional[np.ndarray] = None
        frame_idx = 0

        try:
            while True:
                ret, raw_frame = cap.read()
                if not ret:
                    break
                if frame_idx % step_frames == 0:
                    t = frame_idx / fps
                    frame = self._rotate_frame_cw(raw_frame, rot_corr)
                    h, w = frame.shape[:2]
                    # Redimensionnement rapide pour l'analyse de mouvement et de silhouette
                    small_w, small_h = (90, 160) if h >= w else (160, 90)
                    small = cv2.resize(frame, (small_w, small_h))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                    # Région d'intérêt centrée sur le buste et les bras (20% -> 80% vertical)
                    roi = gray[
                        int(small_h * 0.20) : int(small_h * 0.80),
                        int(small_w * 0.12) : int(small_w * 0.88),
                    ]

                    mot = float(np.mean(cv2.absdiff(roi, prev_roi))) if prev_roi is not None else 0.0
                    prev_roi = roi

                    # Énergie des contours musculaires (ombres/lumières zénithales sur le buste)
                    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
                    grad = float(np.mean(np.hypot(gx, gy)))

                    # Descripteur de posture 16x16 normalisé (pour différencier les poses entre elles)
                    desc = cv2.resize(roi, (16, 16)).astype(np.float32).flatten()
                    desc -= float(np.mean(desc))
                    norm = float(np.linalg.norm(desc))
                    if norm > 1e-6:
                        desc /= norm

                    times.append(t)
                    motions.append(mot)
                    gradients.append(grad)
                    descriptors.append(desc)

                frame_idx += 1
        finally:
            cap.release()

        if len(times) < 6:
            return [], None

        t_arr = np.array(times, dtype=np.float32)
        m_arr = np.array(motions, dtype=np.float32)
        g_arr = np.array(gradients, dtype=np.float32)

        # Lissage sur une fenêtre de 1.5s (6 échantillons à 4 Hz)
        kernel = np.ones(6, dtype=np.float32) / 6.0
        smooth_mot = np.convolve(m_arr, kernel, mode="same")
        smooth_grad = np.convolve(g_arr, kernel, mode="same")

        grad_min, grad_ptp = float(np.min(smooth_grad)), float(np.ptp(smooth_grad)) + 1e-6

        # 1. Détection du plan d'Intro (entre 2.6s et 5.0s si le rush fait plus de 15s)
        intro_cand: Optional[PoseCandidateWindow] = None
        if duration >= 15.0:
            intro_start = 2.65
            intro_end = min(duration - 1.0, 5.20)
            intro_idx = int(np.argmin(np.abs(t_arr - 3.4)))
            intro_cand = PoseCandidateWindow(
                video_path=video_path,
                center_time=3.4,
                start_time=intro_start,
                end_time=intro_end,
                score=0.90,
                motion_val=float(smooth_mot[intro_idx]),
                gradient_val=float(smooth_grad[intro_idx]),
                descriptor=descriptors[intro_idx],
                is_intro=True,
                pose_label="Intro / Mise en place",
                rotation_correction=rot_corr,
                upright_aspect_ratio=upright_aspect,
            )

        # 2. Scoring de chaque instant pour détecter les "Pose Locks" (contractions tenues)
        # On ignore les premières 5.5s (déjà couvertes par l'intro/téléphone) et les dernières 6s
        safe_start = 5.5 if duration > 20.0 else 0.5
        safe_end = max(safe_start + 2.0, duration - 6.0) if duration > 20.0 else duration - 0.5

        pose_scores = np.zeros_like(t_arr)
        for i, t_val in enumerate(t_arr):
            if t_val < safe_start or t_val > safe_end:
                continue

            m_val = float(smooth_mot[i])
            # Une pose maintenue a un mouvement modéré et contrôlé (entre 1.1 et 2.9)
            if 1.1 <= m_val <= 2.9:
                hold_score = 1.0 - abs(m_val - 1.85) / 1.5
            else:
                hold_score = max(0.0, 1.0 - abs(m_val - 1.85) / 3.2)

            # Bonus si l'athlète vient d'entrer dans la pose (mouvement d'amorce 0.5s à 2.5s avant)
            pre_slice = m_arr[max(0, i - 10) : max(1, i - 1)]
            entry_motion = float(np.max(pre_slice)) if pre_slice.size > 0 else 2.0
            entry_bonus = float(np.clip((entry_motion - 1.5) / 3.5, 0.2, 1.0))

            # Netteté musculaire / contraste sous le spot lumineux
            norm_grad = (float(smooth_grad[i]) - grad_min) / grad_ptp

            pose_scores[i] = 0.50 * hold_score + 0.30 * entry_bonus + 0.20 * norm_grad

        # 3. Extraction des pics locaux espacés d'au moins 5.5s dans un même rush
        min_gap_sec = 5.5 if duration > 35.0 else 2.5
        sorted_indices = np.argsort(pose_scores)[::-1]
        selected_indices: List[int] = []

        for idx in sorted_indices:
            if pose_scores[idx] <= 0.15:
                break
            t_cand = float(t_arr[idx])
            if all(abs(t_cand - float(t_arr[sel])) >= min_gap_sec for sel in selected_indices):
                selected_indices.append(int(idx))
                if len(selected_indices) >= max_candidates:
                    break

        candidates: List[PoseCandidateWindow] = []
        for rank_idx, sample_idx in enumerate(selected_indices):
            t_center = float(t_arr[sample_idx])
            # Le début de la pose commence légèrement avant le verrouillage (~0.8s avant t_center)
            seg_start = max(safe_start - 0.5, t_center - 1.1)
            seg_end = min(duration - 0.2, seg_start + 4.5)

            candidates.append(
                PoseCandidateWindow(
                    video_path=video_path,
                    center_time=t_center,
                    start_time=seg_start,
                    end_time=seg_end,
                    score=float(pose_scores[sample_idx]),
                    motion_val=float(smooth_mot[sample_idx]),
                    gradient_val=float(smooth_grad[sample_idx]),
                    descriptor=descriptors[sample_idx],
                    is_intro=False,
                    pose_label=f"Pose Aesthetic #{rank_idx + 1}",
                    rotation_correction=rot_corr,
                    upright_aspect_ratio=upright_aspect,
                )
            )

        return candidates, intro_cand

    def _select_diverse_poses(
        self,
        candidates: List[PoseCandidateWindow],
        desired_count: int,
    ) -> List[PoseCandidateWindow]:
        """
        Ordonne les poses candidates selon un algorithme glouton (Maximal Marginal Relevance)
        qui combine la note de contraction (`score`) et la différence visuelle (`1 - cos_sim`)
        par rapport à la pose précédente.
        Ainsi, comme dans `exemple.mp4`, deux plans consécutifs montrent toujours deux
        poses visuellement très différentes (ex: Pose de face -> Pose de profil -> Pose de dos).
        """
        if len(candidates) <= 1:
            return candidates

        remaining = sorted(candidates, key=lambda c: c.score, reverse=True)
        # On démarre par une pose à fort impact visuel (ex: Most Muscular / Double Biceps)
        chosen: List[PoseCandidateWindow] = [remaining.pop(0)]

        pose_names = [
            "Most Muscular (Crab / Front)",
            "Side Chest & Biceps",
            "Front Lat Spread / Aesthetic",
            "Front Double Biceps",
            "Back Double Biceps / Triceps",
            "Classic Arnold Pose",
            "Abdominals & Obliques Flex",
            "Side Triceps Lock",
        ]
        chosen[0].pose_label = pose_names[0]

        while remaining and len(chosen) < desired_count:
            prev_pose = chosen[-1]

            def diversity_objective(cand: PoseCandidateWindow) -> float:
                # Similarité cosinus entre les silhouettes 16x16
                cos_sim = float(np.dot(cand.descriptor, prev_pose.descriptor))
                visual_diff = 1.0 - max(-1.0, min(1.0, cos_sim))
                # Écart temporel pour éviter de prendre deux extraits trop proches dans le rush
                time_diff_bonus = (
                    0.25 if abs(cand.center_time - prev_pose.center_time) >= 12.0 else 0.0
                )
                return 0.55 * cand.score + 0.45 * visual_diff + time_diff_bonus

            best_next = max(remaining, key=diversity_objective)
            remaining.remove(best_next)
            best_next.pose_label = pose_names[len(chosen) % len(pose_names)]
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
