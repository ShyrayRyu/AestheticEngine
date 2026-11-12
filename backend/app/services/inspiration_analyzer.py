"""
Service d'Analyse de Vidéo d'Inspiration / Référence (`inspiration_analyzer.py`).
---------------------------------------------------------------------------------
Lorsqu'un utilisateur dépose une vidéo TikTok d'inspiration (ex: `exemple.mp4`),
ce service analyse automatiquement :
  1. Le format réel d'affichage (`9:16` vertical ou `16:9` paysage) après rotation.
  2. La durée exacte de la vidéo de référence.
  3. Les timestamps exacts de chaque changement de plan ("Hard Cuts" / Scene Cuts)
     par détection de sauts d'histogramme et de différence inter-trames (OpenCV),
     afin de reproduire exactement le même rythme de découpe sur les rushs de l'utilisateur.
  4. L'extraction automatique de la bande-son de la vidéo d'inspiration via FFmpeg
     (si l'utilisateur souhaite reprendre le son de l'inspiration sans avoir à
     télécharger un MP3 séparé).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np

from app.models.schemas import BeatDropEvent, InspirationAnalysisResult

logger = logging.getLogger(__name__)


class InspirationVideoAnalyzer:
    """
    Extrait la structure de montage (format, durée, timestamps des cuts, piste audio)
    d'une vidéo TikTok d'inspiration.
    """

    def __init__(self, min_cut_spacing_sec: float = 1.1) -> None:
        self.min_cut_spacing_sec = min_cut_spacing_sec

    def analyze_inspiration_video(
        self,
        inspiration_path: Path,
        extracted_audio_dest: Optional[Path] = None,
    ) -> InspirationAnalysisResult:
        """
        Analyse complète d'une vidéo d'inspiration (.mp4, .mov).
        """
        if not inspiration_path.exists():
            raise FileNotFoundError(f"Vidéo d'inspiration introuvable : {inspiration_path}")

        cap = cv2.VideoCapture(str(inspiration_path))
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo d'inspiration : {inspiration_path.name}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = round(float(n_frames / fps), 3) if fps > 0 else 15.0

        # Lecture de la 1re frame pour connaître le ratio réel d'affichage (après rotation)
        ret, first_frame = cap.read()
        if ret and first_frame is not None:
            h, w = first_frame.shape[:2]
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)

        detected_aspect: Literal["9:16", "16:9"] = "9:16" if h >= w else "16:9"

        # Rembobinage pour scanner toutes les transitions franches (Hard Cuts)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cut_timestamps = self._detect_scene_cuts(cap=cap, fps=fps, duration=duration)
        cap.release()

        # Extraction optionnelle de la piste audio de la vidéo d'inspiration
        audio_extracted = False
        if extracted_audio_dest is not None:
            audio_extracted = self._extract_audio_stream(
                video_path=inspiration_path,
                output_audio_path=extracted_audio_dest,
            )

        logger.info(
            "Inspiration '%s' analysée : Format=%s | Durée=%.2fs | %d coupes détectées %s | Audio extrait=%s",
            inspiration_path.name,
            detected_aspect,
            duration,
            len(cut_timestamps),
            cut_timestamps,
            audio_extracted,
        )

        return InspirationAnalysisResult(
            filename=inspiration_path.name,
            duration=duration,
            detected_aspect_ratio=detected_aspect,
            scene_cuts=cut_timestamps,
            audio_extracted=audio_extracted,
        )

    @staticmethod
    def cuts_to_beat_drops(scene_cuts: List[float]) -> List[BeatDropEvent]:
        """Convertit les timestamps de coupe de la vidéo d'inspiration en objets BeatDropEvent."""
        drops: List[BeatDropEvent] = []
        prev_t = 0.0
        for cut_t in scene_cuts:
            drops.append(
                BeatDropEvent(
                    timestamp=round(float(cut_t), 3),
                    intensity=0.90,
                    buildup_start=round(prev_t, 3),
                    sub_bass_energy=0.85,
                )
            )
            prev_t = float(cut_t)
        return drops

    def _detect_scene_cuts(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        duration: float,
    ) -> List[float]:
        """
        Détecte les coupes franches (changements de plans) dans la vidéo d'inspiration
        en combinant :
          - La différence absolue pixel à pixel sur une grille basse résolution (luminance + structure)
          - La distance de corrélation d'histogramme HSV entre deux frames consécutives
          - Un seuillage adaptatif (z-score local) pour isoler les vrais changements de pose/plan
            tout en ignorant les simples mouvements continus.
        """
        frame_scores: List[float] = []
        frame_times: List[float] = []

        prev_small: Optional[np.ndarray] = None
        prev_hist: Optional[np.ndarray] = None
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            small = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)

            if prev_small is not None and prev_hist is not None:
                luma_diff = float(np.mean(cv2.absdiff(gray, prev_small)))
                hist_dist = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                # Score combiné de discontinuité visuelle inter-trames
                combined_jump = luma_diff + (hist_dist * 25.0)
                frame_scores.append(combined_jump)
                frame_times.append(idx / fps)

            prev_small = gray
            prev_hist = hist
            idx += 1

        if len(frame_scores) < 10:
            return []

        scores_arr = np.array(frame_scores, dtype=np.float32)
        times_arr = np.array(frame_times, dtype=np.float32)

        # Calcul du rapport entre chaque saut inter-trames et la médiane locale (±15 frames)
        # Lors d'un "Hard Cut", la différence sur 1 frame saute brutalement de > 2.5x la moyenne locale
        median_val = float(np.median(scores_arr))
        std_val = float(np.std(scores_arr))
        global_threshold = max(9.5, median_val + 2.2 * std_val)

        candidate_indices: List[int] = []
        win = max(5, int(round(fps * 0.4)))

        for i in range(win, len(scores_arr) - win):
            t_val = float(times_arr[i])
            if t_val < 0.7 or t_val > duration - 0.7:
                continue

            val = float(scores_arr[i])
            if val < global_threshold:
                continue

            # Doit être le maximum strict dans son voisinage et nettement supérieur au mouvement continu
            neighborhood = np.concatenate([scores_arr[i - win : i - 1], scores_arr[i + 2 : i + win]])
            local_med = float(np.median(neighborhood)) if neighborhood.size > 0 else median_val

            if val >= float(np.max(scores_arr[i - 3 : i + 4])) and val >= local_med * 2.1:
                candidate_indices.append(i)

        # Filtrage par espacement minimal (`min_cut_spacing_sec`) en privilégiant les sauts les plus forts
        candidate_indices.sort(key=lambda idx_c: float(scores_arr[idx_c]), reverse=True)
        selected_cuts: List[float] = []

        for idx_c in candidate_indices:
            t_cand = round(float(times_arr[idx_c]), 3)
            if all(abs(t_cand - existing) >= self.min_cut_spacing_sec for existing in selected_cuts):
                selected_cuts.append(t_cand)
                if len(selected_cuts) >= 16:
                    break

        selected_cuts.sort()
        return selected_cuts

    @staticmethod
    def _extract_audio_stream(video_path: Path, output_audio_path: Path) -> bool:
        """
        Extrait la piste audio d'une vidéo d'inspiration au format MP3 via FFmpeg.
        Retourne `True` si une piste audio valide a été extraite, `False` si la vidéo est muette.
        """
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_audio_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode == 0 and output_audio_path.exists() and output_audio_path.stat().st_size > 1024
