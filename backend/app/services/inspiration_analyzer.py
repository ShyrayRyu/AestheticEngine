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
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np

from app.models.schemas import (
    BeatDropEvent,
    InspirationAnalysisResult,
    RenderStyleCategory,
)

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
        step = max(1, int(round(fps / 15.0)))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % step != 0:
                idx += 1
                continue

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

    @staticmethod
    def _extract_thumbnail(video_path: Path, output_thumb: Path, timestamp: float = 2.5) -> bool:
        """Extrait une image JPEG d'aperçu de la vidéo d'inspiration."""
        output_thumb.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.5, timestamp):.2f}",
            "-i",
            str(video_path),
            "-vf",
            "scale=360:640:force_original_aspect_ratio=decrease",
            "-vframes",
            "1",
            "-q:v",
            "3",
            str(output_thumb),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode == 0 and output_thumb.exists() and output_thumb.stat().st_size > 512

    @staticmethod
    def _analyze_visual_tone(video_path: Path) -> Tuple[str, float, float]:
        """
        Analyse l'ambiance lumineuse, le contraste et la saturation pour suggérer
        l'étalonnage adapté ('dark_aesthetic', 'natural_aesthetic', ou 'obsidian_monochrome').
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return "natural_aesthetic", 100.0, 50.0

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_frames < int(fps * 2):
            cap.release()
            return "natural_aesthetic", 100.0, 50.0

        sample_pts = np.linspace(int(fps * 0.8), n_frames - int(fps * 0.8), 5, dtype=int)
        lumas: List[float] = []
        contrasts: List[float] = []
        saturations: List[float] = []

        for f_idx in sample_pts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            s = hsv[:, :, 1]
            lumas.append(float(np.mean(v)))
            contrasts.append(float(np.std(v)))
            saturations.append(float(np.mean(s)))

        cap.release()
        mean_luma = float(np.mean(lumas)) if lumas else 100.0
        mean_contrast = float(np.mean(contrasts)) if contrasts else 50.0
        mean_sat = float(np.mean(saturations)) if saturations else 50.0

        if mean_sat < 22.0:
            return "obsidian_monochrome", mean_luma, mean_contrast
        if mean_luma < 85.0 and mean_contrast > 40.0:
            return "dark_aesthetic", mean_luma, mean_contrast
        return "natural_aesthetic", mean_luma, mean_contrast

    @staticmethod
    def _estimate_audio_bpm(audio_path: Path) -> Optional[float]:
        """Estime rapidement le BPM de la piste audio de l'exemple via Librosa."""
        if not audio_path.exists():
            return None
        try:
            import librosa

            y, sr = librosa.load(str(audio_path), sr=11025, duration=18.0, mono=True)
            if y.size > sr * 2:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm_val = float(np.atleast_1d(tempo)[0])
                if 50.0 <= bpm_val <= 230.0:
                    return round(bpm_val, 1)
        except Exception as exc:
            logger.debug("Échec estimation BPM sur %s : %s", audio_path.name, exc)
        return None

    def categorize_inspirations(
        self,
        inspiration_paths: List[Path],
        project_id: str,
        thumbs_dir: Path,
        audio_dir: Path,
    ) -> Tuple[List[InspirationAnalysisResult], List[RenderStyleCategory]]:
        """
        Analyse une collection de vidéos d'exemples / inspirations, extrait leurs
        caractéristiques clés (format, coupes, musique, look) et construit des
        catégories de rendu personnalisées que l'utilisateur peut choisir.
        """
        results: List[InspirationAnalysisResult] = []
        categories: List[RenderStyleCategory] = []

        for idx, insp_path in enumerate(inspiration_paths):
            audio_dest = audio_dir / f"inspiration_{idx}.mp3"
            thumb_dest = thumbs_dir / f"inspiration_thumb_{idx}.jpg"

            try:
                # 1. Analyse de base de la structure vidéo et cuts
                analysis = self.analyze_inspiration_video(
                    inspiration_path=insp_path,
                    extracted_audio_dest=audio_dest,
                )

                # 2. Extraction vignette d'aperçu
                thumb_time = min(max(1.0, analysis.duration * 0.25), max(1.0, analysis.duration - 0.5))
                if self._extract_thumbnail(insp_path, thumb_dest, thumb_time):
                    analysis.thumbnail_url = (
                        f"/api/v1/projects/{project_id}/inspiration-thumbnail/{idx}"
                    )

                # 3. Étalonnage et ambiance visuelle
                lut_preset, _, _ = self._analyze_visual_tone(insp_path)

                # 4. Analyse musicale de l'exemple
                bpm: Optional[float] = None
                if analysis.audio_extracted and audio_dest.exists():
                    bpm = self._estimate_audio_bpm(audio_dest)
                analysis.bpm = bpm

                # 5. Pacing et densité des coupes
                num_plans = len(analysis.scene_cuts) + 1
                avg_cut = round(analysis.duration / max(1, num_plans), 2)

                if avg_cut < 2.2:
                    pacing = "Rapide"
                    drop_sens = 0.85
                    rhythm_desc = "Coupes franches très nerveuses et rythmées"
                elif avg_cut <= 3.8:
                    pacing = "Modéré"
                    drop_sens = 0.65
                    rhythm_desc = "Rythme de posing équilibré avec bonne tenue des poses"
                else:
                    pacing = "Posé"
                    drop_sens = 0.40
                    rhythm_desc = "Maintiens isométriques longs et posés sous la lumière"

                if lut_preset == "dark_aesthetic":
                    look_desc = "Look Dark Aesthetic (fort contraste & striations accentuées)"
                    lut_badge = "Dark Aesthetic"
                elif lut_preset == "obsidian_monochrome":
                    look_desc = "Look Monochrome Obsidian (noir & blanc sculpté)"
                    lut_badge = "Monochrome"
                else:
                    look_desc = "Look Natural Aesthetic (couleurs fidèles et netteté pure)"
                    lut_badge = "Natural Look"

                bpm_badge = (
                    f"{int(bpm)} BPM"
                    if bpm
                    else ("Son inclus" if analysis.audio_extracted else "Muet")
                )
                category_id = f"custom_example_{idx}"
                analysis.suggested_category_id = category_id

                clean_name = (
                    insp_path.stem.replace("media_", "Exemple ")
                    .replace("_", " ")
                    .title()
                )
                cat_title = f"Style #{idx + 1} : {clean_name} ({pacing})"
                cat_desc = (
                    f"Calqué sur '{insp_path.name}' : format {analysis.detected_aspect_ratio}, "
                    f"{num_plans} plans (~{avg_cut}s par pose), {look_desc}. {rhythm_desc}."
                )
                cat_badge = (
                    f"{pacing} • {analysis.detected_aspect_ratio} • {lut_badge} • {bpm_badge}"
                )

                categories.append(
                    RenderStyleCategory(
                        id=category_id,
                        name=cat_title,
                        description=cat_desc,
                        badge=cat_badge,
                        pacing_label=pacing,
                        average_cut_duration=avg_cut,
                        target_duration=analysis.duration,
                        aspect_ratio=analysis.detected_aspect_ratio,
                        editing_mode="pose_cut",
                        lut_preset=lut_preset,
                        drop_sensitivity=drop_sens,
                        has_audio=analysis.audio_extracted,
                        audio_url=(
                            f"/api/v1/projects/{project_id}/inspiration-audio/{idx}"
                            if analysis.audio_extracted
                            else None
                        ),
                        bpm=bpm,
                        cut_timestamps=analysis.scene_cuts,
                        source_example_filename=insp_path.name,
                        thumbnail_url=analysis.thumbnail_url,
                        is_from_user_example=True,
                    )
                )
                results.append(analysis)
            except Exception as exc:
                logger.warning(
                    "Échec de l'analyse de l'inspiration %s : %s",
                    insp_path.name,
                    exc,
                )
                continue

        # Catégories esthétiques prédéfinies complémentaires
        curated_presets = [
            RenderStyleCategory(
                id="preset_breath_divinity",
                name="Breath Divinity (CapCut Velocity & Dark Aesthetic)",
                description=(
                    "Édit cinématique TikTok de haut niveau calqué sur le style 'Breath Divinity' : "
                    "Velocity ramping dynamique, flash blanc d'impact et micro-zoom punch sur les beat drops, "
                    "aberration chromatique et étalonnage acier sombre aux striations musculaires chirurgicales."
                ),
                badge="CapCut Velocity • Flash & Zoom FX • Look Acier Sombre",
                pacing_label="Dynamique",
                average_cut_duration=1.8,
                target_duration=19.5,
                aspect_ratio="16:9",
                editing_mode="breath_divinity",
                lut_preset="breath_divinity",
                drop_sensitivity=0.85,
                has_audio=False,
                audio_url=None,
                bpm=90.0,
                cut_timestamps=[],
                source_example_filename="breath_divinity.mp4",
                thumbnail_url=None,
                is_from_user_example=False,
            ),
            RenderStyleCategory(
                id="preset_dark_phonk",
                name="Dark Aesthetic Phonk (Coupes Franches Rapides)",
                description=(
                    "TikTok musculation sombre et percutant. Coupes franches calées sur "
                    "les kicks de basse toutes les ~2.0s, contraste profond pour faire "
                    "ressortir les veines et striations musculaires."
                ),
                badge="Rapide • 9:16 • Dark Aesthetic • 130 BPM",
                pacing_label="Rapide",
                average_cut_duration=2.0,
                target_duration=21.5,
                aspect_ratio="9:16",
                editing_mode="pose_cut",
                lut_preset="dark_aesthetic",
                drop_sensitivity=0.80,
                has_audio=False,
                audio_url=None,
                bpm=130.0,
                cut_timestamps=[],
                source_example_filename=None,
                thumbnail_url=None,
                is_from_user_example=False,
            ),
            RenderStyleCategory(
                id="preset_pure_posing",
                name="Aesthetic Posing Flow (Maintiens Longs & Épurés)",
                description=(
                    "Mise en valeur maximale des poses de bodybuilding. Plans tenus "
                    "pendant 3.5s à 4.5s sur les poses isométriques verrouillées, "
                    "étalonnage naturel valorisant le volume musculaire."
                ),
                badge="Posé • 9:16 • Natural Look • 105 BPM",
                pacing_label="Posé",
                average_cut_duration=3.6,
                target_duration=24.0,
                aspect_ratio="9:16",
                editing_mode="pose_cut",
                lut_preset="natural_aesthetic",
                drop_sensitivity=0.45,
                has_audio=False,
                audio_url=None,
                bpm=105.0,
                cut_timestamps=[],
                source_example_filename=None,
                thumbnail_url=None,
                is_from_user_example=False,
            ),
            RenderStyleCategory(
                id="preset_speed_ramp",
                name="Speed Ramping Agressif (Amorce 1.35x + Ralenti 0.75x)",
                description=(
                    "Style très cinématique : accélération dynamique lors de l'amorce de "
                    "la contraction puis ralenti lourd au pic de l'effort sur chaque drop."
                ),
                badge="Dynamique • 9:16 • Speed Ramping • 120 BPM",
                pacing_label="Dynamique",
                average_cut_duration=2.5,
                target_duration=18.5,
                aspect_ratio="9:16",
                editing_mode="speed_ramp",
                lut_preset="dark_aesthetic",
                drop_sensitivity=0.65,
                has_audio=False,
                audio_url=None,
                bpm=120.0,
                cut_timestamps=[],
                source_example_filename=None,
                thumbnail_url=None,
                is_from_user_example=False,
            ),
        ]

        categories.extend(curated_presets)
        return results, categories
