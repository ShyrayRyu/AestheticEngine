"""
Moteur de Montage Vidéo & Synchronisation Beat-Cut via FFmpeg Natif (Sans Étirement, Sans Transition).
------------------------------------------------------------------------------------------------------
Pourquoi FFmpeg natif au lieu de MoviePy 1.0.3 pour le recadrage et la concaténation :
  - Les vidéos filmées au smartphone (iPhone, Xiaomi, Samsung) en vertical 9:16 sont stockées
    en `1920x1080` avec une métadonnée `Display Matrix: rotation = -90`. MoviePy 1.0.3 lit
    `1920x1080` de manière erronée et étire horizontalement l'image d'un facteur 1.78x.
  - FFmpeg applique automatiquement la rotation de la Display Matrix AVANT le graphe de filtres `-vf`.
  - Le filtre `crop='min(iw,ih*R)':'min(ih,iw/R)',scale=W:H:flags=lanczos,setsar=1` garantit
    mathématiquement un **rognage centré pur (SAR 1:1)** en `9:16` (1080x1920) ou `16:9` (1920x1080)
    sans **aucun étirement** des proportions musculaires.
  - La concaténation utilise des **coupes franches ("Hard Cuts", sans transition)** synchronisées
    à la frame près sur les timestamps des Beat Drops, exactement comme dans `exemple.mp4`.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import cv2

from app.core.config import settings
from app.models.schemas import BeatDropEvent, SelectedVideoSegment

logger = logging.getLogger(__name__)


class SpeedRampVideoEditor:
    """
    Assemble les segments de posing retenus sur la grille des Beat Drops avec :
      - Respect absolu des proportions (rognage centré 9:16 ou 16:9, zéro étirement)
      - Coupes franches ("Hard Cuts" sans transition) calées sur chaque beat drop
      - Mode "pose_cut" (vitesse fluide 1.0x fidèle à `exemple.mp4`) ou "speed_ramp".
    """

    def __init__(self, target_fps: int = 30) -> None:
        self.target_fps = target_fps

    def assemble_synced_timeline(
        self,
        segments: List[SelectedVideoSegment],
        beat_drops: List[BeatDropEvent],
        audio_path: Optional[Path],
        output_raw_video_path: Path,
        effective_duration: float = 21.13,
        aspect_ratio: Literal["9:16", "16:9"] = "9:16",
        editing_mode: Literal["pose_cut", "speed_ramp"] = "pose_cut",
        fast_speed_factor: float = 1.35,
        slow_speed_factor: float = 0.75,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> List[SelectedVideoSegment]:
        """
        Découpe chaque séquence de posing pour remplir exactement les intervalles
        musicaux `[0 -> drop_1, drop_1 -> drop_2, ..., drop_N -> effective_duration]`
        par coupes franches sans transition, puis multiplexe la piste audio.
        """
        if not segments:
            raise ValueError("Aucun segment vidéo disponible pour le montage.")

        # Définition de la résolution cible selon le format 9:16 ou 16:9
        if aspect_ratio == "16:9":
            target_w, target_h = 1920, 1080
        else:
            target_w, target_h = 1080, 1920

        # Construction de la liste des intervalles temporels sur la timeline musicale
        cut_points: List[float] = [0.0]
        for drop in beat_drops:
            if 0.8 <= drop.timestamp < effective_duration - 0.8:
                if drop.timestamp - cut_points[-1] >= 1.0:
                    cut_points.append(round(drop.timestamp, 3))
        cut_points.append(round(effective_duration, 3))

        # Si aucun beat drop intermédiaire n'existe, on découpe toutes les ~3.0s
        if len(cut_points) < 3:
            step = 3.0
            cut_points = [round(float(t), 3) for t in list(range(0, int(effective_duration), int(step)))]
            if cut_points[-1] < effective_duration:
                cut_points.append(round(effective_duration, 3))

        intervals: List[Tuple[float, float]] = [
            (cut_points[i], cut_points[i + 1]) for i in range(len(cut_points) - 1)
        ]

        temp_dir = output_raw_video_path.parent / "parts"
        temp_dir.mkdir(parents=True, exist_ok=True)
        part_files: List[Path] = []
        synced_metadata: List[SelectedVideoSegment] = []

        for idx, (t_start_tl, t_end_tl) in enumerate(intervals):
            slot_duration = round(max(0.8, t_end_tl - t_start_tl), 3)
            seg = segments[idx % len(segments)]
            src_path = Path(seg.source_video)
            src_duration = self._get_video_duration(src_path)
            crop_expr = self._build_segment_crop_filter(seg=seg, aspect_ratio=aspect_ratio)

            if progress_callback:
                pct = int(((idx + 1) / len(intervals)) * 90)
                rot_info = (
                    f", redressé {seg.rotation_correction}°"
                    if seg.rotation_correction != 0
                    else ""
                )
                progress_callback(
                    f"Découpe plan #{idx + 1}/{len(intervals)} ({seg.pose_label}) "
                    f"→ [{t_start_tl:.2f}s - {t_end_tl:.2f}s] ({aspect_ratio} bien cadré{rot_info})",
                    pct,
                )

            part_path = temp_dir / f"part_{idx:03d}.mp4"

            if editing_mode == "pose_cut" or idx == 0:
                # Mode Posing 1.0x (Style exemple.mp4) :
                # On centre la fenêtre autour du verrouillage de pose pour capter la mise en tension
                # suivie du maintien de la contraction pendant toute la durée `slot_duration`.
                if idx == 0 and seg.pose_label.startswith("Intro"):
                    clip_start = max(0.0, seg.start_time)
                else:
                    # On démarre ~35% avant le center_time pour voir l'entrée dans la pose puis le lock
                    clip_start = max(0.0, seg.peak_time_in_source - 0.35 * slot_duration)

                if clip_start + slot_duration > src_duration - 0.1:
                    clip_start = max(0.0, src_duration - slot_duration - 0.1)

                clip_end = round(clip_start + slot_duration, 3)

                self._render_trimmed_part_1x(
                    src_path=src_path,
                    part_path=part_path,
                    start_sec=clip_start,
                    duration_sec=slot_duration,
                    crop_expr=crop_expr,
                )
                fast_f, slow_f = 1.0, 1.0
            else:
                # Mode Speed Ramping : approche accélérée (40% du slot) + maintien ralenti (60% du slot)
                fast_tl_dur = round(slot_duration * 0.40, 3)
                slow_tl_dur = round(slot_duration - fast_tl_dur, 3)

                src_pre_dur = fast_tl_dur * fast_speed_factor
                src_post_dur = slow_tl_dur * slow_speed_factor

                peak_ts = min(
                    max(src_pre_dur + 0.2, seg.peak_time_in_source),
                    max(src_pre_dur + 0.2, src_duration - src_post_dur - 0.1),
                )
                clip_start = max(0.0, peak_ts - src_pre_dur)
                clip_end = min(src_duration, peak_ts + src_post_dur)

                self._render_speed_ramped_part(
                    src_path=src_path,
                    part_path=part_path,
                    pre_start=clip_start,
                    peak_ts=peak_ts,
                    fast_tl_dur=fast_tl_dur,
                    slow_tl_dur=slow_tl_dur,
                    fast_factor=fast_speed_factor,
                    slow_factor=slow_speed_factor,
                    crop_expr=crop_expr,
                    temp_dir=temp_dir,
                    part_idx=idx,
                )
                fast_f, slow_f = fast_speed_factor, slow_speed_factor

            part_files.append(part_path)
            synced_metadata.append(
                seg.model_copy(
                    update={
                        "start_time": round(clip_start, 3),
                        "end_time": round(clip_end, 3),
                        "peak_time_in_source": round(
                            clip_start + min(slot_duration * 0.35, 1.0), 3
                        ),
                        "timeline_start": round(t_start_tl, 3),
                        "timeline_end": round(t_end_tl, 3),
                        "target_beat_drop": round(t_start_tl, 3),
                        "speed_fast_factor": fast_f,
                        "speed_slow_factor": slow_f,
                    }
                )
            )

        # Concaténation FFmpeg sans aucune transition (Hard Cuts purs) + mixage de la piste audio
        if progress_callback:
            progress_callback(
                f"Assemblage final des {len(part_files)} plans en {target_w}x{target_h} (coupes franches)...",
                96,
            )

        self._concat_parts_with_audio(
            part_files=part_files,
            audio_path=audio_path,
            output_path=output_raw_video_path,
            total_duration=cut_points[-1],
        )

        return synced_metadata

    def _render_trimmed_part_1x(
        self,
        src_path: Path,
        part_path: Path,
        start_sec: float,
        duration_sec: float,
        crop_expr: str,
    ) -> None:
        """
        Découpe un sous-plan à vitesse réelle 1.0x avec rognage centré exact (`crop_expr`)
        et cadence d'images constante (`fps=30`), sans aucun étirement.
        """
        vf = f"{crop_expr},fps={self.target_fps},format=yuv420p"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration_sec:.3f}",
            "-i",
            str(src_path),
            "-an",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "17",
            "-video_track_timescale",
            "15360",
            str(part_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Erreur FFmpeg découpe plan : {res.stderr[-300:]}")

    def _render_speed_ramped_part(
        self,
        src_path: Path,
        part_path: Path,
        pre_start: float,
        peak_ts: float,
        fast_tl_dur: float,
        slow_tl_dur: float,
        fast_factor: float,
        slow_factor: float,
        crop_expr: str,
        temp_dir: Path,
        part_idx: int,
    ) -> None:
        """Génère un plan en deux phases (approche rapide + ralenti sur le beat drop)."""
        fast_part = temp_dir / f"sub_{part_idx:03d}_fast.mp4"
        slow_part = temp_dir / f"sub_{part_idx:03d}_slow.mp4"

        src_fast_dur = fast_tl_dur * fast_factor
        vf_fast = f"{crop_expr},setpts={1.0 / fast_factor:.4f}*PTS,fps={self.target_fps},format=yuv420p"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{pre_start:.3f}",
                "-t",
                f"{src_fast_dur:.3f}",
                "-i",
                str(src_path),
                "-an",
                "-vf",
                vf_fast,
                "-t",
                f"{fast_tl_dur:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "17",
                "-video_track_timescale",
                "15360",
                str(fast_part),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        src_slow_dur = slow_tl_dur * slow_factor
        vf_slow = f"{crop_expr},setpts={1.0 / slow_factor:.4f}*PTS,fps={self.target_fps},format=yuv420p"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{peak_ts:.3f}",
                "-t",
                f"{src_slow_dur:.3f}",
                "-i",
                str(src_path),
                "-an",
                "-vf",
                vf_slow,
                "-t",
                f"{slow_tl_dur:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "17",
                "-video_track_timescale",
                "15360",
                str(slow_part),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        concat_list = temp_dir / f"sub_{part_idx:03d}.txt"
        concat_list.write_text(
            f"file '{fast_part.resolve()}'\nfile '{slow_part.resolve()}'\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(part_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    def _concat_parts_with_audio(
        self,
        part_files: List[Path],
        audio_path: Optional[Path],
        output_path: Path,
        total_duration: float,
    ) -> None:
        """
        Concatène tous les sous-plans en coupes franches (sans transition).
        - Si une piste audio est fournie, elle est multiplexée sur `total_duration`.
        - Si aucune piste audio n'est fournie (mode sans musique), le son d'origine des
          rushs est supprimé et remplacé par une piste audio muette (volume strictement à 0).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        concat_file = output_path.parent / "concat_parts.txt"
        lines = [f"file '{p.resolve()}'" for p in part_files]
        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if audio_path is not None and audio_path.exists():
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-t",
                f"{total_duration:.3f}",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        else:
            # Mode sans musique : piste audio stéréo muette (son à 0) pour une compatibilité maximale
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-t",
                f"{total_duration:.3f}",
                "-c:v",
                "copy",
                "-af",
                "volume=0.0",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Erreur FFmpeg concaténation finale : {res.stderr[-300:]}")

    @staticmethod
    def _build_segment_crop_filter(
        seg: SelectedVideoSegment,
        aspect_ratio: Literal["9:16", "16:9"],
    ) -> str:
        """
        Construit la chaîne de filtres vidéo FFmpeg pour un segment donné :
          1. Redressement automatique dans le bon sens si l'IA a détecté une rotation
             (`90°` -> `transpose=1`, `270°` -> `transpose=2`, `180°` -> `hflip,vflip`).
          2. Rognage intelligent centré sur le buste/visage de l'athlète `(focus_x, focus_y)`
             sans aucun étirement (`setsar=1`) au format `9:16` (1080x1920) ou `16:9` (1920x1080).
        """
        rot = seg.rotation_correction % 360
        rot_prefix = ""
        if rot == 90:
            rot_prefix = "transpose=1,"
        elif rot == 270:
            rot_prefix = "transpose=2,"
        elif rot == 180:
            rot_prefix = "hflip,vflip,"

        fx = max(0.15, min(0.85, float(seg.focus_x)))
        fy = max(0.20, min(0.80, float(seg.focus_y)))

        if aspect_ratio == "16:9":
            crop_scale = (
                f"crop=w='min(iw,ih*16/9)':h='min(ih,iw*9/16)':"
                f"x='clip(iw*{fx:.3f}-ow/2,0,iw-ow)':y='clip(ih*{fy:.3f}-oh/2,0,ih-oh)',"
                f"scale=1920:1080:flags=lanczos,setsar=1"
            )
        else:
            crop_scale = (
                f"crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)':"
                f"x='clip(iw*{fx:.3f}-ow/2,0,iw-ow)':y='clip(ih*{fy:.3f}-oh/2,0,ih-oh)',"
                f"scale=1080:1920:flags=lanczos,setsar=1"
            )

        return f"{rot_prefix}{crop_scale}"

    @staticmethod
    def _get_video_duration(video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 10.0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            return float(n_frames / fps) if fps > 0 else 10.0
        finally:
            cap.release()
