"""
Service d'analyse musicale et de détection des points de coupe ("Beat Cuts / Drops") via Librosa.
------------------------------------------------------------------------------------------------
Optimisé pour les montages TikTok de Posing / Musculation (style `exemple.mp4`) :
  1. Détecte le 1er impact musical franc après l'intro (autour de 1.6s - 2.4s)
     pour caler la coupe entre le plan d'introduction (mise en place) et la 1re pose.
  2. Sélectionne ensuite les temps forts / changements de mesures (toutes les 1.8s à 4.2s)
     quantifiés sur la grille de battements (`beat_track`) avec une forte énergie
     percussive et Sub-Bass (20 Hz - 150 Hz).
  3. Termine le montage sur un temps fort proche de la durée cible (`target_duration`,
     par défaut ~21.2s comme sur les formats TikTok Aesthetic de référence).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.models.schemas import AudioAnalysisResult, BeatDropEvent

logger = logging.getLogger(__name__)


class AudioBeatDropAnalyzer:
    """
    Analyseur audio basé sur Librosa pour extraire le BPM, la grille rythmique
    et les timestamps exacts des coupes franches ("Beat Cuts / Drops") sans transition.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def analyze_track(
        self,
        audio_path: Union[str, Path],
        sensitivity: float = 0.65,
        target_duration: float = 21.5,
    ) -> AudioAnalysisResult:
        """
        Analyse un fichier audio et génère la liste des points de coupe (Beat Drops)
        entre `0.0s` et `effective_duration`.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier audio introuvable : {path}")

        logger.info("Chargement de la piste audio avec Librosa : %s", path.name)

        y, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        full_duration = float(librosa.get_duration(y=y, sr=sr))
        if full_duration <= 1.0 or y.size == 0:
            raise ValueError("La piste audio est trop courte ou vide.")

        effective_duration = min(full_duration, max(6.0, float(target_duration)))

        # Séparation Harmonique / Percussive pour isoler les kicks et attaques franches
        _, y_percussive = librosa.effects.hpss(y, margin=(1.0, 2.5))

        onset_env = librosa.onset.onset_strength(
            y=y_percussive,
            sr=sr,
            hop_length=self.hop_length,
            aggregate=np.median,
        )

        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
        )
        bpm_value = float(np.ndim(tempo) and np.asarray(tempo).flat[0] or tempo)
        beat_timestamps = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=self.hop_length
        ).tolist()

        sub_bass_env = self._compute_sub_bass_energy(y=y, sr=sr)
        contrast_curve = self._compute_drop_contrast(
            sub_bass_env=sub_bass_env,
            onset_env=onset_env,
            sr=sr,
        )

        n_frames = min(len(onset_env), len(sub_bass_env), len(contrast_curve))
        onset_norm = self._normalize_array(onset_env[:n_frames])
        bass_norm = self._normalize_array(sub_bass_env[:n_frames])
        contrast_norm = self._normalize_array(contrast_curve[:n_frames])

        composite_score = self._normalize_array(
            0.40 * onset_norm + 0.35 * bass_norm + 0.25 * contrast_norm
        )

        # Alignement de la durée finale sur le beat le plus proche de target_duration
        beats_in_window = [b for b in beat_timestamps if b <= effective_duration + 0.4]
        if beats_in_window and effective_duration < full_duration - 0.5:
            snapped_end = min(
                beats_in_window, key=lambda b: abs(b - effective_duration)
            )
            if snapped_end >= 6.0:
                effective_duration = float(round(snapped_end, 3))

        beat_drops = self._select_posing_cut_points(
            composite_score=composite_score,
            bass_norm=bass_norm,
            beat_timestamps=beat_timestamps,
            sr=sr,
            effective_duration=effective_duration,
            sensitivity=sensitivity,
        )

        logger.info(
            "Analyse audio terminée : BPM=%.1f | Durée utile=%.2fs | %d coupes (Beat Drops) détectées",
            bpm_value,
            effective_duration,
            len(beat_drops),
        )

        return AudioAnalysisResult(
            bpm=round(bpm_value, 1),
            duration=round(full_duration, 2),
            effective_duration=round(effective_duration, 3),
            beat_timestamps=[round(float(t), 3) for t in beat_timestamps if t <= effective_duration],
            beat_drops=beat_drops,
        )

    def _select_posing_cut_points(
        self,
        composite_score: np.ndarray,
        bass_norm: np.ndarray,
        beat_timestamps: List[float],
        sr: int,
        effective_duration: float,
        sensitivity: float,
    ) -> List[BeatDropEvent]:
        """
        Construit la séquence idéale de points de coupe pour un TikTok de Posing :
          - Coupe #1 (Drop d'entrée) : entre 1.5s et 2.4s (fin de l'intro, entrée de la 1re pose)
          - Coupes suivantes : tous les 1.8s à 4.0s sur les temps forts musicaux (beats à forte attaque)
            afin d'obtenir des plans de pose lisibles, intenses et parfaitement calés sur le son.
        """
        clamped_sens = float(np.clip(sensitivity, 0.1, 0.95))
        # Espacement minimal et maximal entre deux coupes selon la sensibilité
        min_gap = 2.6 - clamped_sens * 0.9   # ~2.0s pour sens=0.65 (comme exemple.mp4 : 1.83s min)
        max_gap = 4.6 - clamped_sens * 1.1   # ~3.9s pour sens=0.65 (comme exemple.mp4 : 4.17s max)

        # Grille de candidats : tous les beats détectés par Librosa + les pics d'onset locaux
        candidates = sorted(
            t for t in beat_timestamps if 1.45 <= t <= effective_duration - 1.5
        )
        if not candidates:
            # Fallback si aucun beat n'est détecté : grille régulière toutes les 2.8s
            step = 2.8
            candidates = list(np.arange(1.9, max(2.5, effective_duration - 1.5), step))

        def score_at_time(t_sec: float) -> float:
            f_idx = int(round((t_sec * sr) / self.hop_length))
            f_idx = max(0, min(len(composite_score) - 1, f_idx))
            # Max local dans une fenêtre de ±3 trames (~70 ms)
            w_start = max(0, f_idx - 3)
            w_end = min(len(composite_score), f_idx + 4)
            return float(np.max(composite_score[w_start:w_end]))

        def bass_at_time(t_sec: float) -> float:
            f_idx = int(round((t_sec * sr) / self.hop_length))
            f_idx = max(0, min(len(bass_norm) - 1, f_idx))
            return float(bass_norm[f_idx])

        selected_times: List[float] = []

        # 1. Premier point de coupe (Drop d'intro entre 1.50s et 2.50s, ex: 1.90s dans exemple.mp4)
        intro_candidates = [t for t in candidates if 1.50 <= t <= 2.55]
        if intro_candidates:
            first_cut = max(intro_candidates, key=score_at_time)
            selected_times.append(first_cut)
        elif candidates:
            selected_times.append(candidates[0])

        # 2. Progression séquentielle de coupe en coupe jusqu'à la fin de la durée cible
        cursor = selected_times[-1] if selected_times else 0.0
        while cursor + min_gap < effective_duration - 1.6:
            window_cands = [
                t
                for t in candidates
                if (cursor + min_gap) <= t <= min(effective_duration - 1.5, cursor + max_gap)
            ]
            if window_cands:
                # Favorise le beat ayant le meilleur score d'impact (onset + sub-bass + contraste)
                best_next = max(window_cands, key=score_at_time)
                selected_times.append(best_next)
                cursor = best_next
            else:
                # Si aucun beat n'est dans la fenêtre [min_gap, max_gap], on prend le suivant disponible
                after_cands = [t for t in candidates if t > cursor + min_gap]
                if not after_cands:
                    break
                selected_times.append(after_cands[0])
                cursor = after_cands[0]

        beat_drops: List[BeatDropEvent] = []
        prev_time = 0.0
        for t_cut in selected_times:
            intensity = float(np.clip(score_at_time(t_cut), 0.15, 1.0))
            sub_bass = float(np.clip(bass_at_time(t_cut), 0.05, 1.0))
            beat_drops.append(
                BeatDropEvent(
                    timestamp=round(float(t_cut), 3),
                    intensity=round(intensity, 3),
                    buildup_start=round(prev_time, 3),
                    sub_bass_energy=round(sub_bass, 3),
                )
            )
            prev_time = float(t_cut)

        return beat_drops

    def _compute_sub_bass_energy(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Calcule l'énergie RMS dans la bande 20 Hz - 150 Hz."""
        n_fft = 2048
        stft_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        bass_mask = (freqs >= 20.0) & (freqs <= 150.0)
        if not np.any(bass_mask):
            return np.zeros(stft_mag.shape[1], dtype=np.float32)
        bass_power = np.sqrt(np.mean(stft_mag[bass_mask, :] ** 2, axis=0))
        return gaussian_filter1d(bass_power, sigma=1.2)

    def _compute_drop_contrast(
        self,
        sub_bass_env: np.ndarray,
        onset_env: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Mesure le saut dynamique d'énergie entre l'instant précédant et suivant chaque trame."""
        n = min(len(sub_bass_env), len(onset_env))
        combined = 0.55 * self._normalize_array(sub_bass_env[:n]) + 0.45 * self._normalize_array(
            onset_env[:n]
        )
        pre_frames = max(3, int((0.6 * sr) / self.hop_length))
        post_frames = max(3, int((0.8 * sr) / self.hop_length))
        contrast = np.zeros(n, dtype=np.float32)
        eps = 0.08
        for i in range(pre_frames, n - post_frames):
            pre_mean = float(np.mean(combined[i - pre_frames : i]))
            post_mean = float(np.mean(combined[i : i + post_frames]))
            jump_ratio = (post_mean - pre_mean) / (pre_mean + eps)
            contrast[i] = max(0.0, jump_ratio * combined[i])
        return contrast

    @staticmethod
    def _normalize_array(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - min_val) / (max_val - min_val)).astype(np.float32)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Analyse une piste audio et extrait les points de coupe Beat Drops."
    )
    parser.add_argument("audio_file", type=str, help="Chemin du fichier audio (.mp3, .wav)")
    parser.add_argument("--sensitivity", type=float, default=0.65)
    parser.add_argument("--duration", type=float, default=21.5)
    args = parser.parse_args()

    analyzer = AudioBeatDropAnalyzer()
    result = analyzer.analyze_track(
        args.audio_file, sensitivity=args.sensitivity, target_duration=args.duration
    )
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
