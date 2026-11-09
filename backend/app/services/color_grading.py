"""
Service d'Étalonnage "Aesthetic Posing & Bodybuilding" via FFmpeg & LUT 3D (.cube).
-----------------------------------------------------------------------------------
Propose 3 profils adaptés aux vidéos de posing filmées sous éclairage zénithal :
  1. `natural_aesthetic` (Défaut — Style `exemple.mp4`) :
     Conserve fidèlement l'exposition, les tons chairs et l'éclairage naturel du miroir/salle,
     avec uniquement une légère accentuation de netteté (`unsharp` doux) sur les stries et veines.
  2. `dark_aesthetic` :
     Applique une LUT 3D `.cube` équilibrée (ombres acier doux, contraste S-curve léger)
     sans boucher les noirs.
  3. `obsidian_monochrome` :
     Noir & Blanc esthétique haute définition inspiré de l'âge d'or du bodybuilding.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class DarkAestheticColorGrader:
    """
    Gère l'application du profil colorimétrique final via FFmpeg.
    """

    def __init__(self, lut_dir: Path = settings.LUT_DIR) -> None:
        self.lut_dir = lut_dir
        self.lut_dir.mkdir(parents=True, exist_ok=True)

    def ensure_default_dark_lut(self, preset_name: str = "dark_aesthetic") -> Path:
        """
        Génère une LUT 3D 17x17x17 équilibrée qui préserve la visibilité dans les basses
        lumières (essentiel lorsque le rush est déjà filmé dans une pièce sombre sous spot).
        """
        lut_path = self.lut_dir / f"{preset_name}.cube"
        size = 17
        lines = [
            "# AestheticEngine - Balanced Dark Aesthetic 3D LUT",
            f'TITLE "{preset_name.upper()}"',
            f"LUT_3D_SIZE {size}",
            "DOMAIN_MIN 0.0 0.0 0.0",
            "DOMAIN_MAX 1.0 1.0 1.0",
            "",
        ]

        for b_idx in range(size):
            for g_idx in range(size):
                for r_idx in range(size):
                    r = r_idx / float(size - 1)
                    g = g_idx / float(size - 1)
                    b = b_idx / float(size - 1)

                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    sat = 0.82
                    r_d = luma + (r - luma) * sat
                    g_d = luma + (g - luma) * sat
                    b_d = luma + (b - luma) * sat

                    # Courbe douce qui préserve les ombres tout en donnant du relief aux hautes lumières
                    def soft_curve(val: float) -> float:
                        return float(min(1.0, max(0.0, (val**1.04) * 1.02)))

                    r_out = soft_curve(r_d)
                    g_out = soft_curve(g_d)
                    b_out = min(1.0, soft_curve(b_d) + 0.012 * ((1.0 - luma) ** 2))

                    lines.append(f"{r_out:.6f} {g_out:.6f} {b_out:.6f}")

        lut_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lut_path

    def apply_dark_aesthetic_grade(
        self,
        input_video_path: Path,
        output_video_path: Path,
        lut_preset: str = "natural_aesthetic",
    ) -> Path:
        """
        Applique l'étalonnage choisi sans aucune déformation d'image (SAR 1:1 préservé).
        """
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        if lut_preset == "natural_aesthetic":
            # Rendu fidèle à `exemple.mp4` : couleurs et exposition d'origine + micro-netteté douce
            vf_chain = "unsharp=5:5:0.35:5:5:0.0,setsar=1"
        elif lut_preset == "obsidian_monochrome":
            vf_chain = "hue=s=0,eq=contrast=1.10:brightness=0.01:gamma=0.98,unsharp=5:5:0.55:5:5:0.0,setsar=1"
        else:
            lut_path = self.ensure_default_dark_lut("dark_aesthetic")
            escaped_lut_path = str(lut_path.resolve()).replace("\\", "/").replace(":", "\\:")
            vf_chain = (
                f"lut3d=file='{escaped_lut_path}':interp=tetrahedral,"
                "eq=contrast=1.04:brightness=0.01:saturation=0.90,"
                "unsharp=5:5:0.45:5:5:0.0,setsar=1"
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video_path),
            "-vf",
            vf_chain,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_video_path),
        ]

        logger.info("Exécution de l'étalonnage FFmpeg (%s) : %s", lut_preset, " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error("Erreur FFmpeg étalonnage, copie directe du master : %s", result.stderr)
            shutil.copyfile(input_video_path, output_video_path)

        return output_video_path
