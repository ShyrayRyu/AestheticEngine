"""
Générateur d'export XML (XMEML v5) pour Adobe Premiere Pro & DaVinci Resolve.
-----------------------------------------------------------------------------
Permet d'ouvrir la timeline exacte (9:16 ou 16:9, coupes franches sur les beats,
marqueurs de Beat Drops et piste audio) dans Premiere Pro ou DaVinci Resolve.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

from app.models.schemas import BeatDropEvent, SelectedVideoSegment


class PremiereResolveXMLExporter:
    """Construit un fichier XML d'échange de montage (XMEML v5)."""

    def __init__(self, fps: int = 30) -> None:
        self.fps = fps

    def export_timeline_xml(
        self,
        project_id: str,
        segments: List[SelectedVideoSegment],
        beat_drops: List[BeatDropEvent],
        audio_path: Optional[Path],
        output_xml_path: Path,
        aspect_ratio: Literal["9:16", "16:9"] = "9:16",
    ) -> Path:
        width, height = (1920, 1080) if aspect_ratio == "16:9" else (1080, 1920)

        root = ET.Element("xmeml", version="5")
        sequence = ET.SubElement(root, "sequence", id=f"seq-{project_id}")
        ET.SubElement(sequence, "name").text = f"AestheticEngine_{project_id}_{aspect_ratio.replace(':', 'x')}"

        rate = ET.SubElement(sequence, "rate")
        ET.SubElement(rate, "timebase").text = str(self.fps)
        ET.SubElement(rate, "ntsc").text = "FALSE"

        media = ET.SubElement(sequence, "media")
        video = ET.SubElement(media, "video")
        v_format = ET.SubElement(video, "format")
        sample_char = ET.SubElement(v_format, "samplecharacteristics")
        ET.SubElement(sample_char, "width").text = str(width)
        ET.SubElement(sample_char, "height").text = str(height)
        ET.SubElement(sample_char, "pixelaspectratio").text = "square"

        v_track = ET.SubElement(video, "track")
        timeline_frame_cursor = 0

        for idx, seg in enumerate(segments):
            src_path = Path(seg.source_video).resolve()
            slot_dur = max(0.5, seg.timeline_end - seg.timeline_start) if seg.timeline_end > seg.timeline_start else max(0.5, seg.end_time - seg.start_time)
            dur_frames = max(1, int(round(slot_dur * self.fps)))

            self._append_clipitem(
                track=v_track,
                clip_id=f"clip-{idx + 1}",
                clip_name=f"{seg.pose_label} ({src_path.name})",
                file_path=src_path,
                start_frame=timeline_frame_cursor,
                end_frame=timeline_frame_cursor + dur_frames,
                in_frame=int(round(seg.start_time * self.fps)),
                out_frame=int(round(seg.end_time * self.fps)),
            )
            timeline_frame_cursor += dur_frames

        ET.SubElement(sequence, "duration").text = str(max(1, timeline_frame_cursor))

        if audio_path is not None and audio_path.exists():
            audio = ET.SubElement(media, "audio")
            a_track = ET.SubElement(audio, "track")
            abs_audio = audio_path.resolve()
            self._append_clipitem(
                track=a_track,
                clip_id="audio-master-1",
                clip_name=abs_audio.name,
                file_path=abs_audio,
                start_frame=0,
                end_frame=timeline_frame_cursor,
                in_frame=0,
                out_frame=timeline_frame_cursor,
            )

        for d_idx, drop in enumerate(beat_drops):
            marker = ET.SubElement(sequence, "marker")
            ET.SubElement(marker, "name").text = f"BEAT CUT #{d_idx + 1}"
            ET.SubElement(marker, "comment").text = f"Impact: {int(drop.intensity * 100)}%"
            drop_frame = int(round(drop.timestamp * self.fps))
            ET.SubElement(marker, "in").text = str(drop_frame)
            ET.SubElement(marker, "out").text = str(drop_frame + 1)

        raw_xml = ET.tostring(root, encoding="utf-8")
        pretty_xml = minidom.parseString(raw_xml).toprettyxml(indent="  ", encoding="utf-8")

        output_xml_path.parent.mkdir(parents=True, exist_ok=True)
        output_xml_path.write_bytes(pretty_xml)
        return output_xml_path

    def _append_clipitem(
        self,
        track: ET.Element,
        clip_id: str,
        clip_name: str,
        file_path: Path,
        start_frame: int,
        end_frame: int,
        in_frame: int,
        out_frame: int,
    ) -> None:
        clipitem = ET.SubElement(track, "clipitem", id=clip_id)
        ET.SubElement(clipitem, "name").text = clip_name
        ET.SubElement(clipitem, "duration").text = str(max(1, end_frame - start_frame))

        rate = ET.SubElement(clipitem, "rate")
        ET.SubElement(rate, "timebase").text = str(self.fps)
        ET.SubElement(rate, "ntsc").text = "FALSE"

        ET.SubElement(clipitem, "start").text = str(start_frame)
        ET.SubElement(clipitem, "end").text = str(end_frame)
        ET.SubElement(clipitem, "in").text = str(in_frame)
        ET.SubElement(clipitem, "out").text = str(out_frame)

        file_el = ET.SubElement(clipitem, "file", id=f"file-{clip_id}")
        ET.SubElement(file_el, "name").text = file_path.name
        ET.SubElement(file_el, "pathurl").text = file_path.as_uri()
