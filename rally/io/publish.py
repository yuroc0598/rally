"""Atomic publication of processed video and its describing sidecar."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from .ffmpeg import add_real_context, cut_segments, find_font, render_labeled


def write_output(input_path, output_path, json_path, result, info, cfg, progress) -> None:
    """Publish video first and metadata last so sidecars never describe partial media."""
    def write_sidecar() -> None:
        if not json_path:
            return
        sidecar_path = Path(json_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = sidecar_path.with_name(
            f".{sidecar_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w") as handle:
                json.dump(result.sidecar(), handle, indent=2)
            os.replace(temporary, sidecar_path)
        finally:
            temporary.unlink(missing_ok=True)
        progress(f"wrote {json_path}")

    if not output_path:
        write_sidecar()
        return
    segments = result.segments
    if not segments:
        Path(output_path).unlink(missing_ok=True)
        progress("no rally segments found -> not writing output video")
    else:
        render_segments = add_real_context(
            segments, info.duration_s,
            cfg.point_start_buffer_s, cfg.point_end_buffer_s)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix or '.mp4'}")
        try:
            if cfg.reencode and (cfg.label_points or cfg.inter_point_gap_s > 0):
                font = find_font() if cfg.label_points else None
                if cfg.label_points and font is None:
                    progress("  no font found -> labels drawn with ffmpeg's default font")
                what = "labelled points" if cfg.label_points else "points"
                progress(f"rendering {len(segments)} {what} -> {output_path}")
                render_labeled(
                    input_path, render_segments, str(temporary),
                    gap_s=cfg.inter_point_gap_s,
                    label_prefix=cfg.label_prefix,
                    font=font,
                    video_height=info.height,
                    has_audio=info.has_audio,
                    draw_labels=cfg.label_points,
                )
            else:
                progress(f"cutting {len(segments)} segments -> {output_path}")
                cut_segments(
                    input_path, render_segments, str(temporary), reencode=cfg.reencode)
            if not temporary.exists() or temporary.stat().st_size <= 0:
                raise RuntimeError("video renderer produced no output")
            os.replace(temporary, destination)
            progress(f"wrote {output_path}")
        finally:
            temporary.unlink(missing_ok=True)
    write_sidecar()
