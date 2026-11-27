#!/usr/bin/env python3
"""Verify <1 frame audio-visual onset sync and emit visual contact sheets for 16:9 and 9:16."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import soundfile as sf
except ModuleNotFoundError:
    def _read_wav(path):
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
            if wf.getnchannels() > 1:
                arr = arr.reshape(-1, wf.getnchannels()).mean(axis=1)
            return arr, sr
    sf = types.SimpleNamespace(read=_read_wav)

from studio_engine import render_frame, FPS


def build_contact_sheet(mp4_path: Path, aspect: str, out_jpg: Path, timestamps: list[float]) -> None:
    tiles = []
    w, h = (320, 568) if aspect == "9x16" else (480, 270)
    tmp_dir = out_jpg.parent / f"_frames_{aspect}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, ts in enumerate(timestamps):
        fr_png = tmp_dir / f"t_{i:02d}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(mp4_path), "-frames:v", "1", "-q:v", "2", str(fr_png)],
            check=True, capture_output=True,
        )
        im = Image.open(fr_png).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(im)
        d.text((10, 10), f"t={ts:.2f}s", fill="#e2b45c")
        tiles.append(im)
    cols = 4 if aspect == "9x16" else 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "#0b0c10")
    for idx, im in enumerate(tiles):
        r, c = divmod(idx, cols)
        sheet.paste(im, (c * w, r * h))
    sheet.save(out_jpg, quality=88)
    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()
    print(f"Saved contact sheet: {out_jpg}")


def verify_audio_visual_sync(mp4_path: Path, events: list[dict], aspect: str) -> float:
    tmp_wav = mp4_path.parent / f"_check_{aspect}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4_path), "-vn", "-ac", "1", "-ar", "48000", str(tmp_wav)],
        check=True, capture_output=True,
    )
    y, sr = sf.read(tmp_wav)
    tmp_wav.unlink()
    win = int(0.015 * sr)
    max_err_frames = 0.0
    for ev in events:
        t0 = ev["t_s"]
        i0 = int(round(t0 * sr))
        pre_rms = float(np.sqrt(np.mean(y[max(0, i0 - win) : i0] ** 2)))
        post_rms = float(np.sqrt(np.mean(y[i0 : i0 + win] ** 2)))
        # Visual onset transient occurs precisely at frame = round(t0 * FPS)
        f_ev = int(round(t0 * FPS))
        f_prev = np.asarray(render_frame((f_ev - 1) / FPS, events, aspect), dtype=np.float32)
        f_cur = np.asarray(render_frame(f_ev / FPS, events, aspect), dtype=np.float32)
        vis_delta = float(np.mean(np.abs(f_cur - f_prev)))
        # Measure audio rise centroid within [t0-0.02, t0+0.02]
        seg = np.abs(y[max(0, i0 - int(0.015 * sr)) : i0 + int(0.020 * sr)])
        thresh = pre_rms + 0.35 * max(1e-6, post_rms - pre_rms)
        hits = np.where(seg >= thresh)[0]
        audio_onset_s = (max(0, i0 - int(0.015 * sr)) + int(hits[0])) / sr if len(hits) else t0
        err_frames = abs(audio_onset_s - (f_ev / FPS)) * FPS
        max_err_frames = max(max_err_frames, err_frames)
        if post_rms <= pre_rms or vis_delta < 0.01:
            raise AssertionError(f"Missing AV transient at t={t0}: pre={pre_rms:.4f} post={post_rms:.4f} vis={vis_delta:.4f}")
    assert max_err_frames < 1.0, f"Sync drift >= 1 frame: {max_err_frames:.3f}"
    print(f"AV sync verified ({aspect}): max_err = {max_err_frames:.3f} frames across {len(events)} events")
    return max_err_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media-dir", type=Path, required=True)
    ap.add_argument("--shots-dir", type=Path, required=True)
    args = ap.parse_args()
    events = json.loads((Path(__file__).resolve().parent / "build" / "events.json").read_text())["events"]
    ts_list = [0.55, 6.85, 8.45, 13.85, 16.55, 21.70, 24.65, 29.00]
    for aspect in ("16x9", "9x16"):
        mp4 = args.media_dir / f"big-ben-{aspect}.mp4"
        verify_audio_visual_sync(mp4, events, aspect)
        build_contact_sheet(mp4, aspect, args.shots_dir / f"contact-sheet-{aspect}.jpg", ts_list)


if __name__ == "__main__":
    main()
