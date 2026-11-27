#!/usr/bin/env python3
"""
Stateless dual-format (16:9 & native 9:16) deterministic video engine for Big Ben Harmonics.
Pure time function render_frame(t, aspect, events) -> RGB Image, seekable to any frame.
Inspired by one-letter + @0xwhrrari (studio) + @rincyXIII (dual cut) + @hnshah (clarity).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FPS = 30
BG = "#0b0c10"
PANEL = "#12141a"
GRID = "#262932"
BONE = "#e6e3dc"
MUTED = "#a9a8a2"
ASH = "#85868b"
AMBER = "#e2b45c"
CYAN = "#5ae6f0"
CLAY = "#ba6348"
SAGE = "#6ea388"

BB_MODES = [
    ("200 Hz\nG3", 200.26, 29.03, -6.2, AMBER),
    ("335 Hz\nE4 Dbl", 335.44, 14.17, -1.1, CLAY),
    ("500 Hz\nB4", 499.67, 17.33, 0.0, CYAN),
    ("678 Hz\nE5", 677.60, 9.28, -3.2, SAGE),
    ("883 Hz\nA5->A4", 883.34, 7.96, -0.4, "#c97ca6"),
    ("1088\nC#6", 1088.24, 7.11, -8.5, MUTED),
    ("1315\nE6", 1315.45, 8.53, -12.8, MUTED),
    ("1748\nA6", 1748.04, 5.44, -16.4, MUTED),
]


def load_font(size: int, kind: str = "serif") -> ImageFont.FreeTypeFont:
    candidates = {
        "serif": [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/NewYork.ttf",
        ],
        "serif_reg": [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        ],
        "mono": [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Courier.ttc",
        ],
    }[kind]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


FONTS = {
    (sz, k): load_font(sz, k)
    for sz in (16, 18, 20, 22, 24, 26, 28, 32, 36, 42, 48, 56, 68, 76)
    for k in ("serif", "serif_reg", "mono")
}


def get_act(t: float) -> tuple[int, float, float]:
    bounds = [(0.0, 6.5), (6.5, 13.5), (13.5, 20.5), (20.5, 27.5), (27.5, 33.6)]
    for idx, (t0, t1) in enumerate(bounds, start=1):
        if t < t1 or idx == 5:
            return idx, max(0.0, t - t0), max(0.0, min(1.0, (t - t0) / (t1 - t0)))
    return 5, 0.0, 1.0


def strike_pulse(t: float, events: list[dict]) -> tuple[float, dict | None]:
    """Return [0..1] flash/decay pulse driven strictly by the most recent audio event."""
    last_ev = None
    for ev in events:
        if ev["t_s"] <= t + 1e-6:
            last_ev = ev
    if last_ev is None:
        return 0.0, None
    dt = t - last_ev["t_s"]
    return math.exp(-dt / 0.45) if dt >= 0 else 0.0, last_ev


def draw_bar_spectrum(draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int,
                      t_rel: float, solo_idx: int | None = None, abbey_filter: bool = False) -> None:
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=PANEL, outline=GRID, width=2)
    base_y = y0 + h - 54
    draw.line([x0 + 28, base_y, x0 + w - 24, base_y], fill=ASH, width=2)
    n = len(BB_MODES)
    slot = (w - 64) / n
    bw = int(slot * 0.66)
    for i, (lbl, fz, t60, rel0, col) in enumerate(BB_MODES):
        tau = t60 / 6.907755
        lvl = rel0 - 8.686 * (t_rel / tau)
        if abs(fz - 335.44) < 2 and t_rel > 0:
            lvl += 2.5 * math.cos(2 * math.pi * 2.705 * t_rel)
        if abbey_filter:
            lvl -= (4.2 * (fz / 500.0) ** 1.35 + (12.0 if fz > 900.0 else 0.0))
        if solo_idx is not None and solo_idx != i:
            lvl -= 34.0
        frac = max(0.03, min(1.0, (lvl + 42.0) / 44.0))
        bh = int(frac * (h - 92))
        bx = int(x0 + 36 + i * slot + (slot - bw) * 0.5)
        draw.rounded_rectangle([bx, base_y - bh, bx + bw, base_y], radius=4, fill=col)
        draw.text((bx + bw // 2, base_y + 10), lbl.split("\n")[0], fill=MUTED, font=FONTS[(18, "mono")], anchor="mt")


def draw_cross_section(draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int,
                       is_yard: bool, pulse: float) -> None:
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=PANEL, outline=GRID, width=2)
    gy = y0 + h - 42
    draw.line([x0 + 26, gy, x0 + w - 26, gy], fill=MUTED, width=2)
    # Tower
    tx = x0 + int(w * 0.14)
    tw = 32
    ty_top = y0 + 52
    draw.rectangle([tx - tw // 2, ty_top, tx + tw // 2, gy], fill=CYAN)
    draw.polygon([(tx - tw // 2 - 4, ty_top), (tx, ty_top - 34), (tx + tw // 2 + 4, ty_top)], fill=CYAN)
    by = ty_top + 28
    # Pulse ring on impact
    if pulse > 0.04:
        rr = int(12 + (1.0 - pulse) * 70)
        draw.ellipse([tx - rr, by - rr, tx + rr, by + rr], outline=CLAY, width=3)
    draw.ellipse([tx - 8, by - 8, tx + 8, by + 8], fill=CLAY)
    draw.text((tx, ty_top - 48), "Elizabeth Tower (65m)", fill=BONE, font=FONTS[(18, "mono")], anchor="mm")

    rx_x = x0 + int(w * (0.86 if is_yard else 0.36))
    if is_yard:
        ax = x0 + int(w * 0.56)
        draw.rectangle([ax - 55, gy - 84, ax + 55, gy], fill="#2b2d36", outline=MUTED, width=2)
        draw.polygon([(ax - 62, gy - 84), (ax, gy - 110), (ax + 62, gy - 84)], fill="#353844", outline=MUTED)
        draw.text((ax, gy - 42), "Abbey", fill=BONE, font=FONTS[(22, "serif_reg")], anchor="mm")
        draw.line([tx, by, ax, gy - 110], fill=CLAY, width=4)
        draw.line([ax, gy - 110, rx_x, gy - 6], fill=AMBER, width=4)
        draw.text((x0 + w - 24, gy + 18), "Little Dean's Yard (354.5m)", fill=AMBER, font=FONTS[(18, "mono")], anchor="rm")
    else:
        draw.line([tx, by, rx_x, gy - 6], fill=CYAN, width=4)
        draw.text((rx_x, gy + 18), "Close Belfry (10m)", fill=CYAN, font=FONTS[(18, "mono")], anchor="mm")
    draw.ellipse([rx_x - 8, gy - 14, rx_x + 8, gy + 2], fill=AMBER if is_yard else CYAN)


def render_frame(t: float, events: list[dict], aspect: str) -> Image.Image:
    is_vert = (aspect == "9x16")
    W, H = (1080, 1920) if is_vert else (1920, 1080)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    act, t_in, prog = get_act(t)
    pulse, ev = strike_pulse(t, events)
    t_since_ev = max(0.0, t - ev["t_s"]) if ev else 0.0

    # Safe padding: vertical reels reserve >=130 px top/bottom for platform UI overlays
    pad_x = 72 if is_vert else 96
    top_y = 140 if is_vert else 68
    bot_y = H - (150 if is_vert else 64)

    # Progress track along very top safe edge
    draw.line([pad_x, top_y - 24, W - pad_x, top_y - 24], fill=GRID, width=4)
    total_u = min(1.0, max(0.0, t / 33.60))
    draw.line([pad_x, top_y - 24, int(pad_x + (W - 2 * pad_x) * total_u), top_y - 24], fill=AMBER, width=4)

    kicker = {
        1: "ACT I · NOT ONE NOTE",
        2: "ACT II · THE EVOLVING CHORD",
        3: "ACT III · ACROSS WESTMINSTER ABBEY",
        4: "ACT IV · 1890 TO 2024 & OSINT",
        5: "WESTMINSTER · zidingz.com",
    }[act]
    headline = {
        1: "Big Ben rings an E3/A4 clash,\nnot a pure bell tone.",
        2: "Fourteen seconds later, the\nchord transforms completely.",
        3: "Westminster Abbey strips 49 dB\nof bronze clang before arrival.",
        4: "Recovered 1890 cylinder speed\nlocks 134 years to 63.7 m OSINT.",
        5: "The Chords Inside Big Ben",
    }[act]

    draw.text((pad_x, top_y), kicker, fill=AMBER, font=FONTS[(24 if is_vert else 22, "mono")])
    h_font = FONTS[(56 if is_vert else 56, "serif")]
    draw.multiline_text((pad_x, top_y + 40), headline, fill=BONE, font=h_font, spacing=12)

    # Impact halo indicator in header corner
    hx, hy = W - pad_x - 36, top_y + 32
    if pulse > 0.05:
        rr = int(12 + (1.0 - pulse) * 32)
        draw.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], outline=CLAY, width=3)
    draw.ellipse([hx - 10, hy - 10, hx + 10, hy + 10], fill=CLAY if pulse > 0.25 else GRID)

    # Main stage geometry per aspect ratio
    stage_y = top_y + (220 if is_vert else 196)
    stage_h = bot_y - stage_y - 54
    stage_w = W - 2 * pad_x

    if act == 1:
        draw_bar_spectrum(draw, pad_x, stage_y, stage_w, stage_h, t_since_ev, abbey_filter=False)
        note_y = stage_y + 24
        sub_line = (
            "1969 BBC WAV: G3 200 · E4 335 · B4 500 · A5 883 Hz"
            if is_vert
            else "8 measured partials (1969 BBC WAV): G3 200.3 · E4 332.7/335.4 · B4 499.7 · A5 883.3 Hz"
        )
        draw.text((pad_x + 28, note_y), sub_line, fill=BONE, font=FONTS[(22 if not is_vert else 20, "mono")])
    elif act == 2:
        # Top or Left: isolated partial / doublet bars; Bottom or Right: Terhardt virtual chord flip
        solo = 4 if t < 8.4 else (1 if t < 10.6 else None)
        u_flip = min(1.0, max(0.0, (t - 8.4) / 3.8))
        if is_vert:
            h1 = int(stage_h * 0.50)
            draw_bar_spectrum(draw, pad_x, stage_y, stage_w, h1, t_since_ev, solo_idx=solo)
            cy = stage_y + h1 + 24
            ch = stage_h - h1 - 24
        else:
            w1 = int(stage_w * 0.56)
            draw_bar_spectrum(draw, pad_x, stage_y, w1, stage_h, t_since_ev, solo_idx=solo)
            cy = stage_y
            ch = stage_h
            cx = pad_x + w1 + 24
            cw = stage_w - w1 - 24
        if is_vert:
            cx, cw = pad_x, stage_w
        draw.rectangle([cx, cy, cx + cw, cy + ch], fill=PANEL, outline=GRID, width=2)
        draw.text((cx + 24, cy + 22), "Terhardt Virtual-Pitch Salience", fill=BONE, font=FONTS[(28, "serif")])
        bars = [
            ("E3 / E4 (168 Hz)", 1.0 * (1 - u_flip) + 0.70 * u_flip, CLAY),
            ("A4 Clash (441 Hz)", 0.68 * (1 - u_flip) + 0.02 * u_flip, "#c97ca6"),
            ("B4 Fifth (500 Hz)", 0.45 * (1 - u_flip) + 1.00 * u_flip, CYAN),
            ("G3 Hum (200 Hz)", 0.11 * (1 - u_flip) + 0.56 * u_flip, AMBER),
        ]
        for bi, (blbl, bval, bcol) in enumerate(bars):
            by_row = cy + 88 + bi * (int((ch - 120) / 4))
            draw.text((cx + 24, by_row), f"{blbl}  {bval:.2f}", fill=MUTED, font=FONTS[(22, "mono")])
            draw.rectangle([cx + 24, by_row + 32, cx + cw - 24, by_row + 54], fill=GRID)
            draw.rectangle([cx + 24, by_row + 32, int(cx + 24 + (cw - 48) * bval), by_row + 54], fill=bcol)
    elif act == 3:
        is_yard = (t >= 16.50)
        if is_vert:
            h1 = int(stage_h * 0.48)
            draw_cross_section(draw, pad_x, stage_y, stage_w, h1, is_yard, pulse)
            draw_bar_spectrum(draw, pad_x, stage_y + h1 + 22, stage_w, stage_h - h1 - 22, t_since_ev, abbey_filter=is_yard)
        else:
            w1 = int(stage_w * 0.50)
            draw_cross_section(draw, pad_x, stage_y, w1, stage_h, is_yard, pulse)
            draw_bar_spectrum(draw, pad_x + w1 + 24, stage_y, stage_w - w1 - 24, stage_h, t_since_ev, abbey_filter=is_yard)
    elif act == 4:
        draw.rectangle([pad_x, stage_y, pad_x + stage_w, stage_y + stage_h], fill=PANEL, outline=GRID, width=2)
        rows = [
            ("1890 Edison Wax Cylinder", "alpha = 0.9464x (-0.95 st)", "200.0 / 331.9 / 500.7 / 677.3 Hz", AMBER),
            ("1953 / 1969 BBC WAVs", "50 Hz UK Mains ENF Lock", "T60 = 29.0 s Hum · 2.71 Hz Doublet", CYAN),
            ("2017 Codec Stress Test", "185 kbps vs 67 kbps MP3", "0.01 Hz drift · 0.01 dB ratio", SAGE),
            ("2024 NYE Broadcast", "Post-Restoration Match", "335.97 / 500.42 / 884.61 Hz", BONE),
            ("Westminster Acoustic OSINT", "LOO Slant Benchmark", "63.7 m MAE vs 82.6 m (Bridge St -0.40 ms)", CLAY),
        ]
        step_y = stage_h // len(rows)
        for ri, (title_r, tag_r, val_r, col_r) in enumerate(rows):
            ry = stage_y + 28 + ri * step_y
            active = (ri <= int(prog * 6.5))
            draw.line([pad_x + 24, ry + step_y - 14, pad_x + stage_w - 24, ry + step_y - 14], fill=GRID, width=1)
            draw.text((pad_x + 28, ry + 6), title_r, fill=col_r if active else ASH, font=FONTS[(28 if not is_vert else 24, "serif")])
            draw.text((pad_x + 28, ry + 44), f"{tag_r}  |  {val_r}", fill=BONE if active else ASH, font=FONTS[(20 if not is_vert else 16, "mono")])
    else:
        # Coda pause card: clean, static, high-contrast (>2.5s readable)
        draw.rectangle([pad_x, stage_y, pad_x + stage_w, stage_y + stage_h], fill=PANEL, outline=AMBER, width=2)
        cx, cy = W // 2, stage_y + stage_h // 2
        draw.text((cx, cy - 86), "Measured across 20 recordings (1890-2024)", fill=MUTED, font=FONTS[(28 if not is_vert else 24, "serif_reg")], anchor="mm")
        draw.text((cx, cy - 20), "Full interactive + code + LaTeX preprint:", fill=BONE, font=FONTS[(32 if not is_vert else 28, "serif")], anchor="mm")
        draw.text((cx, cy + 44), "zidingz.com/the-chords-inside-big-ben", fill=CYAN, font=FONTS[(32 if not is_vert else 26, "mono")], anchor="mm")
        draw.text((cx, cy + 106), "Ziding Zhang · Westminster School / Little Dean's Yard", fill=AMBER, font=FONTS[(22 if not is_vert else 20, "mono")], anchor="mm")

    # Subtitle strip inside bottom safe zone
    sub_lbl = ev["label"] if ev else "Listening across Westminster..."
    draw.text((pad_x, bot_y - 24), sub_lbl, fill=MUTED, font=FONTS[(22 if not is_vert else 20, "mono")])
    draw.text((W - pad_x, bot_y - 24), f"{t:05.2f}s", fill=ASH, font=FONTS[(20, "mono")], anchor="ra")
    return img


def render_cut(aspect: str, wav_path: Path, events: list[dict], out_mp4: Path, out_poster: Path) -> None:
    w, h = (1080, 1920) if aspect == "9x16" else (1920, 1080)
    total_frames = int(round(33.60 * FPS))
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
        "-i", str(wav_path),
        "-c:v", "libx264", "-crf", "21", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    poster_t = 16.95
    poster_saved = False
    for f_idx in range(total_frames):
        t = f_idx / FPS
        im = render_frame(t, events, aspect)
        proc.stdin.write(im.tobytes())
        if not poster_saved and t >= poster_t:
            im.save(out_poster, format="WEBP", quality=88)
            poster_saved = True
    proc.stdin.close()
    err = proc.stderr.read() if proc.stderr else b""
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {aspect}: {err.decode('utf-8', 'ignore')[-800:]}")
    sz_mb = out_mp4.stat().st_size / (1024 * 1024)
    print(f"Rendered {out_mp4} ({w}x{h}, {total_frames} frames, {sz_mb:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    build_dir = Path(__file__).resolve().parent / "build"
    wav_path = build_dir / "master_audio.wav"
    events_data = json.loads((build_dir / "events.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for aspect in ("16x9", "9x16"):
        render_cut(
            aspect,
            wav_path,
            events_data["events"],
            args.out_dir / f"big-ben-{aspect}.mp4",
            args.out_dir / f"big-ben-{aspect}-poster.webp",
        )


if __name__ == "__main__":
    main()
