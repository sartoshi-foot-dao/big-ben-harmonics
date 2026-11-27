#!/usr/bin/env python3
"""
Stateless dual-format (16:9 & native 9:16) broadsheet-cinema video engine for Big Ben Harmonics.
Pure time function render_frame(t, events, aspect) -> RGB Image, seekable to any frame.
Harmonizes warm rag-paper (#FAF9F5), charcoal ink (#141413), terracotta (#BA6348), slate (#3D5A6C),
and aged bronze (#8C6239) with the Little Dean's Yard dusk cel-and-gouache painting.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FPS = 30
DURATION_S = 33.60

PAPER = "#FAF9F5"
CREAM = "#F1EEE4"
INK = "#141413"
STONE = "#6B6A64"
RULE = "#DAD6CA"
CLAY = "#BA6348"
SLATE = "#3D5A6C"
SAGE = "#587560"
BRONZE = "#8C6239"
AMBER = "#F4BE6E"
DUSK_INK = "#161C24"

BB_MODES = [
    ("200 Hz · G3 hum", 200.26, 29.03, -6.2, BRONZE),
    ("335 Hz · E4 doublet", 335.44, 14.17, -1.1, CLAY),
    ("500 Hz · B4 fifth", 499.67, 17.33, 0.0, SLATE),
    ("678 Hz · E5 octave", 677.60, 9.28, -3.2, SAGE),
    ("883 Hz · A5 clash", 883.34, 7.96, -0.4, "#9A4A64"),
]

BLOG_ROOT = Path(__file__).resolve().parents[4]
STILL_PATH = BLOG_ROOT / "public/images/the-chords-inside-big-ben/still.webp"
_BG_CACHE: dict[tuple[int, int, str], Image.Image] = {}


def load_font(size: int, kind: str = "serif") -> ImageFont.FreeTypeFont:
    candidates = {
        "serif": [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/System/Library/Fonts/NewYork.ttf",
        ],
        "serif_reg": [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        ],
        "sans": [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ],
        "sans_reg": [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ],
        "mono": [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ],
    }[kind]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


FONTS: dict[tuple[int, str], ImageFont.FreeTypeFont] = {}


def font(sz: int, kind: str = "serif") -> ImageFont.FreeTypeFont:
    k = (sz, kind)
    if k not in FONTS:
        FONTS[k] = load_font(sz, kind)
    return FONTS[k]


def get_dusk_plate(W: int, H: int, mode: str = "dark") -> Image.Image:
    key = (W, H, mode)
    if key in _BG_CACHE:
        return _BG_CACHE[key].copy()
    if STILL_PATH.exists():
        im = Image.open(STILL_PATH).convert("RGB")
        # Cover crop preserving Elizabeth Tower on ~60% right-center
        sw, sh = im.size
        target_ar = W / H
        src_ar = sw / sh
        if target_ar < src_ar:
            new_w = int(sh * target_ar)
            x0 = int((sw - new_w) * 0.58)
            im = im.crop((x0, 0, x0 + new_w, sh))
        else:
            new_h = int(sw / target_ar)
            y0 = int((sh - new_h) * 0.25)
            im = im.crop((0, y0, sw, y0 + new_h))
        im = im.resize((W, H), Image.Resampling.LANCZOS)
        if mode == "dark":
            im = ImageEnhance.Brightness(im).enhance(0.58)
            im = ImageEnhance.Contrast(im).enhance(1.06)
        else:
            im = ImageEnhance.Brightness(im).enhance(0.70)
    else:
        im = Image.new("RGB", (W, H), DUSK_INK)
    _BG_CACHE[key] = im
    return im.copy()


def get_act(t: float) -> tuple[int, float, float]:
    bounds = [(0.0, 6.5), (6.5, 13.5), (13.5, 20.5), (20.5, 27.5), (27.5, 33.6)]
    for idx, (t0, t1) in enumerate(bounds, start=1):
        if t < t1 or idx == 5:
            return idx, max(0.0, t - t0), max(0.0, min(1.0, (t - t0) / (t1 - t0)))
    return 5, 0.0, 1.0


def strike_pulse(t: float, events: list[dict]) -> tuple[float, Optional[dict]]:
    last_ev = None
    for ev in events:
        if ev["t_s"] <= t + 1e-6:
            last_ev = ev
    if last_ev is None:
        return 0.0, None
    dt = t - last_ev["t_s"]
    return math.exp(-dt / 0.48) if dt >= 0 else 0.0, last_ev


def draw_organ_ribbons(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    w: int,
    h: int,
    t_rel: float,
    solo_idx: Optional[int] = None,
    dark_bg: bool = False,
    abbey_filter: bool = False,
) -> None:
    fg = "#F4F1E8" if dark_bg else INK
    mut = "#C6C2B8" if dark_bg else STONE
    track_col = "#2A323D" if dark_bg else CREAM
    lbl_w = min(290, int(w * 0.28))
    bx0 = x0 + lbl_w
    bw_max = w - lbl_w - 24
    row_h = h // len(BB_MODES)

    for i, (lbl, fz, t60, rel0, col) in enumerate(BB_MODES):
        ry = y0 + i * row_h
        cy = ry + row_h // 2
        tau = t60 / 6.907755
        lvl = rel0 - 8.68589 * (t_rel / tau)
        if abbey_filter:
            lvl -= 4.2 * (fz / 500.0) ** 1.35 + (10.0 if fz > 800 else 0.0)
        if solo_idx is not None and solo_idx != i:
            lvl -= 36.0
        beat = 1.0
        if abs(fz - 335.44) < 2.0:
            beat = 0.45 + 0.55 * abs(math.cos(math.pi * 2.705 * t_rel))
        frac = max(0.02, min(1.0, (lvl + 38.0) / 40.0)) * (beat if solo_idx == 1 else (0.72 + 0.28 * beat))
        dimmed = solo_idx is not None and solo_idx != i

        draw.text((x0 + 8, cy - 2), lbl, fill=mut if dimmed else fg, font=font(24 if w > 900 else 20, "serif_reg"), anchor="lm")
        # Warm pill track & live ribbon
        rh = max(14, int(row_h * 0.44))
        draw.rounded_rectangle([bx0, cy - rh // 2, bx0 + bw_max, cy + rh // 2], radius=rh // 2, fill=track_col)
        fill_w = max(rh, int(bw_max * frac))
        draw.rounded_rectangle([bx0, cy - rh // 2, bx0 + fill_w, cy + rh // 2], radius=rh // 2, fill=col if not dimmed else "#BDB9AF")
        db_txt = f"{lvl:+.0f} dB" if not dimmed else "muted"
        draw.text((bx0 + bw_max - 12, cy), db_txt, fill=mut, font=font(18, "sans_reg"), anchor="rm")


def draw_beating_waves(
    draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int, t_rel: float, is_vert: bool
) -> None:
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=14, fill=CREAM, outline=RULE, width=2)
    draw.text((x0 + 28, y0 + 24), "332.74 Hz + 335.44 Hz: one swell every 0.37 seconds", fill=INK, font=font(26 if not is_vert else 22, "serif"))
    cy = y0 + int(h * 0.58)
    draw.line([x0 + 28, cy, x0 + w - 28, cy], fill=RULE, width=2)

    pts_pos, pts_neg = [], []
    n_pts = 240
    span_s = 2.6
    amp = h * 0.28
    for i in range(n_pts):
        u = i / (n_pts - 1)
        tau = u * span_s
        x = int(x0 + 28 + u * (w - 56))
        env = abs(math.cos(math.pi * 2.705 * tau)) * math.exp(-tau / 4.2)
        pts_pos.append((x, int(cy - env * amp)))
        pts_neg.append((x, int(cy + env * amp)))

    poly = pts_pos + list(reversed(pts_neg))
    draw.polygon(poly, fill="#E9D6CD")
    draw.line(pts_pos, fill=CLAY, width=4)
    draw.line(pts_neg, fill=CLAY, width=4)

    cur_u = (t_rel % span_s) / span_s
    cur_x = int(x0 + 28 + cur_u * (w - 56))
    draw.line([cur_x, y0 + 74, cur_x, y0 + h - 32], fill=SLATE, width=4)


def draw_abbey_cutaway(
    draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int, is_yard: bool, pulse: float, t_in: float
) -> None:
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=14, fill=CREAM, outline=RULE, width=2)
    gy = y0 + h - 54
    draw.line([x0 + 34, gy, x0 + w - 34, gy], fill=INK, width=3)

    tx = x0 + int(w * 0.12)
    ty_top = y0 + int(h * 0.24)
    draw.rectangle([tx - 18, ty_top, tx + 18, gy], fill=SLATE)
    draw.polygon([(tx - 22, ty_top), (tx, ty_top - 42), (tx + 22, ty_top)], fill="#273D4A")
    by = ty_top + 24
    if pulse > 0.04:
        rr = int(14 + (1.0 - pulse) * 76)
        draw.ellipse([tx - rr, by - rr, tx + rr, by + rr], outline=CLAY, width=4)
    draw.ellipse([tx - 9, by - 9, tx + 9, by + 9], fill=CLAY)
    draw.text((tx, ty_top - 64), "Elizabeth Tower (65 m)", fill=INK, font=font(20, "sans"), anchor="mm")

    ax = x0 + int(w * 0.50)
    ry = gy - int(h * 0.42)
    if is_yard:
        draw.polygon([(ax - 68, gy), (ax - 68, ry + 36), (ax, ry), (ax + 68, ry + 36), (ax + 68, gy)], fill="#CFCAC0", outline=STONE, width=2)
        draw.text((ax, gy - 44), "Westminster Abbey", fill=INK, font=font(22, "serif"), anchor="mm")
        # High clang blocked
        draw.line([tx + 12, by, ax, ry + 6], fill=CLAY, width=4)
        draw.text((int(0.5 * (tx + ax)), by - 20), "5 kHz clang: -49.5 dB", fill=CLAY, font=font(19, "sans"), anchor="mm")
        # Deep 200 Hz wave diffracts smoothly into courtyard
        arc_pts = []
        rx_x = x0 + int(w * 0.87)
        for k in range(32):
            u = k / 31.0
            px = tx + 12 + u * (rx_x - tx - 12)
            py = (1 - u) ** 2 * by + 2 * (1 - u) * u * (ry - 24) + (u ** 2) * (gy - 10)
            arc_pts.append((px, py))
        draw.line(arc_pts, fill=SLATE, width=5)
        draw.ellipse([rx_x - 10, gy - 18, rx_x + 10, gy + 2], fill=BRONZE)
        draw.text((rx_x, gy + 26), "Little Dean's Yard (354.5 m)", fill=BRONZE, font=font(20, "sans"), anchor="rm")
    else:
        rx_x = x0 + int(w * 0.34)
        draw.line([tx + 12, by, rx_x, gy - 10], fill=SLATE, width=5)
        draw.ellipse([rx_x - 10, gy - 18, rx_x + 10, gy + 2], fill=SLATE)
        draw.text((rx_x, gy + 26), "10 m Belfry Mic (raw bright impact)", fill=SLATE, font=font(20, "sans"), anchor="lm")


def draw_quarters_overlap(
    draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int, t: float, is_vert: bool
) -> None:
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=14, fill=CREAM, outline=RULE, width=2)
    bells = [
        ("G#4 · 870 Hz", 20.80, CLAY),
        ("F#4 · 771 Hz", 21.60, SLATE),
        ("E4  · 679 Hz", 22.40, SAGE),
        ("B3  · 505 Hz", 23.20, BRONZE),
    ]
    bx0 = x0 + (220 if not is_vert else 170)
    bw = w - (250 if not is_vert else 200)
    row_h = h // 4
    for i, (lbl, t_hit, col) in enumerate(bells):
        cy = y0 + int((i + 0.5) * row_h)
        draw.text((x0 + 26, cy), lbl, fill=INK, font=font(24 if not is_vert else 20, "serif"), anchor="lm")
        start_u = (t_hit - 20.5) / 4.2
        cur_u = max(0.0, min(1.0, (t - 20.5) / 4.2))
        sx = int(bx0 + start_u * bw)
        if t >= t_hit:
            ex = int(bx0 + cur_u * bw)
            if ex > sx:
                draw.rounded_rectangle([sx, cy - 15, max(sx + 20, ex), cy + 15], radius=15, fill=col)
            draw.ellipse([sx - 10, cy - 10, sx + 10, cy + 10], fill="#FAF9F5", outline=col, width=4)
        else:
            draw.ellipse([sx - 6, cy - 6, sx + 6, cy + 6], outline=RULE, width=2)


def render_frame(t: float, events: list[dict], aspect: str) -> Image.Image:
    is_vert = (aspect == "9x16")
    W, H = (1080, 1920) if is_vert else (1920, 1080)
    act, t_in, prog = get_act(t)
    pulse, ev = strike_pulse(t, events)
    t_since = max(0.0, t - ev["t_s"]) if ev else 0.0

    # Acts 1 & 5 open and close on the atmospheric Little Dean's Yard dusk painting; Acts 2-4 rest on clean rag paper
    on_dusk = act in (1, 5)
    if on_dusk:
        img = get_dusk_plate(W, H, mode="dark")
    else:
        img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)

    fg = "#F6F3EB" if on_dusk else INK
    mut = "#D4CEC2" if on_dusk else STONE
    accent = AMBER if on_dusk else CLAY

    pad_x = 72 if is_vert else 96
    top_y = 144 if is_vert else 72
    bot_y = H - (144 if is_vert else 68)

    # Progress hairline
    draw.line([pad_x, top_y - 26, W - pad_x, top_y - 26], fill="#384352" if on_dusk else RULE, width=4)
    draw.line([pad_x, top_y - 26, int(pad_x + (W - 2 * pad_x) * (t / DURATION_S)), top_y - 26], fill=accent, width=4)

    kickers = {
        1: "BIG BEN · ONE BLOW, MANY VOICES",
        2: "THE 14-SECOND CHORD SHIFT",
        3: "ACROSS WESTMINSTER ABBEY",
        4: "WESTMINSTER QUARTERS & 1890 LOCK",
        5: "LISTEN · zidingz.com",
    }
    heads = {
        1: "Big Ben does not ring one note.\nOne hammer blow wakes a chord.",
        2: "At impact, A5 clashes with E.\nThen B4 and G3 glow warm.",
        3: "Westminster Abbey's roof filters\n49 dB of bright bronze clang.",
        4: "Four chimes overlap in air,\nlocked since the 1890 cylinder.",
        5: "The Chords Inside Big Ben",
    }

    draw.text((pad_x, top_y), kickers[act], fill=accent, font=font(22 if is_vert else 20, "sans"))
    draw.multiline_text((pad_x, top_y + 38), heads[act], fill=fg, font=font(50 if is_vert else 58, "serif"), spacing=12)

    # Tactile pulse indicator on exact strike frames
    hx, hy = W - pad_x - 32, top_y + 30
    if pulse > 0.04:
        rr = int(12 + (1.0 - pulse) * 34)
        draw.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], outline=accent, width=4)
    draw.ellipse([hx - 10, hy - 10, hx + 10, hy + 10], fill=accent if pulse > 0.2 else ("#3A4552" if on_dusk else RULE))

    stage_y = top_y + (226 if is_vert else 200)
    stage_h = bot_y - stage_y - 52
    stage_w = W - 2 * pad_x

    if act == 1:
        # Subtle warm dark panel over bottom half of dusk cel so organ pipes are effortless to read
        panel_y = stage_y + int(stage_h * (0.28 if is_vert else 0.34))
        panel_h = stage_y + stage_h - panel_y
        draw.rounded_rectangle([pad_x, panel_y, pad_x + stage_w, panel_y + panel_h], radius=14, fill="#161D26", outline="#323F4F", width=2)
        draw_organ_ribbons(draw, pad_x + 24, panel_y + 20, stage_w - 48, panel_h - 40, t_since, dark_bg=True)
    elif act == 2:
        if t < 8.4:
            solo = 4  # 883 Hz A5 clash
            draw_organ_ribbons(draw, pad_x, stage_y, stage_w, stage_h, t_since, solo_idx=solo)
        elif t < 10.6:
            draw_beating_waves(draw, pad_x, stage_y, stage_w, stage_h, t_since, is_vert)
        else:
            solo = None
            draw_organ_ribbons(draw, pad_x, stage_y, stage_w, stage_h, t_since, solo_idx=None)
    elif act == 3:
        is_yard = t >= 16.50
        draw_abbey_cutaway(draw, pad_x, stage_y, stage_w, stage_h, is_yard, pulse, t_in)
    elif act == 4:
        if t < 24.60:
            draw_quarters_overlap(draw, pad_x, stage_y, stage_w, stage_h, t, is_vert)
        else:
            draw.rounded_rectangle([pad_x, stage_y, pad_x + stage_w, stage_y + stage_h], radius=14, fill=CREAM, outline=RULE, width=2)
            eras = [
                ("1890 Edison wax cylinder", "Speed recovered x0.9464: 200.0 · 331.9 · 500.7 Hz", CLAY),
                ("1969 BBC belfry WAVs", "UK grid locked: 200.3 · 335.4 · 499.7 · 883.3 Hz", SLATE),
                ("2024 NYE broadcast", "Post-restoration: 200.6 · 336.0 · 500.4 · 884.6 Hz", BRONZE),
            ]
            step = stage_h // 3
            for i, (et, ev_txt, ecol) in enumerate(eras):
                ey = stage_y + 24 + i * step
                draw.ellipse([pad_x + 36, ey + 28, pad_x + 56, ey + 48], fill=ecol)
                draw.text((pad_x + 78, ey + 18), et, fill=INK, font=font(30 if not is_vert else 24, "serif"))
                draw.text((pad_x + 78, ey + 62), ev_txt, fill=STONE, font=font(22 if not is_vert else 18, "sans_reg"))
    else:
        # Quiet dusk coda card anchored in lower third while Elizabeth Tower glows cleanly above
        card_h = 260 if not is_vert else 320
        cy0 = bot_y - card_h - 48
        draw.rounded_rectangle([pad_x, cy0, pad_x + stage_w, cy0 + card_h], radius=14, fill="#161D26", outline=AMBER, width=2)
        cx = W // 2
        draw.text((cx, cy0 + 68), "Hear the score, solo every partial & read the preprint", fill="#F6F3EB", font=font(32 if not is_vert else 26, "serif"), anchor="mm")
        draw.text((cx, cy0 + 132), "zidingz.com/the-chords-inside-big-ben", fill=AMBER, font=font(30 if not is_vert else 24, "sans"), anchor="mm")
        draw.text((cx, cy0 + 190), "zidingz.com/big-ben-from-little-deans-yard", fill="#D4CEC2", font=font(24 if not is_vert else 20, "sans_reg"), anchor="mm")

    sub_lbl = ev["label"] if ev else "Listening across Westminster..."
    draw.text((pad_x, bot_y - 22), sub_lbl, fill=mut, font=font(20 if not is_vert else 18, "sans_reg"))
    draw.text((W - pad_x, bot_y - 22), f"{t:04.1f} s", fill=mut, font=font(20, "sans_reg"), anchor="ra")
    return img


def render_cut(aspect: str, wav_path: Path, events: list[dict], out_mp4: Path, out_poster: Path) -> None:
    w, h = (1080, 1920) if aspect == "9x16" else (1920, 1080)
    total_frames = int(round(DURATION_S * FPS))
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
        "-i", str(wav_path),
        "-c:v", "libx264", "-crf", "21", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    poster_saved = False
    for f_idx in range(total_frames):
        t = f_idx / FPS
        im = render_frame(t, events, aspect)
        proc.stdin.write(im.tobytes())
        if not poster_saved and t >= 0.85:
            im.save(out_poster, format="WEBP", quality=90)
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
