#!/usr/bin/env python3
"""
Stateless dual-format (16:9 & native 9:16) broadsheet-cinema video engine for Big Ben Harmonics.
Built on a pure time contract render_frame(t, events, aspect) -> RGB PIL Image:
  - Start very simple -> explain the physical mechanism -> build up -> settle on measured facts
  - Slower, spacious editorial pacing with minimal typography & generous #FAF9F5 paper margins
  - Subpixel PyCairo vector curves, brass/bell cross-sections, & wave interference geometry
  - Four vectorized physical transitions ported from zidingz-cinema: line, fold, wash, daybreak
"""

from __future__ import annotations

import os
import shutil
import sys

def _ensure_deps() -> None:
    try:
        import cairo  # noqa: F401
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        return
    except ModuleNotFoundError:
        uv_bin = shutil.which("uv")
        if uv_bin and os.environ.get("BB_UV_BOOT") != "1":
            env = dict(os.environ, BB_UV_BOOT="1")
            args = [
                uv_bin, "run", "--offline",
                "--with", "pycairo",
                "--with", "opencv-python-headless",
                "--with", "numpy",
                "--with", "scipy",
                "--with", "Pillow",
                "python3", *sys.argv,
            ]
            try:
                os.execvpe(uv_bin, args, env)
            except OSError:
                pass


_ensure_deps()

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cairo
import cv2
import numpy as np
from PIL import Image, ImageEnhance

FPS = 30
DURATION_S = 56.00

# Broadsheet & copperplate palette (matches visual-suite & westminster-engraving)
PAPER_HEX = "#FAF9F5"
INK_RGB = (20 / 255.0, 20 / 255.0, 19 / 255.0)
STONE_RGB = (107 / 255.0, 106 / 255.0, 100 / 255.0)
RULE_RGB = (212 / 255.0, 207 / 255.0, 194 / 255.0)
CLAY_RGB = (186 / 255.0, 99 / 255.0, 72 / 255.0)
SLATE_RGB = (61 / 255.0, 90 / 255.0, 108 / 255.0)
SAGE_RGB = (88 / 255.0, 117 / 255.0, 96 / 255.0)
GOLD_RGB = (158 / 255.0, 118 / 255.0, 40 / 255.0)
PLUM_RGB = (146 / 255.0, 68 / 255.0, 94 / 255.0)

NIGHT_BG_TOP = (10 / 255.0, 16 / 255.0, 28 / 255.0)
NIGHT_BG_BOT = (18 / 255.0, 29 / 255.0, 46 / 255.0)
NIGHT_STONE = (24 / 255.0, 34 / 255.0, 49 / 255.0)
NIGHT_EDGE = (118 / 255.0, 150 / 255.0, 184 / 255.0)
NIGHT_TEXT = (244 / 255.0, 240 / 255.0, 230 / 255.0)
NIGHT_MUTED = (168 / 255.0, 179 / 255.0, 194 / 255.0)
NIGHT_GOLD = (244 / 255.0, 186 / 255.0, 98 / 255.0)
NIGHT_FROST = (108 / 255.0, 184 / 255.0, 232 / 255.0)
NIGHT_CLAY = (224 / 255.0, 120 / 255.0, 86 / 255.0)

BB_SPECTRUM = [
    ("200.26 Hz", "G3 (+37c) hum", 200.26, 29.03, -2.72, -6.0, 0.0, GOLD_RGB),
    ("335.51 Hz", "E4 (+31c) nominal", 335.51, 14.17, -2.59, -1.1, -16.8, CLAY_RGB),
    ("499.75 Hz", "B4 (+20c) quint", 499.75, 17.33, 0.0, 0.0, -5.2, SLATE_RGB),
    ("677.60 Hz", "E5 (+48c) octave", 677.60, 9.28, -9.21, 0.0, -36.6, SAGE_RGB),
    ("883.34 Hz", "A5 (+7c) super-3rd", 883.34, 7.96, -13.29, -0.4, -41.3, PLUM_RGB),
]

QUARTER_DATA = [
    ("Q1", "G#4", 869.75, 513.66, 0.591, CLAY_RGB),
    ("Q2", "F#4", 770.97, 454.50, 0.590, GOLD_RGB),
    ("Q3", "E4", 678.99, 402.60, 0.593, SAGE_RGB),
    ("Q4", "B3", 505.14, 298.72, 0.591, SLATE_RGB),
]

BLOG_ROOT = Path(__file__).resolve().parents[4]
STILL_PATH = BLOG_ROOT / "public/images/the-chords-inside-big-ben/still.webp"

_PAPER_CACHE: dict[tuple[int, int], np.ndarray] = {}
_DUSK_CACHE: dict[tuple[int, int, str], np.ndarray] = {}
_BOKEH_KERNEL: dict[int, np.ndarray] = {}

# Blog 56 s cut acts & transition windows
BLOG_TRANSITIONS = [
    # (idx_from, idx_to, t0, t1, kind)
    (1, 2, 11.35, 12.65, "line"),
    (2, 3, 23.35, 24.65, "fold"),
    (3, 4, 35.85, 37.15, "wash"),
    (4, 5, 46.85, 48.15, "daybreak"),
]

BLOG_ACT_BOUNDS = {
    1: (0.00, 12.00),
    2: (12.00, 24.00),
    3: (24.00, 36.50),
    4: (36.50, 47.50),
    5: (47.50, 56.00),
}


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0: float, e1: float, x: float) -> float:
    if e1 <= e0:
        return 1.0 if x >= e1 else 0.0
    t = clamp01((x - e0) / (e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def ease_out3(x: float) -> float:
    t = clamp01(x)
    return 1.0 - (1.0 - t) ** 3


def ease_io3(x: float) -> float:
    t = clamp01(x)
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) * 0.5


def get_paper_plate(W: int, H: int) -> np.ndarray:
    """Cached warm rag-paper base (#FAF9F5) with fine physical fiber grain (float32 RGB [0,1])."""
    key = (W, H)
    if key not in _PAPER_CACHE:
        rng = np.random.default_rng(1858)
        base = np.empty((H, W, 3), dtype=np.float32)
        base[:, :, 0] = 250.0 / 255.0
        base[:, :, 1] = 249.0 / 255.0
        base[:, :, 2] = 245.0 / 255.0
        # Subtle radial paper vignette + two-octave fiber grain
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = ((xx / W - 0.5) ** 2 + (yy / H - 0.48) ** 2)
        vig = (1.0 - 0.065 * rr)[:, :, None]
        hf = rng.normal(0.0, 0.008, (H, W, 1)).astype(np.float32)
        lf_small = rng.normal(0.0, 0.010, (max(4, H // 6), max(4, W // 6), 1)).astype(np.float32)
        lf = cv2.resize(lf_small, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, None]
        plate = np.clip(base * vig + hf + lf, 0.0, 1.0)
        _PAPER_CACHE[key] = plate
    return _PAPER_CACHE[key]


def get_dusk_plate_arr(W: int, H: int, mode: str = "coda") -> np.ndarray:
    """Cached Little Dean's Yard cel-and-gouache plate as float32 RGB [0, 1]."""
    key = (W, H, mode)
    if key not in _DUSK_CACHE:
        if STILL_PATH.exists():
            im = Image.open(STILL_PATH).convert("RGB")
            sw, sh = im.size
            target_ar = W / H
            src_ar = sw / sh
            if target_ar < src_ar:
                new_w = int(sh * target_ar)
                x0 = int((sw - new_w) * 0.56)
                im = im.crop((x0, 0, x0 + new_w, sh))
            else:
                new_h = int(sw / target_ar)
                y0 = int((sh - new_h) * 0.25)
                im = im.crop((0, y0, sw, y0 + new_h))
            im = im.resize((W, H), Image.Resampling.LANCZOS)
            if mode == "dark":
                im = ImageEnhance.Brightness(im).enhance(0.62)
            elif mode == "coda":
                im = ImageEnhance.Brightness(im).enhance(0.92)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        else:
            arr = np.zeros((H, W, 3), dtype=np.float32)
            arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = NIGHT_BG_TOP
        if mode == "coda":
            # Soft left/bottom dusk gradient scrim only where minimal title rests—no opaque card
            yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
            xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
            grad = np.clip((1.0 - xx) * 0.45 + np.clip(yy - 0.45, 0.0, 1.0) * 0.85, 0.0, 0.68)[:, :, None]
            scrim = np.array([14 / 255.0, 20 / 255.0, 30 / 255.0], dtype=np.float32)
            arr = arr * (1.0 - grad) + scrim * grad
        _DUSK_CACHE[key] = arr
    return _DUSK_CACHE[key]


def make_cairo_surface(W: int, H: int, bg_rgb: Optional[np.ndarray] = None) -> tuple[cairo.ImageSurface, cairo.Context, np.ndarray]:
    """Allocate ARGB32 surface initialized cleanly with paper or nocturnal slate background."""
    buf = np.empty((H, W, 4), dtype=np.uint8)
    if bg_rgb is not None:
        buf[:, :, 0] = np.clip(bg_rgb[:, :, 2] * 255.0, 0, 255).astype(np.uint8)
        buf[:, :, 1] = np.clip(bg_rgb[:, :, 1] * 255.0, 0, 255).astype(np.uint8)
        buf[:, :, 2] = np.clip(bg_rgb[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
        buf[:, :, 3] = 255
    else:
        buf[:, :, :] = 255
    surf = cairo.ImageSurface.create_for_data(buf, cairo.FORMAT_ARGB32, W, H, W * 4)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    return surf, ctx, buf


def surface_to_float_rgb(buf: np.ndarray) -> np.ndarray:
    """Zero-alloc BGR(A)->RGB float32 [0,1] array from Cairo image buffer."""
    return buf[:, :, 2::-1].astype(np.float32) * (1.0 / 255.0)


def surface_to_uint8_rgb(buf: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(buf[:, :, 2::-1])


def set_rgb(ctx: cairo.Context, rgb: tuple[float, float, float], alpha: float = 1.0) -> None:
    ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], alpha)


def draw_text(
    ctx: cairo.Context,
    text: str,
    x: float,
    y: float,
    size: float,
    color: tuple[float, float, float],
    alpha: float = 1.0,
    family: str = "Georgia",
    bold: bool = False,
    italic: bool = False,
    align: str = "left",
    valign: str = "baseline",
) -> None:
    if alpha <= 0.005 or not text:
        return
    slant = cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL
    weight = cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL
    ctx.select_font_face(family, slant, weight)
    ctx.set_font_size(size)
    ext = ctx.text_extents(text)
    dx = 0.0
    if align == "center":
        dx = -(ext.x_bearing + ext.width * 0.5)
    elif align == "right":
        dx = -ext.x_advance
    dy = 0.0
    if valign == "middle":
        dy = -(ext.y_bearing + ext.height * 0.5)
    elif valign == "top":
        dy = -ext.y_bearing
    ctx.move_to(x + dx, y + dy)
    ctx.set_source_rgba(color[0], color[1], color[2], clamp01(alpha))
    ctx.show_text(text)
    ctx.new_path()


def draw_arrow(
    ctx: cairo.Context,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[float, float, float],
    alpha: float = 1.0,
    width: float = 2.0,
    head: float = 10.0,
) -> None:
    if alpha <= 0.005:
        return
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 2.0:
        return
    ux, uy = dx / dist, dy / dist
    ctx.save()
    ctx.set_source_rgba(color[0], color[1], color[2], clamp01(alpha))
    ctx.set_line_width(width)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.move_to(x0, y0)
    ctx.line_to(x1 - ux * head * 0.6, y1 - uy * head * 0.6)
    ctx.stroke()
    px, py = -uy, ux
    ctx.move_to(x1, y1)
    ctx.line_to(x1 - ux * head + px * head * 0.45, y1 - uy * head + py * head * 0.45)
    ctx.line_to(x1 - ux * head - px * head * 0.45, y1 - uy * head - py * head * 0.45)
    ctx.close_path()
    ctx.fill()
    ctx.restore()


def draw_folio_rules(ctx: cairo.Context, W: int, H: int, pad_x: float, top_y: float, bot_y: float, dark: bool = False) -> None:
    col = NIGHT_EDGE if dark else INK_RGB
    a = 0.24 if dark else 0.16
    ctx.save()
    ctx.set_source_rgba(col[0], col[1], col[2], a)
    ctx.set_line_width(1.2)
    ctx.move_to(pad_x, top_y - 34)
    ctx.line_to(W - pad_x, top_y - 34)
    ctx.stroke()
    ctx.set_line_width(0.8)
    ctx.move_to(pad_x, bot_y)
    ctx.line_to(W - pad_x, bot_y)
    ctx.stroke()
    ctx.restore()


def draw_header_text(
    ctx: cairo.Context,
    W: int,
    pad_x: float,
    top_y: float,
    kicker: str,
    headline: str,
    t_in: float,
    act_dur: float,
    is_vert: bool,
    dark: bool = False,
    kicker_alpha: float = 1.0,
    retire_dur: float = 0.55,
) -> None:
    """Single uncluttered kicker + short serif headline with masked rise & clean retirement before transition."""
    fg = NIGHT_TEXT if dark else INK_RGB
    mut = NIGHT_MUTED if dark else STONE_RGB
    # Fade out cleanly over final `retire_dur` before outgoing transition starts
    fade_in = smoothstep(0.05, 0.60, t_in)
    fade_out = 1.0 - smoothstep(max(0.8, act_dur - retire_dur), act_dur, t_in)
    a = clamp01(fade_in * fade_out)
    rise = (1.0 - ease_out3(clamp01((t_in - 0.05) / 0.55))) * 16.0

    draw_text(
        ctx,
        kicker,
        pad_x,
        top_y,
        18 if is_vert else 19,
        mut,
        alpha=a * 0.90 * kicker_alpha,
        family="Menlo",
        bold=True,
        valign="top",
    )
    h_size = 44 if is_vert else 56
    max_w = float(W) - 2.0 * pad_x
    ctx.select_font_face("Georgia", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(h_size)
    ext = ctx.text_extents(headline)
    if ext.x_advance > max_w:
        scale_fit = max(0.76, max_w / max(1.0, ext.x_advance))
        h_size = max(33.0, h_size * scale_fit)
        ctx.set_font_size(h_size)
        ext = ctx.text_extents(headline)
    if ext.x_advance > max_w and " " in headline:
        # Balanced 2-line split only if compact scaling wasn't enough
        words = headline.split()
        mid = len(words) // 2
        best = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        l1, l2 = " ".join(words[:best]), " ".join(words[best:])
        draw_text(ctx, l1, pad_x, top_y + 34 + rise, h_size, fg, alpha=a, family="Georgia", bold=True, valign="top")
        draw_text(ctx, l2, pad_x, top_y + 34 + rise + h_size * 1.15, h_size, fg, alpha=a, family="Georgia", bold=True, valign="top")
    else:
        draw_text(
            ctx,
            headline,
            pad_x,
            top_y + 34 + rise,
            h_size,
            fg,
            alpha=a,
            family="Georgia",
            bold=True,
            valign="top",
        )


def layout_metrics(W: int, H: int, is_vert: bool) -> dict[str, float]:
    pad_x = 76.0 if is_vert else 128.0
    top_y = 148.0 if is_vert else 88.0
    bot_y = float(H) - (176.0 if is_vert else 76.0)
    stage_y = top_y + (136.0 if is_vert else 144.0)
    stage_h = bot_y - stage_y - 24.0
    stage_w = float(W) - 2.0 * pad_x
    return {"pad_x": pad_x, "top_y": top_y, "bot_y": bot_y, "sy": stage_y, "sh": stage_h, "sw": stage_w}


# ==============================================================================
# Reusable Pedagogical Scenes (Shared between Blog 56s Cut & YouTube 84s Cut)
# ==============================================================================


def render_quarters_scene(t_rel: float, act_dur: float, t_abs: float, aspect: str, mode_events: list[float]) -> np.ndarray:
    """YouTube Chapter 1 (14 s): 4 Warner bells in E major -> damperless air overlap -> 1969 tierce table."""
    is_vert = (aspect == "9x16")
    W, H = (1080, 1920) if is_vert else (1920, 1080)
    surf, ctx, buf = make_cairo_surface(W, H, get_paper_plate(W, H))
    m = layout_metrics(W, H, is_vert)
    draw_folio_rules(ctx, W, H, m["pad_x"], m["top_y"], m["bot_y"], dark=False)

    # Headline progression
    if t_rel < 4.5:
        head = "Four bells in E major strike the quarters."
    elif t_rel < 9.2:
        head = "Bells have no dampers; notes blur in air."
    else:
        head = "All four tierces sit 26 cents flat."
    draw_header_text(ctx, W, m["pad_x"], m["top_y"], "WESTMINSTER QUARTERS · 1857 WARNER BELLS", head, t_rel, act_dur, is_vert)

    sy, sh, sw, px = m["sy"], m["sh"], m["sw"], m["pad_x"]
    mech_p = smoothstep(4.2, 5.2, t_rel)
    tech_p = smoothstep(8.8, 9.8, t_rel)

    row_h = (sh * (0.72 if is_vert else 0.78)) / 4.0
    start_y = sy + (sh * 0.08)

    for i, (bid, note, fnom, ftier, tratio, col) in enumerate(QUARTER_DATA):
        cy = start_y + (i + 0.5) * row_h
        # Find active bell envelope across repeated strikes in Chapter 1
        amp = 0.0
        imp = 0.0
        for b_idx, bt in enumerate(mode_events):
            # Quarter 1: Q1, Q2, Q3, Q4 (indices 0..3); Quarter 2: Q3, Q1, Q2, Q4 (indices 4..7)
            hit_map = [0, 1, 2, 3, 2, 0, 1, 3]
            if b_idx < len(hit_map) and hit_map[b_idx] == i and t_rel >= bt:
                dt = t_rel - bt
                amp = max(amp, math.exp(-dt / 2.2))
                imp = max(imp, math.exp(-dt / 0.32))

        # Simple circular bronze bell icon on left
        bx = px + (88.0 if not is_vert else 66.0)
        r_base = (30.0 if not is_vert else 24.0) * (1.0 + 0.06 * i)
        if amp > 0.01:
            # Expanding crisp acoustic ring on strike
            ring_r = r_base + (1.0 - amp) * 34.0
            ctx.set_source_rgba(col[0], col[1], col[2], clamp01(0.55 * amp))
            ctx.set_line_width(2.0)
            ctx.arc(bx, cy, ring_r, 0, math.tau)
            ctx.stroke()

        # Vibrating rim profile
        ctx.new_path()
        for s in range(65):
            ang = math.tau * s / 64.0
            wob = amp * (3.2 * math.cos(2 * ang) * math.sin(t_abs * (18.0 - 2.0 * i)) + 1.6 * imp * math.cos(3 * ang))
            rr = r_base + wob
            vx = bx + rr * math.cos(ang)
            vy = cy + rr * math.sin(ang)
            ctx.move_to(vx, vy) if s == 0 else ctx.line_to(vx, vy)
        ctx.close_path()
        ctx.set_source_rgba(col[0], col[1], col[2], 0.14 if amp > 0.02 else 0.04)
        ctx.fill_preserve()
        ctx.set_source_rgba(col[0], col[1], col[2], 0.92 if amp > 0.02 else 0.40)
        ctx.set_line_width(2.8 if amp > 0.02 else 1.8)
        ctx.stroke()

        # Minimal note label
        lbl_x = bx + r_base + 34.0
        draw_text(ctx, f"{note}", lbl_x, cy - 10, 30 if not is_vert else 26, INK_RGB, family="Georgia", bold=True, valign="middle")
        draw_text(ctx, f"{fnom:.2f} Hz", lbl_x, cy + 16, 17 if not is_vert else 15, STONE_RGB, family="Menlo", valign="middle")

        # Mechanism -> overlapping sustain ribbons across horizontal timeline
        track_x = lbl_x + (165.0 if not is_vert else 135.0)
        track_w = (px + sw) - track_x - (240.0 * tech_p if not is_vert else 0.0)
        if track_w > 120:
            ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], 0.55)
            ctx.set_line_width(1.2)
            ctx.move_to(track_x, cy)
            ctx.line_to(track_x + track_w, cy)
            ctx.stroke()
            if amp > 0.01:
                ribbon_w = track_w * min(1.0, 0.18 + 0.82 * amp) * (0.45 + 0.55 * mech_p)
                grad = cairo.LinearGradient(track_x, cy, track_x + ribbon_w, cy)
                grad.add_color_stop_rgba(0.0, col[0], col[1], col[2], 0.88)
                grad.add_color_stop_rgba(1.0, col[0], col[1], col[2], 0.08)
                ctx.set_source(grad)
                ctx.rectangle(track_x, cy - 11.0 * amp - 2.0, ribbon_w, 22.0 * amp + 4.0)
                ctx.fill()

        # Technical callout appearing in Stage 3
        if tech_p > 0.02:
            if not is_vert:
                draw_text(
                    ctx,
                    f"tierce {ftier:.1f} Hz ({tratio:.3f})",
                    px + sw,
                    cy,
                    18,
                    col,
                    alpha=tech_p,
                    family="Menlo",
                    bold=True,
                    align="right",
                    valign="middle",
                )
            else:
                draw_text(
                    ctx,
                    f"tierce {ftier:.1f} Hz ({tratio:.3f})",
                    px + sw,
                    cy + 28,
                    15,
                    col,
                    alpha=tech_p,
                    family="Menlo",
                    align="right",
                    valign="middle",
                )

    if tech_p > 0.05:
        draw_text(
            ctx,
            "1969 BBC WAVs · Simpson minor-third ideal 0.600 vs measured 0.590-0.593 · 40 strokes per hour",
            px,
            m["bot_y"] - 22,
            18 if not is_vert else 15,
            STONE_RGB,
            alpha=tech_p,
            family="Menlo",
        )

    return surface_to_float_rgb(buf)


def render_scene_1_voices(t_rel: float, act_dur: float, t_abs: float, aspect: str, t_pure: float = 0.70, t_bell: float = 3.90) -> np.ndarray:
    """
    Pedagogical Scene 1:
      Simple (0..t_bell): One taut string vibrating in one clean pitch (335.51 Hz).
      Mechanism (t_bell..t_spec): Circular bronze bell shell flexing in multiple modes at once.
      Technical (t_spec..end): Pulls smoothly out into 5 measured 1969 BBC modal voices.
    """
    is_vert = (aspect == "9x16")
    W, H = (1080, 1920) if is_vert else (1920, 1080)
    surf, ctx, buf = make_cairo_surface(W, H, get_paper_plate(W, H))
    m = layout_metrics(W, H, is_vert)
    draw_folio_rules(ctx, W, H, m["pad_x"], m["top_y"], m["bot_y"], dark=False)

    t_mech = t_bell
    t_spec = t_bell + 3.20

    if t_rel < t_mech:
        head = "One string plays one pure pitch."
    elif t_rel < t_spec:
        head = "A bronze bell flexes in many shapes."
    else:
        head = "One blow wakes five measured voices."
    draw_header_text(ctx, W, m["pad_x"], m["top_y"], "MECHANISM I · MODAL VIBRATION", head, t_rel, act_dur, is_vert)

    sy, sh, sw, px = m["sy"], m["sh"], m["sw"], m["pad_x"]

    morph_bell = ease_out3(clamp01((t_rel - t_mech + 0.06) / 0.85)) if t_rel + 1e-4 >= t_mech else 0.0
    spec_open = ease_out3(clamp01((t_rel - t_spec) / 1.10))

    # Stage 1 fades out smoothly into Stage 2 circular bell
    str_alpha = 1.0 - smoothstep(t_mech - 0.1, t_mech + 0.55, t_rel)
    if str_alpha > 0.01:
        x0 = px + sw * 0.14
        x1 = px + sw * 0.86
        cy_s = sy + sh * 0.48
        # Pluck amplitude tied cleanly to t_pure (cosine release -> instant maximum deflection on frame f_ev)
        if t_rel + 1e-4 >= t_pure:
            dt_p = max(0.0, t_rel - t_pure)
            amp_s = (78.0 if not is_vert else 72.0) * math.exp(-dt_p / 2.2)
            phase = math.cos(2.0 * math.pi * 6.5 * dt_p)
        else:
            amp_s = 0.0
            phase = 0.0

        # End pins & equilibrium baseline
        ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], str_alpha)
        ctx.set_line_width(1.5)
        ctx.move_to(x0, cy_s)
        ctx.line_to(x1, cy_s)
        ctx.stroke()

        # Subtle warm acoustic glow halo under string right on pluck
        if t_rel + 1e-4 >= t_pure:
            imp_p = math.exp(-max(0.0, t_rel - t_pure) / 0.35)
            ctx.set_source_rgba(GOLD_RGB[0], GOLD_RGB[1], GOLD_RGB[2], 0.14 * imp_p * str_alpha)
            ctx.arc((x0 + x1) * 0.5, cy_s, 110.0 * (1.0 - 0.5 * imp_p) + 40.0, 0, math.tau)
            ctx.fill()

        # Vibrating fundamental loop (n = 1)
        ctx.new_path()
        for k in range(121):
            u = k / 120.0
            xx = x0 + u * (x1 - x0)
            yy = cy_s - amp_s * math.sin(math.pi * u) * phase
            ctx.move_to(xx, yy) if k == 0 else ctx.line_to(xx, yy)
        ctx.set_source_rgba(SLATE_RGB[0], SLATE_RGB[1], SLATE_RGB[2], str_alpha)
        ctx.set_line_width(4.2)
        ctx.stroke()

        for pin_x in (x0, x1):
            ctx.set_source_rgba(INK_RGB[0], INK_RGB[1], INK_RGB[2], str_alpha)
            ctx.arc(pin_x, cy_s, 6.5, 0, math.tau)
            ctx.fill()

        if t_rel >= t_pure + 0.15:
            draw_text(
                ctx,
                "335.51 Hz · Single fundamental loop (n = 1)",
                (x0 + x1) * 0.5,
                cy_s + (116.0 if not is_vert else 108.0),
                22 if not is_vert else 20,
                SLATE_RGB,
                alpha=str_alpha * smoothstep(t_pure + 0.15, t_pure + 0.65, t_rel),
                family="Menlo",
                align="center",
            )

    # Stage 2 & 3: Flexing circular bell shell on left/top -> 5 measured spectral stems on right/bottom
    if morph_bell > 0.01:
        dt_b = max(0.0, t_rel - t_bell)
        imp_b = math.exp(-dt_b / 0.42) if t_rel >= t_bell else 0.0
        ring_decay = math.exp(-dt_b / 6.5) if t_rel >= t_bell else 0.0

        if is_vert:
            cx_b = W * 0.5
            cy_b = sy + sh * (0.44 - 0.18 * spec_open)
            r_b = 148.0 - 24.0 * spec_open
        else:
            cx_b = px + sw * (0.50 - 0.31 * spec_open)
            cy_b = sy + sh * 0.48
            r_b = 188.0 - 32.0 * spec_open

        # Concentric acoustic wave ripples pulsing outward at strike frame
        for k in range(3):
            w_t = dt_b - k * 0.24
            if 0.0 <= w_t < 1.8:
                rr_w = r_b + w_t * 78.0
                wa = math.exp(-w_t / 0.75) * 0.34 * morph_bell
                ctx.set_source_rgba(GOLD_RGB[0], GOLD_RGB[1], GOLD_RGB[2], wa)
                ctx.set_line_width(2.0)
                ctx.arc(cx_b, cy_b, rr_w, 0, math.tau)
                ctx.stroke()

        # Reference unflexed rim circle
        ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], 0.80 * morph_bell)
        ctx.set_line_width(1.6)
        ctx.arc(cx_b, cy_b, r_b, 0, math.tau)
        ctx.stroke()

        # Superposed eigenmode boundary (n = 2 hum, 3 nominal, 4 quint, 5 super-third)
        modes_geom = [
            (2, 2.6, 1.00, GOLD_RGB),
            (3, 4.1, 0.85, CLAY_RGB),
            (4, 5.7, 0.75, SLATE_RGB),
            (6, 8.2, 0.60, PLUM_RGB),
        ]
        # Draw subtle individual ghosts when in Stage 2 before spectrum pulls out
        ghost_fade = morph_bell * (1.0 - 0.65 * spec_open)
        for n_lobes, omega, a_rel, gcol in modes_geom:
            ctx.new_path()
            for s in range(145):
                ang = math.tau * s / 144.0
                dr = (11.0 + 7.0 * imp_b) * a_rel * ring_decay * math.sin(n_lobes * ang) * math.sin(t_abs * omega)
                rr = r_b + dr
                px_r = cx_b + rr * math.cos(ang)
                py_r = cy_b + rr * math.sin(ang)
                ctx.move_to(px_r, py_r) if s == 0 else ctx.line_to(px_r, py_r)
            ctx.close_path()
            ctx.set_source_rgba(gcol[0], gcol[1], gcol[2], 0.42 * ghost_fade)
            ctx.set_line_width(2.0)
            ctx.stroke()

        # Bold composite bronze rim
        ctx.new_path()
        for s in range(181):
            ang = math.tau * s / 180.0
            dr = 0.0
            for n_lobes, omega, a_rel, _ in modes_geom:
                dr += (7.5 + 4.5 * imp_b) * a_rel * ring_decay * math.sin(n_lobes * ang) * math.sin(t_abs * omega)
            rr = r_b + dr
            px_r = cx_b + rr * math.cos(ang)
            py_r = cy_b + rr * math.sin(ang)
            ctx.move_to(px_r, py_r) if s == 0 else ctx.line_to(px_r, py_r)
        ctx.close_path()
        ctx.set_source_rgba(158 / 255.0, 118 / 255.0, 40 / 255.0, 0.08 * morph_bell)
        ctx.fill_preserve()
        ctx.set_source_rgba(INK_RGB[0], INK_RGB[1], INK_RGB[2], 0.90 * morph_bell)
        ctx.set_line_width(3.4)
        ctx.stroke()

        caption_bell = "200 Hz (ellipse) + 335 Hz (trefoil) + ..."
        draw_text(
            ctx,
            caption_bell,
            cx_b,
            cy_b + r_b + 38.0,
            18 if not is_vert else 17,
            STONE_RGB,
            alpha=morph_bell * (1.0 - spec_open * 0.4),
            family="Menlo",
            align="center",
        )

        # Stage 3: Pull out cleanly into the 5 measured 1969 BBC spectral bars
        if spec_open > 0.01:
            if is_vert:
                sp_x = px
                sp_w = sw
                sp_y = cy_b + r_b + 74.0
                sp_h = (sy + sh) - sp_y - 36.0
            else:
                sp_x = px + sw * 0.42
                sp_w = sw * 0.58
                sp_y = sy + sh * 0.14
                sp_h = sh * 0.72

            row_sp = sp_h / len(BB_SPECTRUM)
            lbl_col_w = 240.0 if not is_vert else 210.0
            bar_x0 = sp_x + lbl_col_w
            bar_max_w = sp_w - lbl_col_w - 130.0

            # Guide rule showing 0 dB vs decay
            draw_text(
                ctx,
                "1969 BBC BELFRY PARTIALS (T60 = 60 dB DECAY TIME)",
                sp_x,
                sp_y - 18.0,
                16 if not is_vert else 14,
                STONE_RGB,
                alpha=spec_open,
                family="Menlo",
                bold=True,
            )

            for i, (f_txt, role_txt, fz, t60, db_rel, _, _, pcol) in enumerate(BB_SPECTRUM):
                row_p = ease_out3(clamp01((spec_open - i * 0.08) / 0.55))
                if row_p <= 0.01:
                    continue
                ry = sp_y + (i + 0.5) * row_sp

                # Connector hairline subtly bridging bell to spectrum
                if not is_vert and row_p > 0.1:
                    ctx.set_source_rgba(pcol[0], pcol[1], pcol[2], 0.22 * row_p)
                    ctx.set_line_width(1.2)
                    ctx.move_to(cx_b + r_b + 14.0, cy_b + (i - 2) * 18.0)
                    ctx.line_to(sp_x - 16.0, ry)
                    ctx.stroke()

                draw_text(ctx, f_txt, sp_x, ry - 9.0, 20 if not is_vert else 18, INK_RGB, alpha=row_p, family="Menlo", bold=True, valign="middle")
                draw_text(ctx, role_txt, sp_x, ry + 13.0, 17 if not is_vert else 15, STONE_RGB, alpha=row_p, family="Georgia", italic=True, valign="middle")

                # Calibrated initial impact level ribbon (-18..+1 dB span)
                norm_lvl = clamp01((db_rel + 18.0) / 19.0)
                if abs(fz - 883.34) < 5.0:
                    norm_lvl = clamp01((-0.4 + 18.0) / 19.0)  # Impact super-third peaks at -0.4 dB
                live_att = math.exp(-3.45 * max(0.0, t_rel - t_bell) / t60)
                fill_w = max(14.0, bar_max_w * norm_lvl * row_p * (0.28 + 0.72 * live_att))

                ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], 0.45 * row_p)
                ctx.set_line_width(1.2)
                ctx.move_to(bar_x0, ry)
                ctx.line_to(bar_x0 + bar_max_w, ry)
                ctx.stroke()

                grad = cairo.LinearGradient(bar_x0, ry, bar_x0 + fill_w, ry)
                grad.add_color_stop_rgba(0.0, pcol[0], pcol[1], pcol[2], 0.94 * row_p)
                grad.add_color_stop_rgba(1.0, pcol[0], pcol[1], pcol[2], 0.42 * row_p)
                ctx.set_source(grad)
                bh = 16.0 if not is_vert else 13.0
                ctx.rectangle(bar_x0, ry - bh * 0.5, fill_w, bh)
                ctx.fill()

                draw_text(
                    ctx,
                    f"T60 {t60:.1f}s",
                    sp_x + sp_w,
                    ry,
                    17 if not is_vert else 15,
                    pcol,
                    alpha=row_p,
                    family="Menlo",
                    bold=True,
                    align="right",
                    valign="middle",
                )

    return surface_to_float_rgb(buf)


def render_scene_2_doublet(t_rel: float, act_dur: float, t_abs: float, aspect: str, t_single: float = 1.10, t_split: float = 4.10) -> np.ndarray:
    """
    Pedagogical Scene 2 (Symmetry Breaking -> Beat Interference -> Measured 2.705 Hz Envelope):
      Simple: One symmetric ring vibrating evenly at 335.44 Hz.
      Mechanism: Cracked/asymmetric ring splits into two orthogonal waves drifting in and out of step.
      Technical: Time axis zooms smoothly out to reveal all 5 swells every 0.370 s + 1953/1969/2024 splits + caveat.
    """
    is_vert = (aspect == "9x16")
    W, H = (1080, 1920) if is_vert else (1920, 1080)
    surf, ctx, buf = make_cairo_surface(W, H, get_paper_plate(W, H))
    m = layout_metrics(W, H, is_vert)
    draw_folio_rules(ctx, W, H, m["pad_x"], m["top_y"], m["bot_y"], dark=False)

    t_zoom = t_split + 3.20
    if t_rel < t_split:
        head = "A symmetric bell rings evenly."
    elif t_rel < t_zoom:
        head = "Two split modes drift in and out of step."
    else:
        head = "2.705 Hz split: one swell every 0.370 s."
    draw_header_text(ctx, W, m["pad_x"], m["top_y"], "MECHANISM II · SYMMETRY BREAKING", head, t_rel, act_dur, is_vert)

    sy, sh, sw, px = m["sy"], m["sh"], m["sw"], m["pad_x"]
    split_p = ease_io3(clamp01((t_rel - t_split) / 0.85))
    zoom_p = ease_io3(clamp01((t_rel - t_zoom) / 1.35))

    # Left / Top ring cross-section illustrating nodal axis splitting
    if is_vert:
        cx_r, cy_r, r_r = W * 0.5, sy + 128.0, 96.0
        wx0, wx1 = px, px + sw
        wy_top = sy + 286.0
        wy_sum = sy + sh * 0.72
    else:
        cx_r, cy_r, r_r = px + 148.0, sy + sh * 0.42, 118.0
        wx0, wx1 = px + 340.0, px + sw
        wy_top = sy + sh * 0.24
        wy_sum = sy + sh * 0.70

    # Symmetric vs cracked bell ring
    active_t = t_split if t_rel + 1e-4 >= t_split else t_single
    dt_ev = max(0.0, t_rel - active_t)
    imp = math.exp(-dt_ev / 0.40) if t_rel + 1e-4 >= active_t else 0.0
    amp_ring = 1.0 if t_rel + 1e-4 >= t_single else 0.0

    # Concentric acoustic pulse halo on ring strike frames
    if t_rel + 1e-4 >= active_t:
        r_halo = r_r + dt_ev * 95.0
        col_h = CLAY_RGB if t_rel >= t_split else SLATE_RGB
        ctx.set_source_rgba(col_h[0], col_h[1], col_h[2], 0.36 * math.exp(-dt_ev / 0.55))
        ctx.set_line_width(3.0)
        ctx.arc(cx_r, cy_r, r_halo, 0, math.tau)
        ctx.stroke()

    ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], 0.8)
    ctx.set_line_width(1.4)
    ctx.arc(cx_r, cy_r, r_r, 0, math.tau)
    ctx.stroke()

    # Mode 1 (cosine axis, 335.44 Hz) in slate
    ctx.new_path()
    for s in range(121):
        ang = math.tau * s / 120.0
        dr = (9.5 + 5.0 * imp) * amp_ring * math.cos(2.0 * ang) * math.sin(t_abs * 5.2)
        vx, vy = cx_r + (r_r + dr) * math.cos(ang), cy_r + (r_r + dr) * math.sin(ang)
        ctx.move_to(vx, vy) if s == 0 else ctx.line_to(vx, vy)
    ctx.close_path()
    ctx.set_source_rgba(SLATE_RGB[0], SLATE_RGB[1], SLATE_RGB[2], 0.88)
    ctx.set_line_width(2.8)
    ctx.stroke()

    # Mode 2 (45-deg orthogonal axis, 332.735 Hz) appears cleanly on split + radial notch
    if split_p > 0.02:
        ctx.new_path()
        for s in range(121):
            ang = math.tau * s / 120.0
            dr = (7.5 + 4.0 * imp) * split_p * math.sin(2.0 * ang) * math.sin(t_abs * 4.55 + 0.6)
            vx, vy = cx_r + (r_r + dr) * math.cos(ang), cy_r + (r_r + dr) * math.sin(ang)
            ctx.move_to(vx, vy) if s == 0 else ctx.line_to(vx, vy)
        ctx.close_path()
        ctx.set_source_rgba(CLAY_RGB[0], CLAY_RGB[1], CLAY_RGB[2], 0.85 * split_p)
        ctx.set_line_width(2.6)
        ctx.stroke()

        # Subtle radial rib/crack mark on the rim at 45 degrees
        na = -math.pi * 0.25
        ctx.set_source_rgba(CLAY_RGB[0], CLAY_RGB[1], CLAY_RGB[2], split_p)
        ctx.set_line_width(4.0)
        ctx.move_to(cx_r + (r_r - 12) * math.cos(na), cy_r + (r_r - 12) * math.sin(na))
        ctx.line_to(cx_r + (r_r + 12) * math.cos(na), cy_r + (r_r + 12) * math.sin(na))
        ctx.stroke()

    draw_text(
        ctx,
        "1 mode (335.44 Hz)" if split_p < 0.3 else "Split into 2 axes",
        cx_r,
        cy_r + r_r + 28.0,
        16,
        CLAY_RGB if split_p >= 0.3 else SLATE_RGB,
        family="Menlo",
        bold=True,
        align="center",
    )

    # Right / Bottom Wave Visualization:
    # Wave window spans smoothly from 0.14 s (showing individual wave crests sliding!) to 1.85 s (5 full beats)
    span_s = 0.14 + zoom_p * (1.85 - 0.14)
    w_span = wx1 - wx0
    n_pts = 320

    # Fade component sine waves gently as we zoom out into the measured envelope
    sine_alpha = 1.0 - 0.82 * zoom_p
    if sine_alpha > 0.02:
        ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], 0.55 * sine_alpha)
        ctx.set_line_width(1.2)
        ctx.move_to(wx0, wy_top)
        ctx.line_to(wx1, wy_top)
        ctx.stroke()

        # Wave 1 (335.440 Hz scaled into visual cycles)
        amp_w = (38.0 if not is_vert else 30.0) * amp_ring
        ctx.new_path()
        for k in range(n_pts):
            u = k / (n_pts - 1)
            tau = u * span_s
            xx = wx0 + u * w_span
            # Visualized carrier cycles compressed gently as we zoom out
            cyc1 = 2.0 * math.pi * (tau * (10.0 / 0.14) if span_s < 0.4 else tau * 28.0) - t_abs * 3.8
            yy = wy_top - amp_w * math.sin(cyc1)
            ctx.move_to(xx, yy) if k == 0 else ctx.line_to(xx, yy)
        ctx.set_source_rgba(SLATE_RGB[0], SLATE_RGB[1], SLATE_RGB[2], 0.85 * sine_alpha)
        ctx.set_line_width(2.6)
        ctx.stroke()

        if split_p > 0.02:
            ctx.new_path()
            for k in range(n_pts):
                u = k / (n_pts - 1)
                tau = u * span_s
                xx = wx0 + u * w_span
                base_c = tau * (10.0 / 0.14) if span_s < 0.4 else tau * 28.0
                # Exact phase offset grows at 2.705 Hz relative to Mode 1!
                cyc2 = 2.0 * math.pi * (base_c - tau * 2.705) - t_abs * 3.8
                yy = wy_top - amp_w * 0.74 * math.sin(cyc2) * split_p
                ctx.move_to(xx, yy) if k == 0 else ctx.line_to(xx, yy)
            ctx.set_source_rgba(CLAY_RGB[0], CLAY_RGB[1], CLAY_RGB[2], 0.85 * sine_alpha * split_p)
            ctx.set_line_width(2.4)
            ctx.stroke()

        draw_text(
            ctx,
            "335.440 Hz (slate)  +  332.735 Hz (terracotta)" if split_p > 0.3 else "335.440 Hz steady sine wave",
            wx0,
            wy_top - amp_w - 22.0,
            17 if not is_vert else 15,
            STONE_RGB,
            alpha=sine_alpha,
            family="Menlo",
        )

    # Measured Sum & Envelope plot (grows organically below, rising upward as waves fade when zoomed!)
    env_y = wy_sum - zoom_p * (110.0 if not is_vert else 120.0)
    env_amp = (56.0 + 52.0 * zoom_p) if not is_vert else (48.0 + 46.0 * zoom_p)
    ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], 0.70)
    ctx.set_line_width(1.4)
    ctx.move_to(wx0, env_y)
    ctx.line_to(wx1, env_y)
    ctx.stroke()

    # Fill positive & negative envelope showing true 0.557 amplitude ratio
    pts_top, pts_bot = [], []
    r_amp = 0.557 * split_p
    for k in range(n_pts):
        u = k / (n_pts - 1)
        tau = u * span_s
        xx = wx0 + u * w_span
        # Exact interference amplitude for f2/f1 ratio = 0.557 and df = 2.705 Hz
        theta = 2.0 * math.pi * 2.705 * tau - (t_abs * 1.4 if zoom_p < 0.7 else 0.0)
        env = math.sqrt(1.0 + r_amp * r_amp + 2.0 * r_amp * math.cos(theta)) / (1.0 + 0.557)
        if zoom_p < 0.65:
            # Blend smoothly from steady line before split into beating sum wave
            env = 0.65 * (1.0 - split_p) + env * split_p
        pts_top.append((xx, env_y - env * env_amp * amp_ring))
        pts_bot.append((xx, env_y + env * env_amp * amp_ring))

    ctx.new_path()
    for i, (px_e, py_e) in enumerate(pts_top):
        ctx.move_to(px_e, py_e) if i == 0 else ctx.line_to(px_e, py_e)
    for px_e, py_e in reversed(pts_bot):
        ctx.line_to(px_e, py_e)
    ctx.close_path()
    ctx.set_source_rgba(CLAY_RGB[0], CLAY_RGB[1], CLAY_RGB[2], 0.16 * (0.25 + 0.75 * split_p))
    ctx.fill()

    for pts_line in (pts_top, pts_bot):
        ctx.new_path()
        for i, (px_e, py_e) in enumerate(pts_line):
            ctx.move_to(px_e, py_e) if i == 0 else ctx.line_to(px_e, py_e)
        ctx.set_source_rgba(CLAY_RGB[0], CLAY_RGB[1], CLAY_RGB[2], 0.92 * (0.35 + 0.65 * split_p))
        ctx.set_line_width(3.0)
        ctx.stroke()

    # Live playhead across the zoomed 1.85 s (5-beat) envelope once settled
    if zoom_p > 0.75:
        sweep_t = (max(0.0, t_rel - t_split) % 1.850) / 1.850
        cur_x = wx0 + sweep_t * w_span
        ctx.set_source_rgba(SLATE_RGB[0], SLATE_RGB[1], SLATE_RGB[2], 0.80 * zoom_p)
        ctx.set_line_width(2.2)
        ctx.move_to(cur_x, env_y - env_amp - 16)
        ctx.line_to(cur_x, env_y + env_amp + 16)
        ctx.stroke()

        # 0.370 s calibrated grid ticks (0 .. 1.850 s)
        for k in range(6):
            tk_x = wx0 + (k * 0.370 / 1.850) * w_span
            ctx.set_source_rgba(INK_RGB[0], INK_RGB[1], INK_RGB[2], 0.18 * zoom_p)
            ctx.set_line_width(1.0)
            ctx.move_to(tk_x, env_y - env_amp - 8)
            ctx.line_to(tk_x, env_y + env_amp + 8)
            ctx.stroke()
            align_tk = "left" if k == 0 else ("right" if k == 5 else "center")
            draw_text(ctx, f"{k * 0.370:.2f}s", tk_x, env_y + env_amp + 30.0, 15, STONE_RGB, alpha=zoom_p, family="Menlo", align=align_tk)

        draw_text(
            ctx,
            "SUM ENVELOPE · df = 2.705 Hz (0.370 s period, ratio 0.557)",
            wx0,
            env_y - env_amp - 24.0,
            17 if not is_vert else 15,
            INK_RGB,
            alpha=zoom_p,
            family="Menlo",
            bold=True,
        )

        # Era consistency + honest caveat below
        draw_text(
            ctx,
            "1953: 2.712 Hz   ·   1969: 2.705 Hz   ·   2024: 2.627 Hz",
            px,
            m["bot_y"] - (54.0 if is_vert else 46.0),
            18 if not is_vert else 16,
            SLATE_RGB,
            alpha=zoom_p,
            family="Menlo",
            bold=True,
        )
        draw_text(
            ctx,
            "Caveat: audio alone cannot separate the 1859 repair crack from wall asymmetry.",
            px,
            m["bot_y"] - 18.0,
            19 if not is_vert else 17,
            STONE_RGB,
            alpha=zoom_p,
            family="Georgia",
            italic=True,
        )

    return surface_to_float_rgb(buf)


def render_scene_3_decay(t_rel: float, act_dur: float, t_abs: float, aspect: str, t_fast: float = 1.20, t_slow: float = 3.80, t_full: float = 6.50) -> np.ndarray:
    """
    Pedagogical Scene 3 (Differential T60 -> 14 s Spectral Evolution + Terhardt Salience):
      Simple -> Mechanism: Side-by-side / stacked contrast: 883 Hz A5 fades fast (7.96 s) vs 200 Hz G3 hum (29.03 s).
      Technical Final: Morphs cleanly into full 0->14 s ink-and-wash spectrogram + Terhardt virtual-pitch bars.
    """
    is_vert = (aspect == "9x16")
    W, H = (1080, 1920) if is_vert else (1920, 1080)
    surf, ctx, buf = make_cairo_surface(W, H, get_paper_plate(W, H))
    m = layout_metrics(W, H, is_vert)
    draw_folio_rules(ctx, W, H, m["pad_x"], m["top_y"], m["bot_y"], dark=False)

    if t_rel < t_full:
        head = "High voices shed energy 3.6x faster."
    else:
        head = "At impact A4 clashes; the tail glows B4 + G3."
    draw_header_text(ctx, W, m["pad_x"], m["top_y"], "MECHANISM III · DIFFERENTIAL DECAY (T60)", head, t_rel, act_dur, is_vert)

    sy, sh, sw, px = m["sy"], m["sh"], m["sw"], m["pad_x"]
    full_p = ease_out3(clamp01((t_rel - t_full + 0.06) / 1.10)) if t_rel + 1e-4 >= t_full else 0.0

    # Stage 1–2: Clean two-row comparison fades smoothly into the 5-row spectrogram
    cmp_alpha = 1.0 - full_p
    if cmp_alpha > 0.01:
        rows_cmp = [
            ("883.34 Hz (A5 super-third)", 7.96, t_fast, PLUM_RGB, 0.30, 18.0),
            ("200.26 Hz (G3 hum) + 499.67 Hz (B4)", 29.03, t_slow, GOLD_RGB, 0.62, 5.5),
        ]
        for label, t60, t_hit, col, y_frac, k_freq in rows_cmp:
            ry = sy + sh * y_frac
            x0, x1 = px, px + sw
            draw_text(ctx, f"{label}   ·   T60 = {t60:.2f} s", x0, ry - 66.0, 20 if not is_vert else 17, col, alpha=cmp_alpha, family="Menlo", bold=True)
            ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], 0.70 * cmp_alpha)
            ctx.set_line_width(1.2)
            ctx.move_to(x0, ry)
            ctx.line_to(x1, ry)
            ctx.stroke()

            if t_rel + 1e-4 >= t_hit:
                dt = max(0.0, t_rel - t_hit)
                cur_u = min(1.0, (dt + 0.04) / 2.6)
                # Subtle radial node pulse on exact strike frame
                imp_c = math.exp(-dt / 0.35)
                ctx.set_source_rgba(col[0], col[1], col[2], 0.28 * imp_c * cmp_alpha)
                ctx.arc(x0 + 18.0, ry, 18.0 + (1.0 - imp_c) * 32.0, 0, math.tau)
                ctx.fill()
                ctx.new_path()
                n_w = 240
                for s in range(n_w):
                    u = (s / (n_w - 1)) * cur_u
                    # Represent 0 -> 14 s of natural decay over the drawn trace
                    sec = u * 14.0
                    env = 46.0 * math.exp(-3.45 * sec / t60)
                    yy = ry - env * math.sin(u * math.tau * k_freq + t_abs * 3.0)
                    xx = x0 + u * (x1 - x0)
                    ctx.move_to(xx, yy) if s == 0 else ctx.line_to(xx, yy)
                ctx.set_source_rgba(col[0], col[1], col[2], cmp_alpha)
                ctx.set_line_width(3.2)
                ctx.stroke()

    # Stage 3: Full 0 -> 14 s ink-and-wash spectrogram + Terhardt virtual-pitch bars
    if full_p > 0.01:
        dt_full = max(0.0, t_rel - t_full)
        # Sweep playhead across 0 -> 14 s over 5.2 seconds
        play_sec = min(14.0, (dt_full / 4.8) * 14.0)
        play_u = play_sec / 14.0

        if is_vert:
            sp_x, sp_w = px, sw
            sp_y, sp_h = sy + 28.0, sh * 0.44
            sal_x, sal_w = px, sw
            sal_y, sal_h = sp_y + sp_h + 76.0, sh * 0.38
        else:
            sp_x, sp_w = px + 230.0, sw * 0.56 - 230.0
            sp_y, sp_h = sy + 46.0, sh * 0.70
            sal_x, sal_w = px + sw * 0.62, sw * 0.38
            sal_y, sal_h = sp_y, sp_h

        draw_text(
            ctx,
            "MEASURED SPECTRAL WASH (0 -> 14 s)",
            sp_x if is_vert else px,
            sp_y - 26.0,
            16,
            STONE_RGB,
            alpha=full_p,
            family="Menlo",
            bold=True,
        )

        # Draw 5 partials (883 top -> 200 bottom, matching 04-fourteen-seconds)
        ordered = list(reversed(BB_SPECTRUM))  # 883, 678, 500, 335, 200
        row_h = sp_h / 5.0
        for idx, (f_txt, role_txt, _, t60, _, db0, db14, pcol) in enumerate(ordered):
            py = sp_y + (idx + 0.5) * row_h
            lbl_short = f"{f_txt}"
            if is_vert:
                draw_text(ctx, lbl_short, px, py - 18.0, 15, pcol, alpha=full_p, family="Menlo", bold=True, valign="middle")
                draw_text(ctx, f"T60 {t60:.1f}s", px + sw, py - 18.0, 14, STONE_RGB, alpha=full_p, family="Menlo", align="right", valign="middle")
            else:
                draw_text(ctx, lbl_short, px, py - 8.0, 18, pcol, alpha=full_p, family="Menlo", bold=True, valign="middle")
                draw_text(ctx, f"{db0:+.1f} -> {db14:+.1f} dB", px, py + 13.0, 14, STONE_RGB, alpha=full_p, family="Menlo", valign="middle")

            bx0 = sp_x + (0.0 if not is_vert else 0.0)
            bw = sp_w
            # Ink-and-wash tapering ribbon over 0..14 s
            pts_u = []
            pts_l = []
            for k in range(65):
                u = k / 64.0
                sec = u * 14.0
                amp = math.exp(-3.45 * sec / t60)
                half_h = (2.0 + 17.0 * amp) * (0.85 if is_vert else 1.0)
                pts_u.append((bx0 + u * bw, py - half_h))
                pts_l.append((bx0 + u * bw, py + half_h))
            ctx.new_path()
            for j, (ux, uy) in enumerate(pts_u):
                ctx.move_to(ux, uy) if j == 0 else ctx.line_to(ux, uy)
            for ux, uy in reversed(pts_l):
                ctx.line_to(ux, uy)
            ctx.close_path()
            ctx.set_source_rgba(pcol[0], pcol[1], pcol[2], 0.34 * full_p)
            ctx.fill()

            ctx.set_source_rgba(pcol[0], pcol[1], pcol[2], 0.90 * full_p)
            ctx.set_line_width(2.0)
            ctx.move_to(bx0, py)
            ctx.line_to(bx0 + bw, py)
            ctx.stroke()

        # Playhead caliper across 0 -> 14 s (with initial soft highlight glow on chord strike)
        imp_full = math.exp(-dt_full / 0.45)
        if imp_full > 0.02:
            ctx.set_source_rgba(GOLD_RGB[0], GOLD_RGB[1], GOLD_RGB[2], 0.14 * imp_full * full_p)
            ctx.rectangle(sp_x, sp_y, 26.0 + 30.0 * (1.0 - imp_full), sp_h)
            ctx.fill()
        cx_cal = sp_x + play_u * sp_w
        ctx.set_source_rgba(INK_RGB[0], INK_RGB[1], INK_RGB[2], 0.85 * full_p)
        ctx.set_line_width(2.2)
        ctx.move_to(cx_cal, sp_y)
        ctx.line_to(cx_cal, sp_y + sp_h)
        ctx.stroke()
        draw_text(ctx, f"{play_sec:.1f} s", cx_cal, sp_y + sp_h + 22.0, 16, INK_RGB, alpha=full_p, family="Menlo", bold=True, align="center")

        for tm, ta in [(0, "left"), (8, "center"), (14, "right")]:
            tx_m = sp_x + (tm / 14.0) * sp_w
            draw_text(ctx, f"{tm}s", tx_m, sp_y - 4.0, 14, STONE_RGB, alpha=full_p * 0.8, family="Menlo", align=ta)

        # Right / Bottom: Terhardt Virtual Pitch Salience bars interpolating live with play_u
        draw_text(ctx, "TERHARDT PERCEIVED PITCH SALIENCE", sal_x, sal_y - 26.0, 16, SLATE_RGB, alpha=full_p, family="Menlo", bold=True)
        w_sal = smoothstep(0.5, 11.0, play_sec)
        sal_rows = [
            ("B4 (499.6 Hz · fifth)", 0.55, 1.00, SLATE_RGB),
            ("E3 (167.8 Hz · strike)", 1.00, 0.70, INK_RGB),
            ("G3 (200.3 Hz · hum)", 0.11, 0.56, GOLD_RGB),
            ("A4 (440.6 Hz · clash)", 0.68, 0.02, CLAY_RGB),
        ]
        s_row_h = sal_h / 4.0
        for idx, (s_lbl, v0, v1, scol) in enumerate(sal_rows):
            ry = sal_y + (idx + 0.5) * s_row_h
            val = v0 + (v1 - v0) * w_sal
            draw_text(ctx, s_lbl, sal_x, ry - 14.0, 18 if not is_vert else 16, INK_RGB, alpha=full_p, family="Georgia", bold=True, valign="middle")
            draw_text(ctx, f"{val:.2f}", sal_x + sal_w, ry - 14.0, 17 if not is_vert else 15, scol, alpha=full_p, family="Menlo", bold=True, align="right", valign="middle")

            bar_y = ry + 12.0
            ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], 0.50 * full_p)
            ctx.set_line_width(1.2)
            ctx.move_to(sal_x, bar_y)
            ctx.line_to(sal_x + sal_w, bar_y)
            ctx.stroke()

            # Ghost mark at impact (v0)
            ctx.set_source_rgba(INK_RGB[0], INK_RGB[1], INK_RGB[2], 0.25 * full_p)
            ctx.rectangle(sal_x + v0 * sal_w - 1.5, bar_y - 9.0, 3.0, 18.0)
            ctx.fill()

            ctx.set_source_rgba(scol[0], scol[1], scol[2], 0.90 * full_p)
            ctx.rectangle(sal_x, bar_y - 6.0, max(6.0, val * sal_w), 12.0)
            ctx.fill()

    return surface_to_float_rgb(buf)


def render_scene_4_abbey(t_rel: float, act_dur: float, t_abs: float, aspect: str, t_close: float = 0.55, t_yard: float = 4.25) -> np.ndarray:
    """
    Pedagogical Scene 4 (Nocturnal Westminster Abbey Roof Diffraction & Winter Mist):
      Simple -> Mechanism: 10 m close strike sends short 5 kHz waves slamming into the Abbey stone roof.
      Technical Final: 354.5 m Little Dean's Yard strike shows long 200 Hz waves bending into the courtyard,
                       with -49.5 dB winter loss, +51.0 ms cold-air lag, 6.9 s vs 3.0 s audible tail & Maekawa caveat.
    """
    is_vert = (aspect == "9x16")
    W, H = (1080, 1920) if is_vert else (1920, 1080)

    # Nocturnal slate-indigo sky gradient matching Scene V midnight yard
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    sky = np.empty((H, W, 3), dtype=np.float32)
    for c in range(3):
        sky[:, :, c] = NIGHT_BG_TOP[c] * (1.0 - yy) + NIGHT_BG_BOT[c] * yy
    surf, ctx, buf = make_cairo_surface(W, H, sky)
    m = layout_metrics(W, H, is_vert)
    draw_folio_rules(ctx, W, H, m["pad_x"], m["top_y"], m["bot_y"], dark=True)

    is_yard = (t_rel + 1e-4) >= t_yard
    yard_p = ease_out3(clamp01((t_rel - t_yard + 0.04) / 0.90)) if is_yard else 0.0
    head = "Short 5 kHz waves stop at the Abbey roof." if not is_yard else "Long 200 Hz waves bend into the yard."
    draw_header_text(ctx, W, m["pad_x"], m["top_y"], "MECHANISM IV · ACOUSTIC DIFFRACTION & WINTER MIST", head, t_rel, act_dur, is_vert, dark=True)

    sy, sh, sw, px = m["sy"], m["sh"], m["sw"], m["pad_x"]
    gy = sy + sh * (0.64 if is_vert else 0.72)

    # Ground line
    ctx.set_source_rgba(NIGHT_EDGE[0], NIGHT_EDGE[1], NIGHT_EDGE[2], 0.45)
    ctx.set_line_width(2.0)
    ctx.move_to(px, gy)
    ctx.line_to(px + sw, gy)
    ctx.stroke()

    # Scale elevation positions: Tower left -> Abbey center -> Little Dean's Yard right
    tx = px + sw * 0.11
    tw = 28.0 if not is_vert else 22.0
    th = sh * (0.36 if is_vert else 0.46)
    ty_top = gy - th
    belfry_x, belfry_y = tx, ty_top + th * 0.24

    ax = px + sw * 0.49
    aw = 136.0 if not is_vert else 86.0
    ah = th * 0.56
    ridge_y = gy - ah

    yx = px + sw * 0.88
    yc_w = 110.0 if not is_vert else 74.0

    # Elizabeth Tower silhouette + glowing clock dial
    ctx.set_source_rgba(NIGHT_STONE[0], NIGHT_STONE[1], NIGHT_STONE[2], 0.98)
    ctx.rectangle(tx - tw * 0.5, ty_top + 28.0, tw, th - 28.0)
    ctx.fill()
    ctx.move_to(tx, ty_top - 24.0)
    ctx.line_to(tx - tw * 0.65, ty_top + 28.0)
    ctx.line_to(tx + tw * 0.65, ty_top + 28.0)
    ctx.close_path()
    ctx.fill()
    t_last_abbey = t_yard if is_yard else t_close
    imp_abbey = math.exp(-max(0.0, t_rel - t_last_abbey) / 0.40) if t_rel + 1e-4 >= t_last_abbey else 0.0
    if imp_abbey > 0.02:
        ctx.set_source_rgba(NIGHT_GOLD[0], NIGHT_GOLD[1], NIGHT_GOLD[2], 0.35 * imp_abbey)
        ctx.arc(tx, belfry_y + 24.0, 26.0 + (1.0 - imp_abbey) * 32.0, 0, math.tau)
        ctx.fill()
    ctx.set_source_rgba(NIGHT_GOLD[0], NIGHT_GOLD[1], NIGHT_GOLD[2], 0.92)
    ctx.arc(tx, belfry_y + 24.0, 9.5 + 3.0 * imp_abbey, 0, math.tau)
    ctx.fill()

    # Westminster Abbey nave roof obstacle
    ctx.set_source_rgba(NIGHT_STONE[0], NIGHT_STONE[1], NIGHT_STONE[2], 0.98)
    ctx.move_to(ax - aw * 0.5, gy)
    ctx.line_to(ax - aw * 0.5, ridge_y + 34.0)
    ctx.line_to(ax, ridge_y)
    ctx.line_to(ax + aw * 0.5, ridge_y + 34.0)
    ctx.line_to(ax + aw * 0.5, gy)
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(NIGHT_EDGE[0], NIGHT_EDGE[1], NIGHT_EDGE[2], 0.42)
    ctx.set_line_width(1.6)
    ctx.stroke()

    # Cloister courtyard walls in Little Dean's Yard
    ctx.set_source_rgba(NIGHT_STONE[0], NIGHT_STONE[1], NIGHT_STONE[2], 0.96)
    ctx.rectangle(yx - yc_w * 0.5 - 14.0, gy - 48.0, 14.0, 48.0)
    ctx.rectangle(yx + yc_w * 0.5, gy - 48.0, 14.0, 48.0)
    ctx.fill()

    # Labels above landmarks (clean dark slate spacing)
    draw_text(ctx, "Elizabeth Tower (65 m)", tx, ty_top - 44.0, 16 if not is_vert else 14, NIGHT_GOLD, family="Menlo", bold=True, align="center")
    draw_text(ctx, "Westminster Abbey (165 m)", ax, ridge_y - 26.0, 17 if not is_vert else 14, NIGHT_TEXT, family="Georgia", align="center")
    draw_text(ctx, "Little Dean's Yard (354.5 m)", yx, gy + 28.0, 16 if not is_vert else 14, NIGHT_FROST, alpha=0.5 + 0.5 * yard_p, family="Menlo", bold=True, align="right")

    # Wavefront visualization:
    # Phase A (close strike): dense tight 5 kHz high-freq ripples emitted from belfry, blocked at Abbey roof
    if t_rel + 1e-4 >= t_close:
        dt_c = max(0.0, t_rel - t_close)
        max_r = max(12.0, ax - tx - 14.0)
        for w_i in range(6):
            prog_w = ((dt_c * 0.55) + w_i * 0.16) % 1.0
            rad_w = prog_w * max_r * 1.15
            if rad_w <= ax - tx + 18.0:
                alpha_w = (1.0 - prog_w) * (0.65 if not is_yard else 0.30)
                ctx.set_source_rgba(NIGHT_CLAY[0], NIGHT_CLAY[1], NIGHT_CLAY[2], alpha_w)
                ctx.set_line_width(2.0)
                # Right-facing high-frequency arc clipped before Abbey roof
                ctx.arc(belfry_x, belfry_y, rad_w, -0.45, 0.52)
                ctx.stroke()
        draw_text(ctx, "5 kHz clang (6.8 cm waves) blocked", 0.5 * (tx + ax), belfry_y - 44.0, 16 if not is_vert else 14, NIGHT_CLAY, alpha=0.95, family="Menlo", align="center")

    # Phase B (Abbey diffraction): wide 200 Hz waves bend smoothly over roof ridge into Little Dean's Yard
    if is_yard:
        dt_y = t_rel - t_yard
        # Dashed diffracted path over Abbey crest
        ctx.save()
        ctx.set_dash([8.0, 6.0])
        ctx.set_source_rgba(NIGHT_FROST[0], NIGHT_FROST[1], NIGHT_FROST[2], 0.72 * yard_p)
        ctx.set_line_width(2.0)
        ctx.move_to(belfry_x, belfry_y)
        ctx.line_to(ax, ridge_y - 4.0)
        ctx.line_to(yx, gy - 12.0)
        ctx.stroke()
        ctx.restore()

        # Expanding wide Huygens arcs diffracting from the Abbey roof crest down into the courtyard
        dist_crest_yard = math.hypot(yx - ax, (gy - 10.0) - ridge_y)
        for w_i in range(4):
            prog_y = ((dt_y * 0.32) + w_i * 0.25) % 1.0
            rad_y = prog_y * dist_crest_yard * 1.05
            wa = (1.0 - prog_y) * 0.82 * yard_p
            ctx.set_source_rgba(NIGHT_FROST[0], NIGHT_FROST[1], NIGHT_FROST[2], wa)
            ctx.set_line_width(3.4)
            ctx.arc(ax, ridge_y, rad_y, -0.15, 0.55)
            ctx.stroke()

        # Measured courtyard receiver dot
        ctx.set_source_rgba(NIGHT_FROST[0], NIGHT_FROST[1], NIGHT_FROST[2], yard_p)
        ctx.arc(yx, gy - 12.0, 8.5, 0, math.tau)
        ctx.fill()

        # Clean bottom metric strip (no cards/boxes—just crisp tabular typography on dark sky)
        if is_vert:
            draw_text(ctx, "0 C mist @ 5 kHz: -49.5 dB   (200 Hz: -12.4 dB)", px, gy + 86.0, 17, NIGHT_CLAY, alpha=yard_p, family="Menlo", bold=True)
            draw_text(ctx, "Cold air lag: +51.0 ms   ·   1969 HF drop: 10.87 dB", px, gy + 118.0, 16, NIGHT_TEXT, alpha=yard_p, family="Menlo")
            draw_text(ctx, "Night tail (43 dBA): 6.9 s vs 3.0 s day (60 dBA)", px, gy + 150.0, 16, NIGHT_GOLD, alpha=yard_p, family="Menlo", bold=True)
            draw_text(ctx, "Modelled Maekawa barrier + ISO 9613-1 atmosphere.", px, m["bot_y"] - 18.0, 16, NIGHT_MUTED, alpha=yard_p, family="Georgia", italic=True)
        else:
            y_metrics = gy + 72.0
            draw_text(ctx, "5 kHz winter loss: -49.5 dB", px, y_metrics, 18, NIGHT_CLAY, alpha=yard_p, family="Menlo", bold=True)
            draw_text(ctx, "200 Hz bends at -12.4 dB · 1969 HF drop: 10.87 dB", px, y_metrics + 28.0, 16, NIGHT_MUTED, alpha=yard_p, family="Menlo")

            draw_text(ctx, "Night tail: 6.9 s vs 3.0 s", px + sw * 0.50, y_metrics, 18, NIGHT_GOLD, alpha=yard_p, family="Menlo", bold=True, align="center")
            draw_text(ctx, "Cold air (0 C vs 28 C): +51.0 ms arrival lag", px + sw * 0.50, y_metrics + 28.0, 16, NIGHT_MUTED, alpha=yard_p, family="Menlo", align="center")

            draw_text(ctx, "Bearing 226.0 deg · 354.5 m", px + sw, y_metrics, 18, NIGHT_FROST, alpha=yard_p, family="Menlo", bold=True, align="right")
            draw_text(ctx, "Modelled Maekawa + ISO 9613-1", px + sw, y_metrics + 28.0, 16, NIGHT_MUTED, alpha=yard_p, family="Georgia", italic=True, align="right")

    return surface_to_float_rgb(buf)


def render_scene_5_archival(t_rel: float, act_dur: float, t_abs: float, aspect: str, t_lock: float = 1.20, t_coda: float = 4.90) -> np.ndarray:
    """
    Pedagogical Scene 5 (1890 Wax Cylinder Modal Lock -> Clean Little Dean's Yard Dusk Coda):
      Stage A (0..t_coda): 4 raw 1890 slow cylinder partials glide across log-frequency axis (alpha = 0.9464x)
                           and snap onto 1969/2024 hairlines (-2.5c to -18.9c accuracy).
      Stage B (t_coda..end): Linear-light dissolve onto still.webp dusk painting with quiet cream serif coda.
    """
    is_vert = (aspect == "9x16")
    W, H = (1080, 1920) if is_vert else (1920, 1080)

    # Render archival comb paper frame
    surf, ctx, buf = make_cairo_surface(W, H, get_paper_plate(W, H))
    m = layout_metrics(W, H, is_vert)
    draw_folio_rules(ctx, W, H, m["pad_x"], m["top_y"], m["bot_y"], dark=False)
    draw_header_text(ctx, W, m["pad_x"], m["top_y"], "ARCHIVAL VERIFICATION · 1890 EDISON WAX CYLINDER", "Speed-corrected (0.9464x), 1890 locks to today.", t_rel, t_coda, is_vert)

    sy, sh, sw, px = m["sy"], m["sh"], m["sw"], m["pad_x"]
    glide_p = ease_io3(clamp01((t_rel - t_lock + 0.04) / 1.65)) if t_rel + 1e-4 >= t_lock else 0.0
    imp_lock = math.exp(-max(0.0, t_rel - t_lock) / 0.40) if t_rel + 1e-4 >= t_lock else 0.0

    rows_1890 = [
        ("Hum (G3)", 189.26, 199.98, 200.26, -2.5, GOLD_RGB),
        ("Nominal (E4)", 314.08, 331.87, 335.51, -18.9, CLAY_RGB),
        ("Quint (B4)", 473.90, 500.74, 499.75, 3.4, SLATE_RGB),
        ("Octave (E5)", 641.03, 677.34, 677.60, -0.7, SAGE_RGB),
    ]

    ax0 = px + (210.0 if not is_vert else 0.0)
    ax1 = px + sw
    ay0 = sy + sh * (0.20 if is_vert else 0.14)
    ay1 = sy + sh * (0.78 if is_vert else 0.74)
    row_h = (ay1 - ay0) / 4.0

    def x_for_hz(hz: float) -> float:
        # Shared log-frequency axis spanning 175 Hz .. 720 Hz cleanly
        u = (math.log2(hz) - math.log2(172.0)) / (math.log2(740.0) - math.log2(172.0))
        return ax0 + u * (ax1 - ax0)

    for idx, (lbl, f_raw, f_corr, f_ref, cents, col) in enumerate(rows_1890):
        ry = ay0 + (idx + 0.5) * row_h
        if is_vert:
            draw_text(ctx, lbl, px, ry - 26.0, 19, INK_RGB, family="Georgia", bold=True, valign="middle")
        else:
            draw_text(ctx, lbl, px, ry, 22, INK_RGB, family="Georgia", bold=True, valign="middle")

        # Horizontal rail + 1969/2024 target reference hairline tick
        ctx.set_source_rgba(RULE_RGB[0], RULE_RGB[1], RULE_RGB[2], 0.75)
        ctx.set_line_width(1.5)
        ctx.move_to(ax0, ry)
        ctx.line_to(ax1, ry)
        ctx.stroke()

        xr = x_for_hz(f_ref)
        x_raw = x_for_hz(f_raw)
        x_corr = x_for_hz(f_corr)
        x_cur = x_raw + (x_corr - x_raw) * glide_p

        # Target hairline
        ctx.set_source_rgba(SLATE_RGB[0], SLATE_RGB[1], SLATE_RGB[2], 0.85)
        ctx.set_line_width(2.2)
        ctx.move_to(xr, ry - 24.0)
        ctx.line_to(xr, ry + 24.0)
        ctx.stroke()

        # Vector bridge arrow showing exact +0.95 semitone speed slide
        if glide_p > 0.05:
            ctx.set_source_rgba(CLAY_RGB[0], CLAY_RGB[1], CLAY_RGB[2], 0.55)
            ctx.set_line_width(2.4)
            ctx.move_to(x_raw, ry)
            ctx.line_to(x_cur, ry)
            ctx.stroke()

        # Ghost dot at raw 1890 pitch
        ctx.set_source_rgba(STONE_RGB[0], STONE_RGB[1], STONE_RGB[2], 0.55)
        ctx.arc(x_raw, ry, 6.0, 0, math.tau)
        ctx.fill()

        # Live sliding dot
        if imp_lock > 0.02:
            ctx.set_source_rgba(col[0], col[1], col[2], 0.34 * imp_lock)
            ctx.arc(x_cur, ry, 18.0 + (1.0 - imp_lock) * 26.0, 0, math.tau)
            ctx.fill()
        ctx.set_source_rgba(col[0], col[1], col[2], 0.96)
        ctx.arc(x_cur, ry, 9.5, 0, math.tau)
        ctx.fill()

        cur_hz = f_raw + (f_corr - f_raw) * glide_p
        tag_txt = f"{cur_hz:.1f} Hz" if glide_p < 0.85 else f"{f_corr:.2f} Hz ({cents:+.1f}c)"
        draw_text(
            ctx,
            tag_txt,
            min(ax1 - 6.0, x_cur + 16.0),
            ry - 22.0 if idx % 2 == 0 else ry + 26.0,
            16 if not is_vert else 14,
            col,
            family="Menlo",
            bold=True,
            align="left" if x_cur < (ax0 + 0.65 * (ax1 - ax0)) else "right",
        )

    draw_text(
        ctx,
        "1890 speed offset x0.9464 (-0.95 st) · Blind log-correlation agrees to 0.0006 · 1953 ENF 50.132 Hz",
        px,
        m["bot_y"] - 20.0,
        17 if not is_vert else 14,
        STONE_RGB,
        family="Menlo",
    )

    comb_arr = surface_to_float_rgb(buf)

    # Coda dissolve into clean full-bleed Little Dean's Yard dusk painting
    if t_rel >= t_coda:
        coda_fade = ease_io3(clamp01((t_rel - t_coda) / 1.05))
        dusk_base = get_dusk_plate_arr(W, H, mode="coda")
        s_coda, ctx_c, buf_c = make_cairo_surface(W, H, dusk_base)
        txt_a = smoothstep(t_coda + 0.35, t_coda + 1.05, t_rel)
        cy_c = H - (260.0 if is_vert else 185.0)
        draw_text(ctx_c, "The Chords Inside Big Ben", px, cy_c - 64.0, 52 if is_vert else 66, NIGHT_TEXT, alpha=txt_a, family="Georgia", bold=True)
        draw_text(ctx_c, "zidingz.com/the-chords-inside-big-ben", px, cy_c, 26 if is_vert else 28, NIGHT_GOLD, alpha=txt_a, family="Menlo", bold=True)
        draw_text(ctx_c, "zidingz.com/big-ben-from-little-deans-yard", px, cy_c + 40.0, 21 if is_vert else 22, NIGHT_MUTED, alpha=txt_a, family="Menlo")
        coda_arr = surface_to_float_rgb(buf_c)
        # Linear-light cross-dissolve
        return np.clip(((comb_arr ** 2.2) * (1.0 - coda_fade) + (coda_arr ** 2.2) * coda_fade), 0.0, 1.0) ** (1.0 / 2.2)

    return comb_arr


# ==============================================================================
# Four Vectorized Physical Transitions (Ported from zidingz-cinema transitions)
# ==============================================================================


def transition_line(arr_a: np.ndarray, arr_b: np.ndarray, p: float, t_abs: float) -> np.ndarray:
    """
    Port of zidingz-cinema/transitions/line.js:
    Vertical column compression in linear light into a vibrating 3-harmonic bronze string,
    glowing at midpoint, and opening vertically into incoming scene B.
    """
    p = clamp01(p)
    if p <= 0.001:
        return arr_a
    if p >= 0.999:
        return arr_b

    H, W, _ = arr_a.shape
    lin_a = np.power(np.clip(arr_a, 0.0, 1.0), 2.2)
    lin_b = np.power(np.clip(arr_b, 0.0, 1.0), 2.2)

    # 16-row energy-conserving vertical column average
    avg_a = np.mean(lin_a[:: max(1, H // 16)], axis=0)  # (W, 3)
    avg_b = np.mean(lin_b[:: max(1, H // 16)], axis=0)

    x_norm = np.linspace(0.0, 1.0, W, dtype=np.float32)
    y_norm = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    center_bias = 1.0 - np.abs(x_norm - 0.5) * 2.0

    # Three-harmonic bronze string curve
    mid_env = math.exp(-((p - 0.50) / 0.16) ** 2)
    ph = (p - 0.34) * 34.0
    string_wave = (
        np.sin(math.pi * x_norm) * math.sin(ph)
        + 0.42 * np.sin(2.0 * math.pi * x_norm) * math.sin(ph * 1.49 + 0.7)
        + 0.20 * np.sin(3.0 * math.pi * x_norm) * math.sin(ph * 2.63 + 1.4)
    ) * 0.032 * mid_env * np.sin(math.pi * x_norm)
    axis_y = (0.5 + string_wave)[None, :]  # (1, W)

    c_prog = np.clip((p + center_bias * 0.10) / 0.48, 0.0, 1.0)
    c_prog = c_prog * c_prog * (3.0 - 2.0 * c_prog)
    e_prog = np.clip((p - 0.52 + center_bias * 0.10) / 0.48, 0.0, 1.0)
    e_prog = e_prog * e_prog * (3.0 - 2.0 * e_prog)

    out_lin = np.full((H, W, 3), np.array([0.008, 0.009, 0.014], dtype=np.float32), dtype=np.float32)

    if p < 0.5:
        half_h = (1.0 - 0.996 * c_prog) * 0.5
        dy = (y_norm - axis_y) / np.maximum(half_h[None, :], 0.0014)
        mask = np.abs(dy) <= 1.0
        map_x = np.broadcast_to((x_norm * (W - 1))[None, :], (H, W)).astype(np.float32)
        map_y = np.clip((dy * 0.5 + 0.5) * (H - 1), 0.0, H - 1).astype(np.float32)
        sampled = cv2.remap(lin_a, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blend_avg = smoothstep_arr(0.55, 0.96, c_prog)[None, :, None]
        energy = (1.0 + 1.2 * smoothstep_arr(0.65, 1.0, c_prog))[None, :, None]
        sheet = (sampled * (1.0 - blend_avg) + avg_a[None, :, :] * blend_avg) * energy
        out_lin[mask] = sheet[mask]
    else:
        half_h = (0.004 + 0.996 * e_prog) * 0.5
        dy = (y_norm - axis_y) / np.maximum(half_h[None, :], 0.0014)
        mask = np.abs(dy) <= 1.0
        map_x = np.broadcast_to((x_norm * (W - 1))[None, :], (H, W)).astype(np.float32)
        map_y = np.clip((dy * 0.5 + 0.5) * (H - 1), 0.0, H - 1).astype(np.float32)
        sampled = cv2.remap(lin_b, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blend_avg = smoothstep_arr(0.04, 0.45, e_prog)[None, :, None]
        energy = (2.2 - 1.2 * smoothstep_arr(0.0, 0.35, e_prog))[None, :, None]
        sheet = (avg_b[None, :, :] * (1.0 - blend_avg) + sampled * blend_avg) * energy
        out_lin[mask] = sheet[mask]

    # Glowing bronze acoustic filament halo near p = 0.5
    dist_axis = np.abs(y_norm - axis_y)
    glow_strength = smoothstep(0.18, 0.46, p) * (1.0 - smoothstep(0.54, 0.84, p))
    if glow_strength > 0.001:
        core = np.exp(-dist_axis * 280.0)[:, :, None] * 2.1
        halo = (np.exp(-dist_axis * 34.0) * 0.55 + np.exp(-dist_axis * 9.0) * 0.16)[:, :, None]
        str_col = (avg_a * (1.0 - smoothstep(0.42, 0.58, p)) + avg_b * smoothstep(0.42, 0.58, p))[None, :, :] * 0.58 + np.array([0.95, 0.52, 0.24], dtype=np.float32) * 0.42
        out_lin += (np.array([1.0, 0.86, 0.68], dtype=np.float32) * core + str_col * halo) * glow_strength

    # Clean endpoint preserve
    srgb = np.power(np.clip(out_lin / (1.0 + np.maximum(0.0, out_lin - 0.85) * 0.4), 0.0, 1.0), 1.0 / 2.2)
    end_a = 1.0 - smoothstep(0.0, 0.08, p)
    end_b = smoothstep(0.92, 1.0, p)
    if end_a > 0:
        srgb = srgb * (1.0 - end_a) + arr_a * end_a
    if end_b > 0:
        srgb = srgb * (1.0 - end_b) + arr_b * end_b
    return np.clip(srgb, 0.0, 1.0)


def smoothstep_arr(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def transition_fold(arr_a: np.ndarray, arr_b: np.ndarray, p: float, t_abs: float) -> np.ndarray:
    """
    Port of zidingz-cinema/transitions/fold.js:
    Center-spine origami gable + perspective broadsheet wing fold with cast shadow and warm rag backface.
    """
    p = clamp01(p)
    if p <= 0.001:
        return arr_a
    if p >= 0.999:
        return arr_b

    H, W, _ = arr_a.shape
    f_fold = ease_io3(clamp01(p / 0.82))
    glide = ease_io3(clamp01((p - 0.55) / 0.45))
    sheet_alpha = 1.0 - smoothstep(0.78, 0.98, p)

    # Cast shadow across incoming Scene B
    sh_alpha = 0.28 * smoothstep(0.0, 0.28, p) * (1.0 - smoothstep(0.75, 0.98, p))
    out = arr_b.copy()
    if sh_alpha > 0.002:
        sh_w = max(2, int(W * 0.5 * (1.0 - 0.85 * f_fold)))
        x_mid = W // 2 + int(0.025 * W)
        xL = max(0, x_mid - sh_w)
        xR = min(W, x_mid + sh_w)
        out[:, xL:xR] *= (1.0 - sh_alpha)

    if sheet_alpha <= 0.002:
        return out

    # Perspective warp left & right sheet halves folding inward toward spine x = W / 2
    half_w = W * 0.5
    ang = f_fold * 1.48  # ~85 deg
    cos_a = max(0.04, math.cos(ang))
    sin_a = math.sin(ang)
    scale_g = 1.0 - 0.48 * glide
    y_shift = -0.10 * H * glide
    persp = 0.20 * sin_a * H * scale_g

    cx = W * 0.5
    h_span = H * scale_g
    y_top = (H - h_span) * 0.5 + y_shift
    y_bot = y_top + h_span

    # Left half: [0, cx] -> spine at cx, outer edge swings forward/inward
    outer_l = cx - half_w * cos_a * scale_g
    src_l = np.float32([[0, 0], [half_w, 0], [half_w, H], [0, H]])
    dst_l = np.float32([[outer_l, y_top - persp], [cx, y_top], [cx, y_bot], [outer_l, y_bot + persp]])
    M_l = cv2.getPerspectiveTransform(src_l, dst_l)
    warp_l = cv2.warpPerspective(arr_a, M_l, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_l = cv2.warpPerspective(np.ones((H, int(half_w)), dtype=np.float32), M_l, (W, H), flags=cv2.INTER_LINEAR)

    # Right half: [cx, W]
    outer_r = cx + half_w * cos_a * scale_g
    src_r = np.float32([[half_w, 0], [W, 0], [W, H], [half_w, H]])
    dst_r = np.float32([[cx, y_top], [outer_r, y_top - persp * 0.85], [outer_r, y_bot + persp * 0.85], [cx, y_bot]])
    M_r = cv2.getPerspectiveTransform(src_r, dst_r)
    warp_r = cv2.warpPerspective(arr_a, M_r, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # Build right-half coverage cleanly in [0..W] coordinates
    mask_src_r = np.zeros((H, W), dtype=np.float32)
    mask_src_r[:, int(half_w):] = 1.0
    mask_r = cv2.warpPerspective(mask_src_r, M_r, (W, H), flags=cv2.INTER_LINEAR)

    # Apply source mask cleanly for left half too
    mask_src_l = np.zeros((H, W), dtype=np.float32)
    mask_src_l[:, : int(half_w)] = 1.0
    mask_l = cv2.warpPerspective(mask_src_l, M_l, (W, H), flags=cv2.INTER_LINEAR)

    # Directional lighting + valley crease highlight/AO
    shade_l = 0.62 + 0.38 * cos_a
    shade_r = 0.82 + 0.18 * cos_a
    # Soft tint toward warm paper grain near settle
    paper_t = np.array([0.94, 0.92, 0.87], dtype=np.float32)
    tone_fade = smoothstep(0.62, 0.94, p) * 0.45

    ml = (mask_l * sheet_alpha)[:, :, None]
    mr = (mask_r * sheet_alpha)[:, :, None]
    fl = warp_l * shade_l * (1.0 - tone_fade) + paper_t * tone_fade
    fr = warp_r * shade_r * (1.0 - tone_fade) + paper_t * tone_fade

    out = out * (1.0 - ml) + fl * ml
    out = out * (1.0 - mr) + fr * mr
    return np.clip(out, 0.0, 1.0)


def transition_wash(arr_a: np.ndarray, arr_b: np.ndarray, p: float, t_abs: float) -> np.ndarray:
    """
    Port of zidingz-cinema/transitions/wash.js:
    Diagonal refractive water crest sweeping across glass with R/G/B dispersion,
    red-first cool Thames absorption, specular crest highlights, and clean swap.
    """
    p = clamp01(p)
    if p <= 0.001:
        return arr_a
    if p >= 0.999:
        return arr_b

    H, W, _ = arr_a.shape
    sh, sw = H // 3, W // 3
    yy = np.linspace(0.0, 1.0, sh, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, sw, dtype=np.float32)[None, :]

    diag = yy * 0.78 + xx * 0.36
    front = -0.25 + 1.75 * smoothstep(0.0, 0.86, p)
    wobble = 0.045 * np.sin(xx * 9.0 + p * 3.1) + 0.018 * np.cos(xx * 18.0 - p * 8.0)
    dist = diag + wobble - front

    crest = np.exp(-((dist + 0.075) / 0.105) ** 2) * 1.15
    body = smoothstep_arr(0.04, -0.08, dist) * np.exp(np.maximum(dist + 0.08, -2.0) * 2.6)
    drain_fade = 1.0 - smoothstep(0.68, 0.98, p)
    thick_s = np.clip((crest + body * 0.60) * drain_fade, 0.0, 1.5)
    swap_s = smoothstep_arr(0.01, -0.10, dist)

    thick = cv2.resize(thick_s, (W, H), interpolation=cv2.INTER_LINEAR)
    swap_m = cv2.resize(swap_s, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, None]

    # Spatial gradient for surface-normal refraction
    gx = cv2.Sobel(thick, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gy = cv2.Sobel(thick, cv2.CV_32F, 0, 1, ksize=3) * 0.25

    grid_x = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W))
    grid_y = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    scale_d = 120.0 + 48.0 * thick

    def _disp_sample(src: np.ndarray, ch_scale: float, ch_idx: int) -> np.ndarray:
        mx = np.clip(grid_x - gx * scale_d * ch_scale, 0.0, W - 1.0).astype(np.float32)
        my = np.clip(grid_y - gy * scale_d * ch_scale, 0.0, H - 1.0).astype(np.float32)
        return cv2.remap(src[:, :, ch_idx], mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    a_disp = np.stack([_disp_sample(arr_a, 1.14, 0), _disp_sample(arr_a, 1.00, 1), _disp_sample(arr_a, 0.86, 2)], axis=-1)
    b_disp = np.stack([_disp_sample(arr_b, 1.14, 0), _disp_sample(arr_b, 1.00, 1), _disp_sample(arr_b, 0.86, 2)], axis=-1)

    col = a_disp * (1.0 - swap_m) + b_disp * swap_m
    # Red-first Thames water absorption under wave crest
    absorb = np.exp(-np.array([0.42, 0.16, 0.08], dtype=np.float32)[None, None, :] * thick[:, :, None] * 1.15)
    col *= 1.0 - np.clip(thick[:, :, None] * 0.72, 0.0, 0.82) * (1.0 - absorb)
    # Specular wave lip highlight
    spec = np.clip(np.hypot(gx, gy) * 2.4, 0.0, 1.0)[:, :, None] * (thick[:, :, None] * 0.42)
    col += np.array([0.88, 0.94, 1.0], dtype=np.float32) * spec

    snap = smoothstep(0.92, 1.0, p)
    if snap > 0:
        col = col * (1.0 - snap) + arr_b * snap
    return np.clip(col, 0.0, 1.0)


def _disc_bokeh(img_lin: np.ndarray, radius_px: int) -> np.ndarray:
    """Fast highlight-weighted circular disc bokeh at 0.5x resolution (matches daybreak.js)."""
    if radius_px <= 1:
        return img_lin
    H, W, _ = img_lin.shape
    sh, sw = H // 2, W // 2
    small = cv2.resize(img_lin, (sw, sh), interpolation=cv2.INTER_AREA)
    r_s = max(1, radius_px // 2)
    if r_s not in _BOKEH_KERNEL:
        k_sz = r_s * 2 + 1
        yy, xx = np.mgrid[-r_s : r_s + 1, -r_s : r_s + 1]
        disk = ((xx * xx + yy * yy) <= (r_s * r_s + 0.5)).astype(np.float32)
        disk /= max(1.0, float(np.sum(disk)))
        _BOKEH_KERNEL[r_s] = disk
    ker = _BOKEH_KERNEL[r_s]
    lum = np.max(small, axis=-1, keepdims=True)
    w = 1.0 + 7.5 * (lum ** 3)
    w_blur = np.maximum(cv2.filter2D(w[:, :, 0], -1, ker)[:, :, None], 1e-4)
    c_blur = cv2.filter2D(small * w, -1, ker) / w_blur
    return cv2.resize(c_blur, (W, H), interpolation=cv2.INTER_LINEAR)


def transition_daybreak(arr_a: np.ndarray, arr_b: np.ndarray, p: float, t_abs: float) -> np.ndarray:
    """
    Port of zidingz-cinema/transitions/daybreak.js:
    Linear HDR highlight-weighted circular bokeh rack focus from nocturnal Scene A onto warm Scene B.
    """
    p = clamp01(p)
    if p <= 0.001:
        return arr_a
    if p >= 0.999:
        return arr_b

    lin_a = np.power(np.clip(arr_a, 0.0, 1.0), 2.2)
    lin_b = np.power(np.clip(arr_b, 0.0, 1.0), 2.2)

    rad_a = int(round(min(1.0, p / 0.56) * 44.0))
    rad_b = int(round(min(1.0, (1.0 - p) / 0.56) * 44.0))
    c_a = _disc_bokeh(lin_a, rad_a)
    c_b = _disc_bokeh(lin_b, rad_b)

    w_a = 1.0 - smoothstep(0.24, 0.66, p)
    w_b = smoothstep(0.34, 0.80, p)
    sum_w = max(w_a + w_b, 1.0)
    lin = (c_a * w_a + c_b * w_b) / (1.0 + 0.45 * (sum_w - 1.0))

    # Warm bronze dawn bloom around midpoint
    dawn_glow = (math.sin(p * math.pi) ** 2.2) * 0.14
    lin += np.array([0.22, 0.11, 0.04], dtype=np.float32)[None, None, :] * dawn_glow
    lin = lin / (1.0 + np.maximum(0.0, lin - 0.85) * 0.45)

    col = np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2)
    snap = smoothstep(0.92, 1.0, p)
    if snap > 0:
        col = col * (1.0 - snap) + arr_b * snap
    return np.clip(col, 0.0, 1.0)


TRANSITIONS: dict[str, Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]] = {
    "line": transition_line,
    "fold": transition_fold,
    "wash": transition_wash,
    "daybreak": transition_daybreak,
}


# ==============================================================================
# Stateless Blog Cut Compositor
# ==============================================================================


def _render_blog_scene(scene_id: int, t: float, aspect: str) -> np.ndarray:
    t0, t1 = BLOG_ACT_BOUNDS[scene_id]
    dur = t1 - t0
    t_rel = t - t0
    if scene_id == 1:
        return render_scene_1_voices(t_rel, dur, t, aspect, t_pure=0.70, t_bell=3.90)
    if scene_id == 2:
        # Transition line ends at 12.65 s; pure 335.44 Hz onset is at 13.10 s (t_rel=1.10), doublet at 16.10 s (t_rel=4.10)
        return render_scene_2_doublet(t_rel, dur, t, aspect, t_single=1.10, t_split=4.10)
    if scene_id == 3:
        # Scene starts at 24.00 s; solo 883 at 25.20 (1.20), hum+B4 at 27.80 (3.80), full 14s at 30.50 (6.50)
        return render_scene_3_decay(t_rel, dur, t, aspect, t_fast=1.20, t_slow=3.80, t_full=6.50)
    if scene_id == 4:
        # Scene starts at 36.50 s; close strike 37.70 (1.20), Abbey yard 41.40 (4.90)
        return render_scene_4_abbey(t_rel, dur, t, aspect, t_close=1.20, t_yard=4.90)
    # Scene 5 starts at 47.50 s; lock at 48.70 (1.20), coda at 52.35 (4.85)
    return render_scene_5_archival(t_rel, dur, t, aspect, t_lock=1.20, t_coda=4.85)


def render_frame(t: float, events: list[dict], aspect: str) -> Image.Image:
    """Stateless pure time renderer for the 56.0 s embedded broadsheet-cinema video."""
    # Check active physical transitions first
    for s_a, s_b, tr_t0, tr_t1, kind in BLOG_TRANSITIONS:
        if tr_t0 <= t <= tr_t1:
            p = (t - tr_t0) / (tr_t1 - tr_t0)
            arr_a = _render_blog_scene(s_a, tr_t0, aspect)
            arr_b = _render_blog_scene(s_b, tr_t1, aspect)
            out = TRANSITIONS[kind](arr_a, arr_b, p, t)
            return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")

    # Otherwise pure interior scene
    scene_id = 5
    for sid, (a0, a1) in BLOG_ACT_BOUNDS.items():
        if t < a1:
            scene_id = sid
            break
    arr = _render_blog_scene(scene_id, t, aspect)
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def render_cut(aspect: str, wav_path: Path, events: list[dict], out_mp4: Path, out_poster: Path) -> None:
    w, h = (1080, 1920) if aspect == "9x16" else (1920, 1080)
    total_frames = int(round(DURATION_S * FPS))
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
        "-i", str(wav_path),
        "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    poster_saved = False
    for f_idx in range(total_frames):
        t = f_idx / FPS
        im = render_frame(t, events, aspect)
        proc.stdin.write(im.tobytes())
        # Save clean hero poster on settled 5-voice modal stem frame (t = 9.40 s)
        if not poster_saved and t >= 9.40:
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
