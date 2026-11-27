#!/usr/bin/env python3
"""Pure-Python 35mm cel-and-gouache study for "The chords inside Big Ben":
Standing in Little Dean's Yard after rain at frosty blue-hour dusk, looking
northeast over wet flagstones and Westminster Abbey's buttressed nave roof
toward Elizabeth Tower's amber belfry louvers and honey-lit clock dial.

Zero image models, zero external textures, zero font dependencies.
Usage: python3 paint_scene.py --out still.webp --crops-dir crops/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import cairo
import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Color, Compositing & Isotropic Gouache Primitives
# ---------------------------------------------------------------------------

def hex_rgb(code: str) -> np.ndarray:
    code = code.lstrip("#")
    return (
        np.array(
            [int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16)],
            dtype=np.float32,
        )
        / 255.0
    )


def smoothstep(
    edge0: np.ndarray | float, edge1: np.ndarray | float, x: np.ndarray | float
) -> np.ndarray:
    e0 = np.asarray(edge0, dtype=np.float32)
    e1 = np.asarray(edge1, dtype=np.float32)
    xv = np.asarray(x, dtype=np.float32)
    denom = np.maximum(e1 - e0, 1e-8)
    t = np.clip((xv - e0) / denom, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def alpha_composite(
    dst_rgb: np.ndarray, src_rgb: np.ndarray, src_a: np.ndarray
) -> np.ndarray:
    a = np.clip(src_a, 0.0, 1.0)[..., None]
    return dst_rgb * (1.0 - a) + src_rgb * a


def screen_blend(
    dst_rgb: np.ndarray, add_rgb: np.ndarray, strength: float = 1.0
) -> np.ndarray:
    b = np.clip(add_rgb * strength, 0.0, 1.0)
    return 1.0 - (1.0 - dst_rgb) * (1.0 - b)


def fbm_gouache(
    h: int,
    w: int,
    octaves: Sequence[tuple[float, float, float]] = (
        (72.0, 68.0, 0.45),
        (26.0, 29.0, 0.32),
        (10.5, 9.5, 0.16),
        (4.5, 4.8, 0.07),
    ),
    seed: int = 101,
) -> np.ndarray:
    """Balanced square-scale gouache mottling (zero stretched striping on fills)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    w_sum = 0.0
    for sx, sy, weight in octaves:
        gh = max(4, int(round(h / float(sy))))
        gw = max(4, int(round(w / float(sx))))
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += float(weight) * up
        w_sum += float(weight)
    acc /= max(w_sum, 1e-6)
    std = float(np.std(acc))
    if std > 1e-6:
        acc = (acc - float(np.mean(acc))) / std
    return acc.astype(np.float32)


def make_paper_tooth(h: int, w: int, seed: int = 707) -> np.ndarray:
    fine = fbm_gouache(h, w, octaves=((3.6, 3.8, 0.55), (1.9, 2.0, 0.45)), seed=seed)
    med = fbm_gouache(h, w, octaves=((11.0, 10.5, 1.0),), seed=seed + 19)
    return np.clip(0.72 * fine + 0.28 * med, -2.5, 2.5).astype(np.float32)


def paint_isotropic_gouache_dabs(
    rgb: np.ndarray,
    mask: np.ndarray,
    paper_tooth: np.ndarray,
    dab_strength: float = 0.044,
    warm_bounce_in_shadow: float = 0.024,
    num_dabs: int = 92,
    tooth_weight: float = 0.015,
    seed: int = 123,
) -> np.ndarray:
    h, w, _ = rgb.shape
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if float(np.max(m)) < 1e-4:
        return rgb.copy()

    mottle = fbm_gouache(h, w, seed=seed)
    sh, sw = max(64, h // 4), max(64, w // 4)
    rng = np.random.default_rng(seed + 41)
    dab_val = np.zeros((sh, sw), dtype=np.float32)
    dab_tmp = np.zeros((sh, sw), dtype=np.float32)

    for _ in range(num_dabs):
        ang = float(rng.uniform(0.0, math.pi))
        hx = float(rng.uniform(13.0, 40.0))
        hy = float(rng.uniform(5.0, 15.0))
        cx = float(rng.uniform(0.0, sw))
        cy = float(rng.uniform(0.0, sh))
        ca, sa = math.cos(ang), math.sin(ang)
        j0, j1 = float(rng.uniform(0.86, 1.12)), float(rng.uniform(0.86, 1.12))
        corners = np.array(
            [
                [cx - hx * ca + hy * j0 * sa, cy - hx * sa - hy * j0 * ca],
                [cx + hx * ca + hy * j1 * sa, cy + hx * sa - hy * j1 * ca],
                [cx + hx * ca - hy * j0 * sa, cy + hx * sa + hy * j0 * ca],
                [cx - hx * ca - hy * j1 * sa, cy - hx * sa + hy * j1 * ca],
            ],
            dtype=np.float32,
        )
        sig = float(rng.uniform(1.3, 2.8))
        pad = int(math.ceil(sig * 3.0)) + 2
        x0 = max(0, int(math.floor(np.min(corners[:, 0]))) - pad)
        y0 = max(0, int(math.floor(np.min(corners[:, 1]))) - pad)
        x1 = min(sw, int(math.ceil(np.max(corners[:, 0]))) + pad)
        y1 = min(sh, int(math.ceil(np.max(corners[:, 1]))) + pad)
        if x1 <= x0 or y1 <= y0:
            continue

        roi_pts = np.round(corners - np.array([[x0, y0]], dtype=np.float32)).astype(np.int32)
        dab = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        cv2.fillConvexPoly(dab, roi_pts, 1.0, cv2.LINE_AA)
        a = cv2.GaussianBlur(dab, (0, 0), sig) * float(rng.uniform(0.18, 0.34))
        dab_val[y0:y1, x0:x1] = dab_val[y0:y1, x0:x1] * (1.0 - a) + float(rng.normal(0.0, 0.42)) * a
        dab_tmp[y0:y1, x0:x1] = dab_tmp[y0:y1, x0:x1] * (1.0 - a) + float(rng.normal(0.0, 0.52)) * a

    dv = cv2.resize(dab_val, (w, h), interpolation=cv2.INTER_LINEAR)
    dt = cv2.resize(dab_tmp, (w, h), interpolation=cv2.INTER_LINEAR)

    out = rgb.copy()
    delta = dab_strength * (0.52 * mottle + 0.48 * dv) + tooth_weight * paper_tooth
    wshift = (dab_strength * 0.36) * (0.48 * mottle + 0.52 * dt)
    out[:, :, 0] += m * (delta + wshift)
    out[:, :, 1] += m * (delta + 0.30 * wshift)
    out[:, :, 2] += m * (delta - wshift)

    lum = 0.2126 * out[:, :, 0] + 0.7152 * out[:, :, 1] + 0.0722 * out[:, :, 2]
    sh_w = np.clip((0.28 - lum) / 0.28, 0.0, 1.0) * m * np.clip(0.5 + 0.5 * dt, 0.0, 1.0)
    out[:, :, 0] += sh_w * warm_bounce_in_shadow * 1.18
    out[:, :, 1] += sh_w * warm_bounce_in_shadow * 0.72
    out[:, :, 2] += sh_w * warm_bounce_in_shadow * 0.34
    return np.clip(out, 0.0, 1.0)


def voronoi_mortar_field(
    uv_h: int,
    uv_w: int,
    span_u: float,
    span_v: float,
    cell_size_m: float = 0.42,
    jitter: float = 0.32,
    warp_amp_m: float = 0.025,
    seed: int = 211,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nx = max(3, int(math.ceil(span_u / cell_size_m)) + 3)
    ny = max(3, int(math.ceil(span_v / cell_size_m)) + 3)

    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    sx = (gx - 1.0 + 0.5 + rng.uniform(-jitter, jitter, size=(ny, nx)).astype(np.float32)) * cell_size_m
    sy = (gy - 1.0 + 0.5 + rng.uniform(-jitter, jitter, size=(ny, nx)).astype(np.float32)) * cell_size_m
    tone = rng.uniform(-1.0, 1.0, size=(ny, nx)).astype(np.float32)

    vv, uu = np.mgrid[0:uv_h, 0:uv_w].astype(np.float32)
    um = (uu / max(uv_w - 1, 1)) * span_u + warp_amp_m * fbm_gouache(uv_h, uv_w, octaves=((36, 36, 1.0),), seed=seed + 3)
    vm = (vv / max(uv_h - 1, 1)) * span_v + warp_amp_m * fbm_gouache(uv_h, uv_w, octaves=((36, 36, 1.0),), seed=seed + 7)

    ix = np.clip(np.floor(um / cell_size_m).astype(np.int32) + 1, 1, nx - 2)
    iy = np.clip(np.floor(vm / cell_size_m).astype(np.int32) + 1, 1, ny - 2)

    d1 = np.full((uv_h, uv_w), 1e9, dtype=np.float32)
    d2 = np.full((uv_h, uv_w), 1e9, dtype=np.float32)
    best_tone = np.zeros((uv_h, uv_w), dtype=np.float32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx = sx[iy + dy, ix + dx]
            cy = sy[iy + dy, ix + dx]
            ct = tone[iy + dy, ix + dx]
            dist = np.sqrt((um - cx) ** 2 + (vm - cy) ** 2)
            closer = dist < d1
            d2 = np.where(closer, d1, np.minimum(d2, dist))
            best_tone = np.where(closer, ct, best_tone)
            d1 = np.where(closer, dist, d1)

    return (d2 - d1).astype(np.float32), best_tone


# ---------------------------------------------------------------------------
# 2. Metric Camera3D & Exact Plane-Basis UV Homography
# ---------------------------------------------------------------------------

class Camera3D:
    def __init__(
        self,
        eye: Sequence[float] | np.ndarray,
        target: Sequence[float] | np.ndarray,
        up: Sequence[float] | np.ndarray,
        fov_deg: float,
        width: int = 2048,
        height: int = 1152,
    ) -> None:
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)

        fwd = self.target - self.eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, self.up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        down /= np.linalg.norm(down)

        self.fwd = fwd
        self.right = right
        self.down = down
        self.R = np.vstack([right, down, fwd])

        self.f = (self.width * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
        self.cx = self.width * 0.5
        self.cy = self.height * 0.5
        self.K = np.array(
            [[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        yy, xx = np.mgrid[0 : self.height, 0 : self.width].astype(np.float32)
        dx_cam = (xx - self.cx) / self.f
        dy_cam = (yy - self.cy) / self.f
        self.ray_len_scale = np.sqrt(dx_cam * dx_cam + dy_cam * dy_cam + 1.0).astype(np.float32)
        rt = self.R.T.astype(np.float32)
        self.ray_world_x = rt[0, 0] * dx_cam + rt[0, 1] * dy_cam + rt[0, 2]
        self.ray_world_y = rt[1, 0] * dx_cam + rt[1, 1] * dy_cam + rt[1, 2]
        self.ray_world_z = rt[2, 0] * dx_cam + rt[2, 1] * dy_cam + rt[2, 2]

        vp_y = self.project_direction(np.array([0.0, 1.0, 0.0]))
        self.horizon_y = float(vp_y[1])

    def project_points(self, pts_world: np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        pts = np.asarray(pts_world, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        rel = pts - self.eye[None, :]
        pc = rel @ self.R.T
        z_cam = pc[:, 2]
        safe_z = np.where(np.abs(z_cam) < 1e-6, 1e-6, z_cam)
        u = self.f * (pc[:, 0] / safe_z) + self.cx
        v = self.f * (pc[:, 1] / safe_z) + self.cy
        if single:
            return float(u[0]), float(v[0]), float(z_cam[0])
        return u, v, z_cam

    def project_direction(self, d_world: np.ndarray) -> tuple[float, float]:
        pc = self.R @ np.asarray(d_world, dtype=np.float64)
        return (
            float(self.f * (pc[0] / max(pc[2], 1e-6)) + self.cx),
            float(self.f * (pc[1] / max(pc[2], 1e-6)) + self.cy),
        )

    def plane_homography(
        self, p0: np.ndarray, e1: np.ndarray, e2: np.ndarray, span_u: float, span_v: float, uv_w: int, uv_h: int
    ) -> np.ndarray:
        e1_u = np.asarray(e1, dtype=np.float64)
        e2_u = np.asarray(e2, dtype=np.float64)
        e1_u /= np.linalg.norm(e1_u)
        e2_u /= np.linalg.norm(e2_u)
        basis = np.column_stack([self.R @ e1_u, self.R @ e2_u, self.R @ (p0 - self.eye)])
        h_uv = np.array(
            [[span_u / float(uv_w), 0.0, 0.0], [0.0, span_v / float(uv_h), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return self.K @ basis @ h_uv

    def ray_plane_hit(self, p0: np.ndarray, normal: np.ndarray):
        n = np.asarray(normal, dtype=np.float64)
        n /= np.linalg.norm(n)
        num = float(np.dot(p0 - self.eye, n))
        den = (
            self.ray_world_x * float(n[0])
            + self.ray_world_y * float(n[1])
            + self.ray_world_z * float(n[2])
        )
        safe_den = np.where(np.abs(den) < 1e-6, 1e-6, den)
        z_cam = (num / safe_den).astype(np.float32)
        dist = z_cam * self.ray_len_scale
        hx = float(self.eye[0]) + z_cam * self.ray_world_x
        hy = float(self.eye[1]) + z_cam * self.ray_world_y
        hz = float(self.eye[2]) + z_cam * self.ray_world_z
        return z_cam, dist, hx, hy, hz

    def warp_plane_panel(
        self,
        panel_rgb: np.ndarray,
        panel_alpha: np.ndarray,
        p0: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        span_u: float,
        span_v: float,
        fog_color: np.ndarray,
        fog_start: float = 6.0,
        fog_density: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        uv_h, uv_w = panel_alpha.shape
        h_mat = self.plane_homography(p0, e1, e2, span_u, span_v, uv_w, uv_h)

        premul = np.empty((uv_h, uv_w, 4), dtype=np.float32)
        a_clamped = np.clip(panel_alpha.astype(np.float32), 0.0, 1.0)
        premul[:, :, :3] = panel_rgb.astype(np.float32) * a_clamped[:, :, None]
        premul[:, :, 3] = a_clamped

        warped = cv2.warpPerspective(
            premul,
            h_mat,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0, 0.0),
        )

        e1_u = np.asarray(e1, dtype=np.float64)
        e2_u = np.asarray(e2, dtype=np.float64)
        e1_u /= np.linalg.norm(e1_u)
        e2_u /= np.linalg.norm(e2_u)
        normal = np.cross(e1_u, e2_u)
        z_cam, dist, hx, hy, hz = self.ray_plane_hit(p0, normal)

        dx = hx - float(p0[0])
        dy = hy - float(p0[1])
        dz = hz - float(p0[2])
        u_m = dx * float(e1_u[0]) + dy * float(e1_u[1]) + dz * float(e1_u[2])
        v_m = dx * float(e2_u[0]) + dy * float(e2_u[1]) + dz * float(e2_u[2])
        valid = (
            (z_cam > 0.05)
            & (u_m >= -0.004)
            & (u_m <= span_u + 0.004)
            & (v_m >= -0.004)
            & (v_m <= span_v + 0.004)
        ).astype(np.float32)

        alpha = np.clip(warped[:, :, 3] * valid, 0.0, 1.0)
        safe_a = np.maximum(alpha, 1e-5)
        rgb = np.clip(warped[:, :, :3] / safe_a[:, :, None], 0.0, 1.0)

        fog_t = 1.0 - np.exp(-np.maximum(0.0, dist - float(fog_start)) * float(fog_density))
        fog_t = np.clip(fog_t, 0.0, 0.55)[:, :, None]
        rgb = rgb * (1.0 - fog_t) + fog_color[None, None, :] * fog_t
        return rgb, alpha, z_cam


def mirror_points_across_plane(pts_3d: np.ndarray, plane_p0: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_3d, dtype=np.float64)
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    d = np.sum((pts - plane_p0) * n, axis=-1, keepdims=True)
    return pts - 2.0 * d * n


# ---------------------------------------------------------------------------
# 3. 2x Supersampled PyCairo Ink & Optics
# ---------------------------------------------------------------------------

def eval_catmull_rom(pts: np.ndarray, samples_per_seg: int = 8, closed: bool = False) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts
    if closed:
        ext = np.vstack([pts[-1:], pts, pts[:2]])
        segs = n
    else:
        ext = np.vstack([pts[:1] * 2 - pts[1:2], pts, pts[-1:] * 2 - pts[-2:-1]])
        segs = n - 1
    out = []
    ts = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)
    t2, t3 = ts * ts, ts * ts * ts
    for i in range(segs):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        c = 0.5 * (
            (2.0 * p1)[None, :]
            + (-p0 + p2)[None, :] * ts[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)[None, :] * t2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3)[None, :] * t3[:, None]
        )
        out.append(c)
    out.append(pts[:1] if closed else pts[-1:])
    return np.vstack(out)


class CairoLayer2x:
    def __init__(self, width: int = 2048, height: int = 1152) -> None:
        self.width = int(width)
        self.height = int(height)
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width * 2, self.height * 2)
        self.ctx = cairo.Context(self.surface)
        self.ctx.scale(2.0, 2.0)
        self.ctx.set_antialias(cairo.ANTIALIAS_BEST)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)

    def fill_poly(self, pts_2d: np.ndarray, rgba: tuple[float, float, float, float]) -> None:
        if len(pts_2d) < 3:
            return
        ctx = self.ctx
        ctx.new_path()
        ctx.move_to(float(pts_2d[0, 0]), float(pts_2d[0, 1]))
        for x, y in pts_2d[1:]:
            ctx.line_to(float(x), float(y))
        ctx.close_path()
        ctx.set_source_rgba(*rgba)
        ctx.fill()

    def fill_poly_linear(self, pts_2d: np.ndarray, p_start: tuple[float, float], p_end: tuple[float, float], stops) -> None:
        if len(pts_2d) < 3:
            return
        ctx = self.ctx
        pat = cairo.LinearGradient(p_start[0], p_start[1], p_end[0], p_end[1])
        for offset, rgba in stops:
            pat.add_color_stop_rgba(offset, *rgba)
        ctx.new_path()
        ctx.move_to(float(pts_2d[0, 0]), float(pts_2d[0, 1]))
        for x, y in pts_2d[1:]:
            ctx.line_to(float(x), float(y))
        ctx.close_path()
        ctx.set_source(pat)
        ctx.fill()

    def fill_poly_radial(self, pts_2d: np.ndarray, c0: tuple[float, float, float], c1: tuple[float, float, float], stops) -> None:
        if len(pts_2d) < 3:
            return
        ctx = self.ctx
        pat = cairo.RadialGradient(c0[0], c0[1], c0[2], c1[0], c1[1], c1[2])
        for offset, rgba in stops:
            pat.add_color_stop_rgba(offset, *rgba)
        ctx.new_path()
        ctx.move_to(float(pts_2d[0, 0]), float(pts_2d[0, 1]))
        for x, y in pts_2d[1:]:
            ctx.line_to(float(x), float(y))
        ctx.close_path()
        ctx.set_source(pat)
        ctx.fill()

    def stroke_catmull_rom(
        self,
        pts_2d: np.ndarray,
        z_depths: np.ndarray | float,
        base_width: float,
        rgba: tuple[float, float, float, float],
        seed: int = 1,
        closed: bool = False,
        fade_dist: float = 4.0,
        taper_ends: float = 0.25,
        tremor: float = 0.22,
        samples_per_seg: int = 8,
    ) -> None:
        pts = np.asarray(pts_2d, dtype=np.float64)
        if len(pts) < 2:
            return
        curve = eval_catmull_rom(pts, samples_per_seg=samples_per_seg, closed=closed)
        n = len(curve)
        zs = np.full(n, float(z_depths), dtype=np.float64) if np.ndim(z_depths) == 0 else np.interp(
            np.linspace(0.0, len(z_depths) - 1, n), np.arange(len(z_depths)), np.asarray(z_depths, dtype=np.float64)
        )
        d = np.gradient(curve, axis=0)
        dlen = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        norm = np.column_stack([-d[:, 1] / dlen[:, 0], d[:, 0] / dlen[:, 0]])

        rng = np.random.default_rng(seed)
        s = np.linspace(0.0, 1.0, n)
        p1, p2 = rng.uniform(0.0, 2.0 * math.pi, size=2)
        wobble = tremor * (0.65 * np.sin(2.0 * math.pi * (3.0 * s) + p1) + 0.35 * np.sin(2.0 * math.pi * (7.0 * s) + p2))
        if closed:
            wobble *= np.sin(math.pi * s)
        curve = curve + norm * wobble[:, None]

        persp = np.clip(1.0 / (1.0 + np.maximum(zs, 0.05) / (fade_dist * 0.65)), 0.35, 1.0)
        taper = (0.85 + 0.15 * np.sin(2.0 * math.pi * s + p1)) if closed else (
            0.25 + 0.75 * smoothstep(0.0, max(taper_ends, 0.05), s) * (1.0 - smoothstep(1.0 - max(taper_ends, 0.05), 1.0, s))
        )
        half_w = (base_width * persp * taper) * 0.5
        ribbon = np.vstack([curve + norm * half_w[:, None], (curve - norm * half_w[:, None])[::-1]])
        ctx = self.ctx
        ctx.set_source_rgba(*rgba)
        ctx.new_path()
        ctx.move_to(float(ribbon[0, 0]), float(ribbon[0, 1]))
        for x, y in ribbon[1:]:
            ctx.line_to(float(x), float(y))
        ctx.close_path()
        ctx.fill()

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        buf = np.frombuffer(self.surface.get_data(), dtype=np.uint8).reshape(self.height * 2, self.width * 2, 4)
        rgba_2x = buf[:, :, [2, 1, 0, 3]]
        pil_1x = Image.fromarray(rgba_2x, mode="RGBA").resize((self.width, self.height), Image.Resampling.LANCZOS)
        arr = np.asarray(pil_1x, dtype=np.float32) / 255.0
        alpha = np.clip(arr[:, :, 3], 0.0, 1.0)
        safe_a = np.maximum(alpha, 1e-5)
        return np.clip(arr[:, :, :3] / safe_a[:, :, None], 0.0, 1.0), alpha


def make_disc_kernel(radius: float = 8.2) -> np.ndarray:
    rad = max(1, int(round(float(radius))))
    ky, kx = np.mgrid[-rad : rad + 1, -rad : rad + 1].astype(np.float32)
    kr = np.sqrt(kx * kx + ky * ky) / max(float(radius), 1.0)
    rim = 0.86 + 0.24 * np.exp(-((kr - 0.84) ** 2) / 0.016)
    disc = np.clip((1.04 - kr) / 0.16, 0.0, 1.0) * rim
    return (disc / max(float(np.sum(disc)), 1e-6)).astype(np.float32)


def apply_foreground_disc_bokeh(
    canvas: np.ndarray, fg_rgb: np.ndarray, fg_alpha: np.ndarray, radius: float = 8.0, chroma_shift_px: float = 1.0
) -> np.ndarray:
    h, w, _ = canvas.shape
    kernel = make_disc_kernel(radius=radius)
    premul = np.empty((h, w, 4), dtype=np.float32)
    premul[:, :, :3] = fg_rgb * fg_alpha[:, :, None]
    premul[:, :, 3] = fg_alpha

    blurred = np.empty_like(premul)
    for c in range(4):
        blurred[:, :, c] = cv2.filter2D(premul[:, :, c], -1, kernel, borderType=cv2.BORDER_CONSTANT)

    m_red = np.array([[1.0, 0.0, chroma_shift_px], [0.0, 1.0, -0.35 * chroma_shift_px]], dtype=np.float32)
    m_blue = np.array([[1.0, 0.0, -chroma_shift_px], [0.0, 1.0, 0.35 * chroma_shift_px]], dtype=np.float32)
    blurred[:, :, 0] = cv2.warpAffine(blurred[:, :, 0], m_red, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    blurred[:, :, 2] = cv2.warpAffine(blurred[:, :, 2], m_blue, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    out_a = np.clip(blurred[:, :, 3], 0.0, 1.0)[:, :, None]
    return np.clip(canvas * (1.0 - out_a) + np.clip(blurred[:, :, :3], 0.0, 1.0), 0.0, 1.0)


def finish_35mm_optics(rgb: np.ndarray, specular_glints: np.ndarray | None = None, seed: int = 909) -> np.ndarray:
    img = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    h, w, _ = img.shape

    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    bright = np.clip((lum - 0.64) / 0.36, 0.0, 1.0) ** 1.25
    if specular_glints is not None:
        bright = np.clip(bright + specular_glints * 0.92, 0.0, 1.40)

    hal_tint = np.array([1.00, 0.86, 0.62], dtype=np.float32)
    h_src = bright[:, :, None] * hal_tint[None, None, :]
    halation = (
        0.18 * cv2.GaussianBlur(h_src, (0, 0), 6.5, borderType=cv2.BORDER_REFLECT_101)
        + 0.16 * cv2.GaussianBlur(h_src, (0, 0), 24.0, borderType=cv2.BORDER_REFLECT_101)
        + 0.13 * cv2.GaussianBlur(h_src, (0, 0), 72.0, borderType=cv2.BORDER_REFLECT_101)
    )
    img = screen_blend(img, halation, strength=1.0)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w * 0.5) / (w * 0.5), (yy - h * 0.5) / (h * 0.5)
    r2 = nx * nx + ny * ny

    ca_scale = 0.0008
    img[:, :, 0] = cv2.remap(img[:, :, 0], (xx + (xx - w * 0.5) * r2 * ca_scale).astype(np.float32), (yy + (yy - h * 0.5) * r2 * ca_scale).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    img[:, :, 2] = cv2.remap(img[:, :, 2], (xx - (xx - w * 0.5) * r2 * ca_scale).astype(np.float32), (yy - (yy - h * 0.5) * r2 * ca_scale).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    img *= (1.0 - 0.14 * np.clip(r2 - 0.18, 0.0, 2.0))[:, :, None]

    black_lift = np.array([0.022, 0.024, 0.032], dtype=np.float32)
    img = black_lift[None, None, :] + img * (1.0 - black_lift[None, None, :])
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    above = np.maximum(img - knee, 0.0)
    img = np.where(img > knee, knee + span * np.tanh(above / span), img)

    rng = np.random.default_rng(seed)
    lum2 = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    env = np.clip(1.10 - 0.90 * (np.clip(lum2, 0.0, 1.0) ** 1.55), 0.12, 1.10)
    g_fine = rng.standard_normal((h, w)).astype(np.float32)
    g_clump = cv2.resize(rng.standard_normal((max(2, h // 2), max(2, w // 2))).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    g_chroma = cv2.resize(rng.standard_normal((max(2, h // 2), max(2, w // 2), 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.clip(img + (0.014 * (0.58 * g_fine + 0.42 * g_clump)[:, :, None] + 0.004 * g_chroma) * env[:, :, None], 0.0, 1.0)

    r_ch, g_ch, b_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y_ch = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    i_ch = 0.596 * r_ch - 0.274 * g_ch - 0.322 * b_ch
    q_ch = 0.211 * r_ch - 0.523 * g_ch + 0.312 * b_ch
    i_soft = cv2.GaussianBlur(i_ch, (0, 0), sigmaX=1.05, sigmaY=0.15, borderType=cv2.BORDER_REFLECT_101)
    q_soft = cv2.GaussianBlur(q_ch, (0, 0), sigmaX=1.15, sigmaY=0.15, borderType=cv2.BORDER_REFLECT_101)
    y_blur = cv2.GaussianBlur(y_ch, (0, 0), sigmaX=0.85, sigmaY=0.05, borderType=cv2.BORDER_REFLECT_101)
    y_out = np.clip(y_ch + 0.07 * (y_ch - y_blur), 0.0, 1.0)

    out = np.empty_like(img)
    out[:, :, 0] = y_out + 0.956 * i_soft + 0.621 * q_soft
    out[:, :, 1] = y_out - 0.272 * i_soft - 0.647 * q_soft
    out[:, :, 2] = y_out - 1.106 * i_soft + 1.703 * q_soft
    out = np.clip(out, 0.0, 1.0)

    row_shift = cv2.GaussianBlur(rng.normal(0.0, 0.065, size=h).astype(np.float32)[:, None], (1, 5), 0.9)[:, 0]
    out = cv2.remap(out, (xx + row_shift[:, None]).astype(np.float32), yy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    scan = 1.0 - 0.006 * (0.5 - 0.5 * np.cos(math.pi * yy))
    return np.clip(out * scan[:, :, None], 0.0, 1.0)


def save_image(rgb_float: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.round(np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(u8, mode="RGB")
    if out_path.suffix.lower() == ".webp":
        img.save(out_path, format="WEBP", quality=90, method=6)
    else:
        img.save(out_path, format="PNG", compress_level=6)


def save_crop_boxes(
    rgb_float: np.ndarray, crops_dir: Path, centers_px: Sequence[tuple[str, tuple[float, float]]], crop_w: int = 600, crop_h: int = 450
) -> None:
    h, w, _ = rgb_float.shape
    crops_dir.mkdir(parents=True, exist_ok=True)
    cw, ch = min(crop_w, w), min(crop_h, h)
    for name, (cx, cy) in centers_px:
        x0 = int(np.clip(round(cx - cw * 0.5), 0, w - cw))
        y0 = int(np.clip(round(cy - ch * 0.5), 0, h - ch))
        save_image(rgb_float[y0 : y0 + ch, x0 : x0 + cw], crops_dir / name)


# ---------------------------------------------------------------------------
# 4. Composition: Open Winter-Dusk Sky Over Abbey Nave & Elizabeth Tower
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 2048, 1152

# Off-axis framing: Elizabeth Tower & full spire at ~60% width with 14% open sky above,
# Abbey flying buttresses & gable on center-left, and 30% wet reflecting flagstones below.
CAM_EYE = np.array([-1.6, -1.0, 1.75], dtype=np.float64)
CAM_TARGET = np.array([1.7, 26.0, 5.55], dtype=np.float64)
CAM_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

_sun_raw = np.array([0.62, -0.25, -0.74], dtype=np.float64)
SUN_DIR = _sun_raw / np.linalg.norm(_sun_raw)
MIST_RGB = hex_rgb("#46596E")

X_WEST_WALL = -4.6
Y_YARD_END = 22.0
Z_WEST_H = 6.9
WEST_WINDOWS = [(4.4, 7.0, 1.5, 4.6), (9.8, 12.4, 1.5, 4.6), (15.0, 17.4, 1.5, 4.6)]


def draw_projected_poly(layer: CairoLayer2x, cam: Camera3D, pts3d, rgba) -> np.ndarray:
    u, v, zc = cam.project_points(np.asarray(pts3d, dtype=np.float64))
    if np.any(np.asarray(zc) <= 0.2):
        return np.empty((0, 2), dtype=np.float64)
    pts2d = np.column_stack([u, v])
    layer.fill_poly(pts2d, rgba)
    return pts2d


def draw_projected_linear(layer: CairoLayer2x, cam: Camera3D, pts3d, p0, p1, stops) -> np.ndarray:
    u, v, zc = cam.project_points(np.asarray(pts3d, dtype=np.float64))
    if np.any(np.asarray(zc) <= 0.2):
        return np.empty((0, 2), dtype=np.float64)
    su, sv, _ = cam.project_points(np.asarray(p0, dtype=np.float64))
    eu, ev, _ = cam.project_points(np.asarray(p1, dtype=np.float64))
    pts2d = np.column_stack([u, v])
    layer.fill_poly_linear(pts2d, (float(su), float(sv)), (float(eu), float(ev)), stops)
    return pts2d


def paint_dusk_sky(cam: Camera3D, tooth: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    ny = np.clip((cam.horizon_y - yy + 45.0) / max(cam.horizon_y + 45.0, 1.0), 0.0, 1.15)
    nx = xx / float(WIDTH)

    horizon = hex_rgb("#C28D6A")   # Winter amber-apricot glow near Thames horizon
    lower = hex_rgb("#687C91")     # Misty slate dusk
    mid = hex_rgb("#354C66")
    zenith = hex_rgb("#162335")

    t1 = smoothstep(0.0, 0.28, ny)
    t2 = smoothstep(0.22, 0.62, ny)
    t3 = smoothstep(0.50, 0.98, ny)
    sky = horizon[None, None, :] * (1.0 - t1[:, :, None]) + lower[None, None, :] * t1[:, :, None]
    sky = sky * (1.0 - t2[:, :, None]) + mid[None, None, :] * t2[:, :, None]
    sky = sky * (1.0 - t3[:, :, None]) + zenith[None, None, :] * t3[:, :, None]

    glow = np.exp(-(((nx - 0.61) / 0.34) ** 2 + ((ny - 0.14) / 0.22) ** 2)) * 0.32
    sky = sky + glow[:, :, None] * hex_rgb("#E5A26E")[None, None, :]

    clouds = fbm_gouache(HEIGHT, WIDTH, octaves=((170.0, 72.0, 0.55), (60.0, 32.0, 0.30), (20.0, 14.0, 0.15)), seed=411)
    c_band = smoothstep(0.18, 0.88, ny) * smoothstep(-0.2, 1.25, clouds) * 0.36
    sky = sky * (1.0 - c_band[:, :, None]) + hex_rgb("#24364B")[None, None, :] * c_band[:, :, None]

    return paint_isotropic_gouache_dabs(sky, np.ones((HEIGHT, WIDTH), dtype=np.float32), tooth, dab_strength=0.025, num_dabs=88, tooth_weight=0.011, seed=419)


def paint_tower_and_abbey(cam: Camera3D, tooth: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float]:
    """Elizabeth Tower rising prominently at ~60% frame width above Westminster Abbey."""
    layer = CairoLayer2x(WIDTH, HEIGHT)

    tx, ty = 17.6, 96.0
    tw = 5.8
    z_base = 8.0
    z_dial0 = 24.0
    z_dial1 = 30.8
    z_belfry0 = 31.1
    z_belfry1 = 36.2
    z_spire0 = 36.2
    z_spire_mid = 41.2
    z_finial = 45.4

    # Main sandstone shaft with sunlit right edge & cool indigo shadow side
    draw_projected_linear(
        layer,
        cam,
        [[tx - tw * 0.5, ty, z_base], [tx + tw * 0.5, ty, z_base], [tx + tw * 0.48, ty, z_dial0], [tx - tw * 0.48, ty, z_dial0]],
        [tx + tw * 0.5, ty, z_base],
        [tx - tw * 0.5, ty, z_base],
        [(0.0, (0.42, 0.42, 0.45, 1.0)), (0.55, (0.28, 0.31, 0.37, 1.0)), (1.0, (0.18, 0.21, 0.28, 1.0))],
    )
    # Recessed gothic panel ribs along tower shaft
    for fx in (-0.34, -0.17, 0.0, 0.17, 0.34):
        p0 = cam.project_points(np.array([tx + tw * fx, ty, z_base + 1.0]))
        p1 = cam.project_points(np.array([tx + tw * fx * 0.96, ty, z_dial0 - 0.6]))
        layer.stroke_catmull_rom(np.array([[p0[0], p0[1]], [p1[0], p1[1]]]), ty, 1.5, (0.12, 0.14, 0.19, 0.58), seed=60 + int(fx * 20), tremor=0.14)

    # Corbelled clock gallery + corner turret pinnacles
    cw = tw * 1.14
    draw_projected_linear(
        layer,
        cam,
        [[tx - cw * 0.5, ty, z_dial0], [tx + cw * 0.5, ty, z_dial0], [tx + cw * 0.5, ty, z_dial1], [tx - cw * 0.5, ty, z_dial1]],
        [tx + cw * 0.5, ty, z_dial0],
        [tx - cw * 0.5, ty, z_dial0],
        [(0.0, (0.45, 0.42, 0.42, 1.0)), (1.0, (0.20, 0.23, 0.30, 1.0))],
    )
    for side in (-1.0, 1.0):
        px = tx + side * cw * 0.50
        draw_projected_poly(
            layer,
            cam,
            [[px - 0.44, ty, z_dial0], [px + 0.44, ty, z_dial0], [px + 0.30, ty, z_belfry0 + 3.2], [px, ty, z_belfry0 + 5.8], [px - 0.30, ty, z_belfry0 + 3.2]],
            (0.18, 0.21, 0.28, 1.0),
        )

    # Glowing Clock Dial + Cast-iron Bezel + 12 Hour Markers & Hands
    dial_c = np.array([tx, ty - 0.15, 0.5 * (z_dial0 + z_dial1)])
    du, dv, _ = cam.project_points(dial_c)
    dial_r = 2.60
    eu, _, _ = cam.project_points(dial_c + np.array([dial_r, 0.0, 0.0]))
    dial_px_r = abs(float(eu) - float(du))

    dial_pts, bezel_pts = [], []
    for k in range(48):
        ang = 2.0 * math.pi * k / 48.0
        p_d = dial_c + np.array([dial_r * math.cos(ang), 0.0, dial_r * math.sin(ang)])
        p_b = dial_c + np.array([dial_r * 1.15 * math.cos(ang), 0.0, dial_r * 1.15 * math.sin(ang)])
        pu, pv, _ = cam.project_points(p_d)
        bu, bv, _ = cam.project_points(p_b)
        dial_pts.append([pu, pv])
        bezel_pts.append([bu, bv])

    layer.fill_poly(np.array(bezel_pts), (0.14, 0.12, 0.11, 1.0))
    layer.fill_poly_radial(
        np.array(dial_pts),
        (float(du), float(dv), 1.5),
        (float(du), float(dv), dial_px_r * 1.08),
        [(0.0, (1.00, 0.95, 0.76, 1.0)), (0.55, (0.98, 0.83, 0.52, 1.0)), (0.88, (0.88, 0.64, 0.34, 1.0)), (1.0, (0.56, 0.38, 0.20, 1.0))],
    )
    inner = []
    for k in range(40):
        ang = 2.0 * math.pi * k / 40.0
        pu, pv, _ = cam.project_points(dial_c + np.array([dial_r * 0.72 * math.cos(ang), 0.0, dial_r * 0.72 * math.sin(ang)]))
        inner.append([pu, pv])
    layer.stroke_catmull_rom(np.array(inner), ty, 1.5, (0.18, 0.14, 0.10, 0.76), closed=True, tremor=0.05)

    for hr in range(12):
        ang = 2.0 * math.pi * hr / 12.0
        p0 = cam.project_points(dial_c + np.array([dial_r * 0.72 * math.cos(ang), 0.0, dial_r * 0.72 * math.sin(ang)]))
        p1 = cam.project_points(dial_c + np.array([dial_r * 0.92 * math.cos(ang), 0.0, dial_r * 0.92 * math.sin(ang)]))
        layer.stroke_catmull_rom(np.array([[p0[0], p0[1]], [p1[0], p1[1]]]), ty, 1.8, (0.16, 0.12, 0.10, 0.85), seed=90 + hr, tremor=0.04)

    for ang_deg, r_f, w_h in [(90.0, 0.78, 3.0), (210.0, 0.52, 3.8)]:
        rad = math.radians(ang_deg)
        pt = cam.project_points(dial_c + np.array([dial_r * r_f * math.cos(rad), 0.0, dial_r * r_f * math.sin(rad)]))
        layer.stroke_catmull_rom(np.array([[float(du), float(dv)], [pt[0], pt[1]]]), ty, w_h, (0.11, 0.09, 0.08, 0.96), seed=99, tremor=0.03)

    # Open Belfry Chamber with Warm Honey Louver Glow + Bell Silhouette Hint
    bw = cw * 0.88
    draw_projected_poly(
        layer,
        cam,
        [[tx - bw * 0.5, ty, z_belfry0], [tx + bw * 0.5, ty, z_belfry0], [tx + bw * 0.47, ty, z_belfry1], [tx - bw * 0.47, ty, z_belfry1]],
        (0.16, 0.19, 0.25, 1.0),
    )
    for bx_f in (-0.30, -0.10, 0.10, 0.30):
        lx0, lx1 = tx + bw * (bx_f - 0.055), tx + bw * (bx_f + 0.055)
        draw_projected_poly(
            layer,
            cam,
            [[lx0, ty, z_belfry0 + 0.9], [lx1, ty, z_belfry0 + 0.9], [lx1, ty, z_belfry1 - 1.0], [lx0, ty, z_belfry1 - 1.0]],
            (0.95, 0.68, 0.34, 0.92),
        )
        # Slanted acoustic louver bars across each belfry opening
        for lz in np.linspace(z_belfry0 + 1.5, z_belfry1 - 1.4, 5):
            p0 = cam.project_points(np.array([lx0, ty, lz]))
            p1 = cam.project_points(np.array([lx1, ty, lz]))
            layer.stroke_catmull_rom(np.array([[p0[0], p0[1]], [p1[0], p1[1]]]), ty, 1.6, (0.14, 0.13, 0.14, 0.85), seed=140, tremor=0.05)

    # Tiered Iron Spire & Ayrton Beacon Lantern
    draw_projected_linear(
        layer,
        cam,
        [[tx - bw * 0.54, ty, z_spire0], [tx + bw * 0.54, ty, z_spire0], [tx + 0.70, ty, z_spire_mid], [tx - 0.70, ty, z_spire_mid]],
        [tx + bw * 0.4, ty, z_spire0],
        [tx - bw * 0.4, ty, z_spire0],
        [(0.0, (0.20, 0.24, 0.31, 1.0)), (1.0, (0.11, 0.15, 0.22, 1.0))],
    )
    draw_projected_poly(layer, cam, [[tx - 0.78, ty, z_spire_mid], [tx + 0.78, ty, z_spire_mid], [tx, ty, z_finial]], (0.13, 0.16, 0.22, 1.0))
    # Ayrton light glowing amber on spire lantern
    ayr = cam.project_points(np.array([tx, ty, z_spire_mid + 0.4]))
    layer.fill_poly_radial(
        np.array([[ayr[0] - 28, ayr[1] - 28], [ayr[0] + 28, ayr[1] - 28], [ayr[0] + 28, ayr[1] + 28], [ayr[0] - 28, ayr[1] + 28]]),
        (float(ayr[0]), float(ayr[1]), 1.0),
        (float(ayr[0]), float(ayr[1]), 22.0),
        [(0.0, (1.0, 0.94, 0.74, 1.0)), (0.35, (0.96, 0.68, 0.34, 0.85)), (1.0, (0.15, 0.18, 0.24, 0.0))],
    )

    # Westminster Abbey Nave Roof Ridge, Flying Buttresses & Gothic Pinnacles (mid-distance barrier!)
    ay = 48.0
    draw_projected_linear(
        layer,
        cam,
        [[-36.0, ay, 0.0], [56.0, ay, 0.0], [56.0, ay, 13.2], [5.8, ay, 13.2], [0.2, ay, 21.0], [-5.4, ay, 13.2], [-36.0, ay, 13.2]],
        [0.2, ay, 21.0],
        [0.2, ay, 6.0],
        [(0.0, (0.14, 0.18, 0.24, 1.0)), (0.55, (0.18, 0.22, 0.29, 1.0)), (1.0, (0.11, 0.14, 0.19, 1.0))],
    )
    # Clerestory Gothic pointed arcades along Abbey east nave
    for cx_abb in (-11.0, -7.2, 4.6, 8.8, 13.0, 17.2, 21.4, 25.6):
        draw_projected_poly(
            layer, cam,
            [[cx_abb - 0.65, ay, 7.6], [cx_abb + 0.65, ay, 7.6], [cx_abb + 0.65, ay, 11.2], [cx_abb, ay, 12.2], [cx_abb - 0.65, ay, 11.2]],
            (0.09, 0.12, 0.17, 0.72),
        )
    # Blind lancets + Rose window + Flying buttress piers along Abbey ridge
    for lx_off, lz_top in [(-1.4, 15.6), (0.2, 16.9), (1.8, 15.6)]:
        draw_projected_poly(
            layer, cam,
            [[lx_off - 0.48, ay, 11.8], [lx_off + 0.48, ay, 11.8], [lx_off + 0.48, ay, lz_top - 0.65], [lx_off, ay, lz_top], [lx_off - 0.48, ay, lz_top - 0.65]],
            (0.09, 0.12, 0.16, 0.78),
        )
    rose_pts = [cam.project_points(np.array([0.2 + 1.02 * math.cos(2 * math.pi * k / 32), ay, 18.2 + 1.02 * math.sin(2 * math.pi * k / 32)]))[:2] for k in range(32)]
    layer.fill_poly(np.array(rose_pts), (0.09, 0.12, 0.16, 0.72))
    layer.stroke_catmull_rom(np.array(rose_pts), ay, 1.6, (0.30, 0.36, 0.44, 0.55), closed=True, tremor=0.06)

    # Vertical lead-seam & buttress ribs across full Abbey facade
    for rx in np.linspace(-24.0, 38.0, 34):
        top_z = 20.8 - abs(rx - 0.2) * 1.36 if -5.0 < rx < 5.4 else 13.1
        p0 = cam.project_points(np.array([rx, ay, 6.5]))
        p1 = cam.project_points(np.array([rx, ay, top_z]))
        layer.stroke_catmull_rom(np.array([[p0[0], p0[1]], [p1[0], p1[1]]]), ay, 1.4, (0.26, 0.32, 0.40, 0.36), seed=210 + int(rx + 40), tremor=0.08)

    # Stepped pinnacles framed strictly clear of Elizabeth Tower (tx/2 = 8.8m at Abbey depth)
    for px, pz0, pz1, pw in [(-14.0, 13.0, 18.8, 0.84), (-9.0, 13.0, 20.2, 0.90), (-5.4, 13.0, 22.0, 1.00), (0.2, 21.0, 24.0, 0.74), (4.6, 13.0, 20.2, 0.92), (17.2, 13.0, 19.2, 0.84), (22.4, 13.0, 18.4, 0.80), (27.4, 13.0, 17.8, 0.76)]:
        draw_projected_poly(
            layer, cam,
            [[px - pw * 0.5, ay, pz0], [px + pw * 0.5, ay, pz0], [px + pw * 0.32, ay, pz0 + (pz1 - pz0) * 0.46], [px, ay, pz1], [px - pw * 0.32, ay, pz0 + (pz1 - pz0) * 0.46]],
            (0.12, 0.15, 0.21, 1.0),
        )
        p_b = cam.project_points(np.array([px + pw * 0.32, ay, pz0 + 0.5]))
        p_t = cam.project_points(np.array([px, ay, pz1]))
        layer.stroke_catmull_rom(np.array([[p_b[0], p_b[1]], [p_t[0], p_t[1]]]), ay, 1.4, (0.44, 0.48, 0.56, 0.45), seed=310 + int(px + 20), tremor=0.10)

    rgb, alpha = layer.finish()
    rgb = paint_isotropic_gouache_dabs(rgb, alpha, tooth, dab_strength=0.032, warm_bounce_in_shadow=0.016, num_dabs=76, tooth_weight=0.011, seed=503)

    # Winter dusk mist veil separating 48m Abbey from 84m Elizabeth Tower
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    mist_band = np.exp(-(((yy - cam.horizon_y + 210.0) / 160.0) ** 2))
    mist_tex = smoothstep(-1.3, 1.5, fbm_gouache(HEIGHT, WIDTH, octaves=((130.0, 56.0, 0.65), (44.0, 24.0, 0.35)), seed=509))
    mist_amt = np.clip(0.22 + 0.24 * mist_band * mist_tex, 0.0, 0.52) * alpha
    mist_col = hex_rgb("#5A6878")
    rgb = rgb * (1.0 - mist_amt[:, :, None]) + mist_col[None, None, :] * mist_amt[:, :, None]

    # Honey dial & belfry aura halo
    d_dial = np.sqrt((xx - float(du)) ** 2 + (yy - float(dv)) ** 2)
    rgb = screen_blend(rgb, (np.exp(-((d_dial / max(dial_px_r * 1.85, 18.0)) ** 2)) * 0.42)[:, :, None] * hex_rgb("#F8C472")[None, None, :], 1.0)
    return np.clip(rgb, 0.0, 1.0), alpha, (float(du), float(dv)), dial_px_r


def make_left_westminster_facade(cam: Camera3D, tooth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth Band 3A: Left-hand School Yard limestone wall with Voronoi ashlar & glowing mullioned windows."""
    uw, uh = 1500, 740
    span_u = 20.5
    span_v = Z_WEST_H

    joint, stone_tone = voronoi_mortar_field(uh, uw, span_u, span_v, cell_size_m=0.44, jitter=0.20, warp_amp_m=0.012, seed=711)
    vv, uu = np.mgrid[0:uh, 0:uw].astype(np.float32)
    y_m = 1.5 + (uu / max(uw - 1, 1)) * span_u
    z_m = span_v - (vv / max(uh - 1, 1)) * span_v

    stone = np.broadcast_to(hex_rgb("#3C3D42"), (uh, uw, 3)).copy()
    stone += 0.032 * stone_tone[:, :, None]
    stone *= (0.70 + 0.30 * smoothstep(0.008, 0.024, joint))[:, :, None]

    # Carved cornice stringcourse & coping parapet
    coping = smoothstep(Z_WEST_H - 0.55, Z_WEST_H - 0.30, z_m)
    stone = stone * (1.0 - 0.18 * coping[:, :, None]) + hex_rgb("#54575F")[None, None, :] * (0.18 * coping[:, :, None])

    win_glow = np.zeros((uh, uw), dtype=np.float32)
    win_rgb = np.zeros((uh, uw, 3), dtype=np.float32)

    for wy0, wy1, wz0, wz1 in WEST_WINDOWS:
        yc, hw = 0.5 * (wy0 + wy1), 0.5 * (wy1 - wy0)
        dy = np.abs(y_m - yc)
        arch_z = wz1 - 0.42 + 0.44 * np.sqrt(np.maximum(0.0, 1.0 - (dy / max(hw, 1e-4)) ** 2))
        in_win = (y_m >= wy0) & (y_m <= wy1) & (z_m >= wz0) & (z_m <= arch_z)
        chamfer = (y_m >= wy0 - 0.16) & (y_m <= wy1 + 0.16) & (z_m >= wz0 - 0.18) & (z_m <= arch_z + 0.16) & (~in_win)
        stone = np.where(chamfer[:, :, None], stone * 0.70 + hex_rgb("#A66D3E")[None, None, :] * 0.36, stone)

        ny_w = np.clip((z_m - wz0) / max(wz1 - wz0, 1e-4), 0.0, 1.0)
        nx_w = (y_m - wy0) / max(wy1 - wy0, 1e-4)
        pane = hex_rgb("#EFA857")[None, None, :] * (1.0 - ny_w[:, :, None]) + hex_rgb("#D27D3B")[None, None, :] * ny_w[:, :, None]
        core = np.exp(-(((ny_w - 0.42) / 0.35) ** 2 + ((nx_w - 0.50) / 0.42) ** 2))
        pane = pane * (1.0 - 0.55 * core[:, :, None]) + hex_rgb("#FFE5AA")[None, None, :] * (0.55 * core[:, :, None])

        mull = (np.abs(nx_w - 0.333) < 0.032) | (np.abs(nx_w - 0.667) < 0.032) | (np.abs(ny_w - 0.48) < 0.030)
        # Anti-aliased lead came strength (no binary dashed moire on oblique masonry perspectives!)
        lead_s = np.maximum(
            smoothstep(0.41, 0.49, np.abs((nx_w * 6.0) % 1.0 - 0.5)),
            smoothstep(0.41, 0.49, np.abs((ny_w * 8.0) % 1.0 - 0.5)),
        ) * (~mull).astype(np.float32)
        pane = pane * (1.0 - 0.82 * lead_s[:, :, None]) + hex_rgb("#2C211B")[None, None, :] * (0.82 * lead_s[:, :, None])
        pane = np.where(mull[:, :, None], hex_rgb("#242224")[None, None, :], pane)

        win_glow = np.maximum(win_glow, (in_win & (~mull)).astype(np.float32) * (1.0 - 0.7 * lead_s))
        win_rgb = np.where(in_win[:, :, None], pane, win_rgb)
        stone = np.where(in_win[:, :, None], pane, stone)

    p0 = np.array([X_WEST_WALL, 1.5, span_v], dtype=np.float64)
    e1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    w_rgb, w_a, _ = cam.warp_plane_panel(np.clip(stone, 0.0, 1.0), np.ones((uh, uw), dtype=np.float32), p0, e1, e2, span_u, span_v, MIST_RGB, fog_start=8.0, fog_density=0.014)
    g_rgb, g_a, _ = cam.warp_plane_panel(np.clip(win_rgb, 0.0, 1.0), win_glow, p0, e1, e2, span_u, span_v, MIST_RGB, fog_start=18.0, fog_density=0.004)
    w_rgb = paint_isotropic_gouache_dabs(w_rgb, w_a * (1.0 - g_a), tooth, dab_strength=0.038, warm_bounce_in_shadow=0.022, num_dabs=80, seed=733)
    w_rgb = alpha_composite(w_rgb, g_rgb, g_a)
    return w_rgb, w_a, g_a


def make_low_courtyard_balustrade(cam: Camera3D, tooth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Low Portland dwarf wall & Cloister terrace parapet at y = Y_YARD_END, spanning cleanly off-frame."""
    uw, uh = 2000, 280
    span_u = 34.0
    span_v = 1.95
    joint, stone_tone = voronoi_mortar_field(uh, uw, span_u, span_v, cell_size_m=0.45, jitter=0.22, warp_amp_m=0.012, seed=619)
    vv, uu = np.mgrid[0:uh, 0:uw].astype(np.float32)
    z_m = span_v - (vv / max(uh - 1, 1)) * span_v

    stone = np.broadcast_to(hex_rgb("#353941"), (uh, uw, 3)).copy()
    stone += 0.030 * stone_tone[:, :, None]
    stone *= (0.68 + 0.32 * smoothstep(0.008, 0.024, joint))[:, :, None]
    coping = smoothstep(span_v - 0.28, span_v - 0.12, z_m)
    stone = stone * (1.0 - 0.28 * coping[:, :, None]) + hex_rgb("#5A616C")[None, None, :] * (0.28 * coping[:, :, None])

    p0 = np.array([-4.8, Y_YARD_END, span_v], dtype=np.float64)
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    rgb, a, _ = cam.warp_plane_panel(np.clip(stone, 0.0, 1.0), np.ones((uh, uw), dtype=np.float32), p0, e1, e2, span_u, span_v, MIST_RGB, fog_start=12.0, fog_density=0.014)
    rgb = paint_isotropic_gouache_dabs(rgb, a, tooth, dab_strength=0.034, warm_bounce_in_shadow=0.020, num_dabs=72, seed=623)
    return rgb, a


def paint_wet_flagstones(cam: Camera3D, upper_scene: np.ndarray, tooth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wet Little Dean's Yard flagstones with SUN_DIR window spill & grazing Fresnel reflections."""
    uw, uh = 1600, 1600
    x0, y0 = X_WEST_WALL - 0.4, -0.5
    span_u = 32.0
    span_v = Y_YARD_END + 1.0

    joint, stone_tone = voronoi_mortar_field(uh, uw, span_u, span_v, cell_size_m=0.68, jitter=0.26, warp_amp_m=0.018, seed=911)
    vv, uu = np.mgrid[0:uh, 0:uw].astype(np.float32)
    wx = x0 + (uu / max(uw - 1, 1)) * span_u
    wy = y0 + (vv / max(uh - 1, 1)) * span_v

    stone = np.broadcast_to(hex_rgb("#262D37"), (uh, uw, 3)).copy()
    stone += 0.030 * stone_tone[:, :, None]
    stone *= (0.64 + 0.36 * smoothstep(0.010, 0.030, joint))[:, :, None]

    # Exact SUN_DIR raytrace onto left wall windows
    t_ray = (X_WEST_WALL - wx) / min(-float(SUN_DIR[0]), -1e-5)
    y_hit = wy - t_ray * float(SUN_DIR[1])
    z_hit = 0.0 - t_ray * float(SUN_DIR[2])

    spill = np.zeros((uh, uw), dtype=np.float32)
    for wy0, wy1, wz0, wz1 in WEST_WINDOWS:
        in_x = smoothstep(wy0 - 0.15, wy0 + 0.15, y_hit) * (1.0 - smoothstep(wy1 - 0.15, wy1 + 0.15, y_hit))
        in_z = smoothstep(wz0 - 0.15, wz0 + 0.15, z_hit) * (1.0 - smoothstep(wz1 - 0.15, wz1 + 0.15, z_hit))
        falloff = np.exp(-0.095 * np.maximum(t_ray - 1.0, 0.0))
        spill = np.maximum(spill, in_x * in_z * falloff * (t_ray > 0.0).astype(np.float32))
    stone += spill[:, :, None] * hex_rgb("#E2934B")[None, None, :] * 0.52

    p0 = np.array([x0, y0, 0.0], dtype=np.float64)
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    floor_rgb, floor_a, _ = cam.warp_plane_panel(np.clip(stone, 0.0, 1.0), np.ones((uh, uw), dtype=np.float32), p0, e1, e2, span_u, span_v, MIST_RGB, fog_start=8.5, fog_density=0.016)
    floor_rgb = paint_isotropic_gouache_dabs(floor_rgb, floor_a, tooth, dab_strength=0.032, warm_bounce_in_shadow=0.022, num_dabs=84, seed=941)

    _, _, hx, hy, _ = cam.ray_plane_hit(p0, np.array([0.0, 0.0, 1.0]))
    puddle_field = fbm_gouache(HEIGHT, WIDTH, octaves=((118.0, 62.0, 0.55), (42.0, 26.0, 0.30), (16.0, 10.0, 0.15)), seed=953)
    puddle_mask = smoothstep(-0.30, 0.70, puddle_field + 0.25 * np.exp(-((hx - 0.4) / 3.8) ** 2))
    wet_sheen = np.clip(0.30 + 0.70 * puddle_mask, 0.0, 1.0) * floor_a

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    rip = np.sin(hy * 19.0 + hx * 3.1) * 0.45 + np.sin(hy * 33.7 - hx * 5.2 + 1.4) * 0.35
    rip_amp = (1.6 + 2.2 * (1.0 - smoothstep(cam.horizon_y, HEIGHT, yy))) * puddle_mask
    y_refl = np.clip(2.0 * cam.horizon_y - yy + rip * rip_amp * 2.1, 0.0, HEIGHT - 1.0).astype(np.float32)
    x_refl = np.clip(xx + rip * rip_amp * 0.90, 0.0, WIDTH - 1.0).astype(np.float32)
    refl_scene = cv2.remap(upper_scene, x_refl, y_refl, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    win_refl = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    floor_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    floor_p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    for wy0, wy1, wz0, wz1 in WEST_WINDOWS:
        c3d = np.array([[X_WEST_WALL, wy0, wz0], [X_WEST_WALL, wy1, wz0], [X_WEST_WALL, wy1, wz1], [X_WEST_WALL, wy0, wz1]], dtype=np.float64)
        mu, mv, mzc = cam.project_points(mirror_points_across_plane(c3d, floor_p0, floor_n))
        if np.all(np.asarray(mzc) > 0.2):
            mask_w = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
            cv2.fillConvexPoly(mask_w, np.round(np.column_stack([mu, mv])).astype(np.int32), 1.0, cv2.LINE_AA)
            mask_w = cv2.GaussianBlur(mask_w, (0, 0), sigmaX=3.5, sigmaY=11.0, borderType=cv2.BORDER_REFLECT_101)
            win_refl += mask_w[:, :, None] * hex_rgb("#F4B25E")[None, None, :] * 0.60

    dist_t = np.clip((yy - cam.horizon_y) / max(1.0, HEIGHT - cam.horizon_y), 0.0, 1.0)
    fresnel = (0.14 + 0.56 * ((1.0 - dist_t) ** 1.4)) * wet_sheen
    final_floor = floor_rgb * (1.0 - fresnel[:, :, None]) + np.clip(refl_scene * 0.85 + win_refl, 0.0, 1.0) * fresnel[:, :, None]
    glints = np.clip((win_refl[:, :, 0] - 0.16) * puddle_mask * 1.4, 0.0, 1.0)
    return np.clip(final_floor, 0.0, 1.0), floor_a, glints


def paint_foreground_branches() -> tuple[np.ndarray, np.ndarray]:
    """Out-of-focus bare winter plane-tree twigs framing upper-right corner."""
    fg = CairoLayer2x(WIDTH, HEIGHT)
    rng = np.random.default_rng(977)
    bough = np.array([[WIDTH + 30.0, 95.0], [WIDTH - 135.0, 145.0], [WIDTH - 295.0, 185.0], [WIDTH - 420.0, 222.0]])
    fg.stroke_catmull_rom(bough, 0.8, 22.0, (0.07, 0.08, 0.10, 0.96), seed=978, taper_ends=0.35, tremor=0.50)
    for idx in range(7):
        t = 0.22 + 0.11 * idx
        bx, by = WIDTH + 20.0 - t * 440.0, 95.0 + t * 125.0
        ang = math.pi + rng.uniform(-0.72, 0.72) + (0.40 if idx % 2 == 0 else -0.44)
        length = rng.uniform(70.0, 148.0)
        mx = bx + 0.52 * length * math.cos(ang) + rng.uniform(-14.0, 14.0)
        my = by + 0.52 * length * math.sin(ang) + rng.uniform(-12.0, 12.0)
        tx, ty = bx + length * math.cos(ang), by + length * math.sin(ang)
        fg.stroke_catmull_rom(np.array([[bx, by], [mx, my], [tx, ty]]), 0.8, rng.uniform(5.0, 9.0), (0.08, 0.09, 0.11, 0.92), seed=980 + idx, taper_ends=0.45, tremor=0.45)
    return fg.finish()


def render_scene(out_path: Path, crops_dir: Path | None = None) -> None:
    cam = Camera3D(CAM_EYE, CAM_TARGET, CAM_UP, fov_deg=56.0, width=WIDTH, height=HEIGHT)
    tooth = make_paper_tooth(HEIGHT, WIDTH, seed=707)

    canvas = paint_dusk_sky(cam, tooth)
    tower_rgb, tower_a, dial_px, dial_px_r = paint_tower_and_abbey(cam, tooth)
    canvas = alpha_composite(canvas, tower_rgb, tower_a)

    wall_rgb, wall_a = make_low_courtyard_balustrade(cam, tooth)
    canvas = alpha_composite(canvas, wall_rgb, wall_a)

    left_rgb, left_a, _ = make_left_westminster_facade(cam, tooth)
    canvas = alpha_composite(canvas, left_rgb, left_a)

    upper_scene = canvas.copy()
    floor_rgb, floor_a, floor_glints = paint_wet_flagstones(cam, upper_scene, tooth)
    canvas = alpha_composite(canvas, floor_rgb, floor_a)

    fg_rgb, fg_a = paint_foreground_branches()
    canvas = apply_foreground_disc_bokeh(canvas, fg_rgb, fg_a, radius=7.8, chroma_shift_px=1.0)

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    dial_glint = np.exp(-(((xx - dial_px[0]) ** 2 + (yy - dial_px[1]) ** 2) / (max(dial_px_r * 1.40, 36.0) ** 2))) * 0.88
    final_rgb = finish_35mm_optics(canvas, specular_glints=np.clip(floor_glints * 0.72 + dial_glint, 0.0, 1.0), seed=909)
    save_image(final_rgb, out_path)

    if crops_dir is not None:
        win_u, win_v, _ = cam.project_points(np.array([X_WEST_WALL, 12.1, 3.7]))
        save_crop_boxes(
            final_rgb,
            crops_dir,
            [
                ("crop_1_dial_belfry.png", (dial_px[0] - 40.0, dial_px[1] - 60.0)),
                ("crop_2_left_mullions.png", (float(win_u), float(win_v))),
                ("crop_3_wet_flagstones.png", (1120.0, 960.0)),
            ],
            crop_w=600,
            crop_h=450,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paint Little Dean's Yard dusk cel study.")
    parser.add_argument("--out", type=Path, default=Path("still.webp"), help="Output image path.")
    parser.add_argument("--crops-dir", type=Path, default=None, help="Optional 1:1 macro crop directory.")
    args = parser.parse_args()
    render_scene(args.out, args.crops_dir)


if __name__ == "__main__":
    main()
