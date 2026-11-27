#!/usr/bin/env python3
"""Pure-Python 35mm cel-and-gouache still (Pass 2): dusk in Little Dean's Yard,
Westminster School, looking northeast across rain-wet flagstones and the
gothic roof/pinnacles of Westminster Abbey toward the illuminated clock dial,
glowing belfry louvers, and spire of the Elizabeth Tower (Big Ben).

Zero image models, zero external textures, zero font dependencies.
Run: python3 paint_scene.py --out still.webp --crops-dir crops/
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


def alpha_composite(dst_rgb: np.ndarray, src_rgb: np.ndarray, src_a: np.ndarray) -> np.ndarray:
    a = np.clip(src_a, 0.0, 1.0)[..., None]
    return dst_rgb * (1.0 - a) + src_rgb * a


def screen_blend(dst_rgb: np.ndarray, add_rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
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
    warm_bounce_in_shadow: float = 0.022,
    num_dabs: int = 92,
    tooth_weight: float = 0.014,
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
    out[:, :, 0] += sh_w * warm_bounce_in_shadow * 1.15
    out[:, :, 1] += sh_w * warm_bounce_in_shadow * 0.72
    out[:, :, 2] += sh_w * warm_bounce_in_shadow * 0.32
    return np.clip(out, 0.0, 1.0)


def voronoi_mortar_field(
    uv_h: int,
    uv_w: int,
    span_u: float,
    span_v: float,
    cell_size_m: float = 0.65,
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


class Camera3D:
    def __init__(self, eye, target, up, fov_deg: float, width: int = 2048, height: int = 1152):
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
        self.fwd, self.right, self.down = fwd, right, down
        self.R = np.vstack([right, down, fwd])
        self.f = (self.width * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
        self.cx = self.width * 0.5
        self.cy = self.height * 0.5
        self.K = np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        yy, xx = np.mgrid[0 : self.height, 0 : self.width].astype(np.float32)
        dx_cam = (xx - self.cx) / self.f
        dy_cam = (yy - self.cy) / self.f
        self.ray_len_scale = np.sqrt(dx_cam * dx_cam + dy_cam * dy_cam + 1.0).astype(np.float32)
        rt = self.R.T.astype(np.float32)
        self.ray_world_x = rt[0, 0] * dx_cam + rt[0, 1] * dy_cam + rt[0, 2]
        self.ray_world_y = rt[1, 0] * dx_cam + rt[1, 1] * dy_cam + rt[1, 2]
        self.ray_world_z = rt[2, 0] * dx_cam + rt[2, 1] * dy_cam + rt[2, 2]
        self.horizon_y = float(self.project_direction(np.array([0.0, 1.0, 0.0]))[1])

    def project_points(self, pts_world: np.ndarray):
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

    def plane_homography(self, p0, e1, e2, span_u: float, span_v: float, uv_w: int, uv_h: int) -> np.ndarray:
        e1_u = np.asarray(e1, dtype=np.float64); e1_u /= np.linalg.norm(e1_u)
        e2_u = np.asarray(e2, dtype=np.float64); e2_u /= np.linalg.norm(e2_u)
        basis = np.column_stack([self.R @ e1_u, self.R @ e2_u, self.R @ (p0 - self.eye)])
        h_uv = np.array([[span_u / float(uv_w), 0.0, 0.0], [0.0, span_v / float(uv_h), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return self.K @ basis @ h_uv

    def ray_plane_hit(self, p0, normal):
        n = np.asarray(normal, dtype=np.float64); n /= np.linalg.norm(n)
        num = float(np.dot(p0 - self.eye, n))
        den = self.ray_world_x * float(n[0]) + self.ray_world_y * float(n[1]) + self.ray_world_z * float(n[2])
        safe_den = np.where(np.abs(den) < 1e-6, 1e-6, den)
        z_cam = (num / safe_den).astype(np.float32)
        dist = z_cam * self.ray_len_scale
        hx = float(self.eye[0]) + z_cam * self.ray_world_x
        hy = float(self.eye[1]) + z_cam * self.ray_world_y
        hz = float(self.eye[2]) + z_cam * self.ray_world_z
        return z_cam, dist, hx, hy, hz

    def warp_plane_panel(self, panel_rgb, panel_alpha, p0, e1, e2, span_u, span_v, fog_color, fog_start=8.0, fog_density=0.012):
        uv_h, uv_w = panel_alpha.shape
        h_mat = self.plane_homography(p0, e1, e2, span_u, span_v, uv_w, uv_h)
        premul = np.empty((uv_h, uv_w, 4), dtype=np.float32)
        a_clamped = np.clip(panel_alpha.astype(np.float32), 0.0, 1.0)
        premul[:, :, :3] = panel_rgb.astype(np.float32) * a_clamped[:, :, None]
        premul[:, :, 3] = a_clamped
        warped = cv2.warpPerspective(
            premul, h_mat, (self.width, self.height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0, 0.0)
        )
        e1_u = np.asarray(e1, dtype=np.float64); e1_u /= np.linalg.norm(e1_u)
        e2_u = np.asarray(e2, dtype=np.float64); e2_u /= np.linalg.norm(e2_u)
        normal = np.cross(e1_u, e2_u)
        z_cam, dist, hx, hy, hz = self.ray_plane_hit(p0, normal)
        dx, dy, dz = hx - float(p0[0]), hy - float(p0[1]), hz - float(p0[2])
        u_m = dx * float(e1_u[0]) + dy * float(e1_u[1]) + dz * float(e1_u[2])
        v_m = dx * float(e2_u[0]) + dy * float(e2_u[1]) + dz * float(e2_u[2])
        valid = ((z_cam > 0.05) & (u_m >= -0.01) & (u_m <= span_u + 0.01) & (v_m >= -0.01) & (v_m <= span_v + 0.01)).astype(np.float32)
        alpha = np.clip(warped[:, :, 3] * valid, 0.0, 1.0)
        safe_a = np.maximum(alpha, 1e-5)
        rgb = np.clip(warped[:, :, :3] / safe_a[:, :, None], 0.0, 1.0)
        fog_t = np.clip(1.0 - np.exp(-np.maximum(0.0, dist - float(fog_start)) * float(fog_density)), 0.0, 0.55)[:, :, None]
        rgb = rgb * (1.0 - fog_t) + fog_color[None, None, :] * fog_t
        return rgb, alpha, z_cam


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
        self.width, self.height = int(width), int(height)
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

    def fill_poly_linear(self, pts_2d, p_start, p_end, stops) -> None:
        if len(pts_2d) < 3:
            return
        ctx = self.ctx
        pat = cairo.LinearGradient(p_start[0], p_start[1], p_end[0], p_end[1])
        for off, rgba in stops:
            pat.add_color_stop_rgba(off, *rgba)
        ctx.new_path()
        ctx.move_to(float(pts_2d[0, 0]), float(pts_2d[0, 1]))
        for x, y in pts_2d[1:]:
            ctx.line_to(float(x), float(y))
        ctx.close_path()
        ctx.set_source(pat)
        ctx.fill()

    def fill_poly_radial(self, pts_2d, c0, c1, stops) -> None:
        if len(pts_2d) < 3:
            return
        ctx = self.ctx
        pat = cairo.RadialGradient(c0[0], c0[1], c0[2], c1[0], c1[1], c1[2])
        for off, rgba in stops:
            pat.add_color_stop_rgba(off, *rgba)
        ctx.new_path()
        ctx.move_to(float(pts_2d[0, 0]), float(pts_2d[0, 1]))
        for x, y in pts_2d[1:]:
            ctx.line_to(float(x), float(y))
        ctx.close_path()
        ctx.set_source(pat)
        ctx.fill()

    def stroke_polyline_ink(self, pts_2d, width_px: float, rgba, closed: bool = False) -> None:
        """Crisp architectural ink stroke along straight polygonal vertices."""
        if len(pts_2d) < 2:
            return
        ctx = self.ctx
        ctx.set_source_rgba(*rgba)
        ctx.set_line_width(width_px)
        ctx.new_path()
        ctx.move_to(float(pts_2d[0, 0]), float(pts_2d[0, 1]))
        for x, y in pts_2d[1:]:
            ctx.line_to(float(x), float(y))
        if closed:
            ctx.close_path()
        ctx.stroke()

    def stroke_catmull_rom(
        self, pts_2d, z_depths, base_width: float, rgba, seed: int = 1, closed: bool = False,
        fade_dist: float = 35.0, taper_ends: float = 0.25, tremor: float = 0.20, samples_per_seg: int = 8
    ) -> None:
        pts = np.asarray(pts_2d, dtype=np.float64)
        if len(pts) < 2:
            return
        curve = eval_catmull_rom(pts, samples_per_seg=samples_per_seg, closed=closed)
        n = len(curve)
        if np.ndim(z_depths) == 0:
            zs = np.full(n, float(z_depths), dtype=np.float64)
        else:
            z_arr = np.asarray(z_depths, dtype=np.float64)
            zs = np.interp(np.linspace(0.0, len(z_arr) - 1, n), np.arange(len(z_arr)), z_arr)
        d = np.gradient(curve, axis=0)
        dlen = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        norm = np.column_stack([-d[:, 1] / dlen[:, 0], d[:, 0] / dlen[:, 0]])
        rng = np.random.default_rng(seed)
        s = np.linspace(0.0, 1.0, n)
        ph1, ph2 = rng.uniform(0.0, 2.0 * math.pi, size=2)
        wobble = tremor * (0.65 * np.sin(2.0 * math.pi * 3.0 * s + ph1) + 0.35 * np.sin(2.0 * math.pi * 7.0 * s + ph2))
        if closed:
            wobble *= np.sin(math.pi * s)
        curve = curve + norm * wobble[:, None]
        persp = np.clip(1.0 / (1.0 + np.maximum(zs, 0.05) / (fade_dist * 0.65)), 0.35, 1.0)
        if closed:
            taper = 0.85 + 0.15 * np.sin(2.0 * math.pi * s + ph1)
        else:
            taper = 0.25 + 0.75 * smoothstep(0.0, max(taper_ends, 0.05), s) * (1.0 - smoothstep(1.0 - max(taper_ends, 0.05), 1.0, s))
        half_w = base_width * persp * taper * 0.5
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
        rgb = np.clip(arr[:, :, :3] / safe_a[:, :, None], 0.0, 1.0)
        return rgb, alpha


def make_disc_kernel(radius: float = 8.0) -> np.ndarray:
    rad = max(1, int(round(float(radius))))
    ky, kx = np.mgrid[-rad : rad + 1, -rad : rad + 1].astype(np.float32)
    kr = np.sqrt(kx * kx + ky * ky) / max(float(radius), 1.0)
    rim = 0.86 + 0.24 * np.exp(-((kr - 0.84) ** 2) / 0.016)
    disc = np.clip((1.04 - kr) / 0.16, 0.0, 1.0) * rim
    return (disc / max(float(np.sum(disc)), 1e-6)).astype(np.float32)


def apply_foreground_disc_bokeh(canvas, fg_rgb, fg_alpha, radius=8.0, chroma_shift_px=1.0):
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
    bright = np.clip((lum - 0.68) / 0.32, 0.0, 1.0) ** 1.25
    if specular_glints is not None:
        bright = np.clip(bright + specular_glints * 0.92, 0.0, 1.4)
    hal_tint = np.array([1.00, 0.88, 0.68], dtype=np.float32)
    h_src = bright[:, :, None] * hal_tint[None, None, :]
    halation = (
        0.19 * cv2.GaussianBlur(h_src, (0, 0), 6.5)
        + 0.18 * cv2.GaussianBlur(h_src, (0, 0), 24.0)
        + 0.15 * cv2.GaussianBlur(h_src, (0, 0), 72.0)
    )
    img = screen_blend(img, halation, strength=1.0)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / (w * 0.5)
    ny = (yy - h * 0.5) / (h * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.00080
    img[:, :, 0] = cv2.remap(img[:, :, 0], (xx + (xx - w * 0.5) * r2 * ca_scale).astype(np.float32), (yy + (yy - h * 0.5) * r2 * ca_scale).astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    img[:, :, 2] = cv2.remap(img[:, :, 2], (xx - (xx - w * 0.5) * r2 * ca_scale).astype(np.float32), (yy - (yy - h * 0.5) * r2 * ca_scale).astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    img *= (1.0 - 0.14 * np.clip(r2 - 0.18, 0.0, 2.0))[:, :, None]

    black_lift = np.array([0.020, 0.018, 0.026], dtype=np.float32)
    img = black_lift[None, None, :] + img * (1.0 - black_lift[None, None, :])
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    above = np.maximum(img - knee, 0.0)
    img = np.where(img > knee, knee + span * np.tanh(above / span), img)

    rng = np.random.default_rng(seed)
    lum2 = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    env = np.clip(1.12 - 0.95 * (np.clip(lum2, 0.0, 1.0) ** 1.55), 0.12, 1.12)
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

    row_shift = cv2.GaussianBlur(rng.normal(0.0, 0.068, size=h).astype(np.float32)[:, None], (1, 5), 0.9)[:, 0]
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


def save_crop_boxes(rgb_float, crops_dir: Path, centers_px, crop_w=600, crop_h=450) -> None:
    h, w, _ = rgb_float.shape
    crops_dir.mkdir(parents=True, exist_ok=True)
    cw, ch = min(crop_w, w), min(crop_h, h)
    for name, (cx, cy) in centers_px:
        x0 = int(np.clip(round(cx - cw * 0.5), 0, w - cw))
        y0 = int(np.clip(round(cy - ch * 0.5), 0, h - ch))
        save_image(rgb_float[y0 : y0 + ch, x0 : x0 + cw], crops_dir / name)


def render_scene(width: int = 2048, height: int = 1152) -> tuple[np.ndarray, list[tuple[str, tuple[float, float]]]]:
    # Camera placed in Little Dean's Yard looking NE:
    # - Wet flagstones occupy bottom ~30% (horizon_y ~ 0.67)
    # - Abbey nave & pinnacles occupy middle band (0.45..0.68)
    # - Entire Elizabeth Tower (clock, glowing belfry louvers, spire & Ayrton light) sits cleanly in top half at ~61.5% X!
    cam = Camera3D(
        eye=np.array([-1.5, -2.0, 1.65]),
        target=np.array([1.2, 26.0, 8.2]),
        up=np.array([0.0, 0.0, 1.0]),
        fov_deg=56.0,
        width=width,
        height=height,
    )
    tooth = make_paper_tooth(height, width, seed=1858)
    fog_col = hex_rgb("#263547")

    # 1. Blue-hour sky with warm terracotta dusk horizon glow behind Elizabeth Tower
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    ny = yy / float(height)
    nx = xx / float(width)
    sky_top = hex_rgb("#111B29")
    sky_mid = hex_rgb("#294158")
    sky_hor = hex_rgb("#CC8761")
    sky_mist = hex_rgb("#5E7384")

    t_up = smoothstep(0.0, 0.62, ny)
    sky = sky_top[None, None, :] * (1.0 - t_up[:, :, None]) + sky_mid[None, None, :] * t_up[:, :, None]
    hor_band = np.exp(-((ny - 0.55) / 0.13) ** 2) * (0.50 + 0.50 * np.exp(-((nx - 0.62) / 0.34) ** 2))
    sky = sky * (1.0 - 0.55 * hor_band[:, :, None]) + sky_hor[None, None, :] * (0.55 * hor_band[:, :, None])
    cloud_f = smoothstep(0.15, 1.35, fbm_gouache(height, width, octaves=((140, 95, 0.6), (48, 34, 0.4)), seed=303))
    sky = sky * (1.0 - 0.14 * cloud_f[:, :, None]) + sky_mist[None, None, :] * (0.14 * cloud_f[:, :, None])
    sky = paint_isotropic_gouache_dabs(sky, np.ones((height, width), dtype=np.float32), tooth, dab_strength=0.022, num_dabs=64, seed=401)
    canvas = sky.copy()

    def proj2(pts):
        u, v, zc = cam.project_points(np.asarray(pts, dtype=np.float64))
        return np.column_stack([u, v]), zc

    # 2. Elizabeth Tower (placed at Y = 80 m, X = 12.2 m so ENTIRE spire + Ayrton light breathe with ~10% sky!)
    tx, ty = 12.2, 80.0
    tw_w = 4.4
    tower_l = CairoLayer2x(width, height)

    # Shaft (Z = 6.0 .. 22.8 m)
    shaft_pts, shaft_zc = proj2([
        [tx - tw_w, ty, 6.0], [tx + tw_w, ty, 6.0],
        [tx + tw_w, ty, 22.8], [tx - tw_w, ty, 22.8],
    ])
    tower_l.fill_poly_linear(
        shaft_pts, tuple(shaft_pts[0]), tuple(shaft_pts[1]),
        [(0.0, (0.21, 0.25, 0.32, 1.0)), (0.55, (0.34, 0.36, 0.40, 1.0)), (1.0, (0.48, 0.43, 0.38, 1.0))],
    )
    for frac in [-0.70, -0.35, 0.0, 0.35, 0.70]:
        rib_pts, rib_z = proj2([[tx + frac * tw_w, ty - 0.05, 8.0], [tx + frac * tw_w, ty - 0.05, 22.6]])
        tower_l.stroke_polyline_ink(rib_pts, 1.5, (0.12, 0.14, 0.20, 0.65))

    # Clock stage overhang (Z = 22.6 .. 29.2 m)
    cbox_pts, _ = proj2([
        [tx - tw_w * 1.06, ty - 0.1, 22.6], [tx + tw_w * 1.06, ty - 0.1, 22.6],
        [tx + tw_w * 1.06, ty - 0.1, 29.2], [tx - tw_w * 1.06, ty - 0.1, 29.2],
    ])
    tower_l.fill_poly_linear(
        cbox_pts, tuple(cbox_pts[0]), tuple(cbox_pts[1]),
        [(0.0, (0.19, 0.22, 0.29, 1.0)), (0.6, (0.29, 0.32, 0.37, 1.0)), (1.0, (0.42, 0.39, 0.35, 1.0))],
    )
    tower_l.stroke_polyline_ink(cbox_pts, 1.7, (0.09, 0.11, 0.16, 0.85), closed=True)

    # Belfry open-louver lantern stage (Z = 29.2 .. 33.4 m)
    belfry_pts, _ = proj2([
        [tx - tw_w * 0.92, ty - 0.1, 29.2], [tx + tw_w * 0.92, ty - 0.1, 29.2],
        [tx + tw_w * 0.92, ty - 0.1, 33.4], [tx - tw_w * 0.92, ty - 0.1, 33.4],
    ])
    tower_l.fill_poly(belfry_pts, (0.16, 0.19, 0.25, 1.0))
    for lx in [-0.62, -0.22, 0.22, 0.62]:
        lw_pts, _ = proj2([
            [tx + (lx - 0.13) * tw_w, ty - 0.12, 29.7], [tx + (lx + 0.13) * tw_w, ty - 0.12, 29.7],
            [tx + (lx + 0.11) * tw_w, ty - 0.12, 32.7], [tx + (lx - 0.11) * tw_w, ty - 0.12, 32.7],
        ])
        tower_l.fill_poly_linear(
            lw_pts, tuple(lw_pts[0]), tuple(lw_pts[2]),
            [(0.0, (0.85, 0.45, 0.26, 0.96)), (0.55, (0.98, 0.78, 0.42, 0.98)), (1.0, (0.62, 0.32, 0.22, 0.92))],
        )
        tower_l.stroke_polyline_ink(lw_pts, 1.2, (0.10, 0.11, 0.15, 0.8), closed=True)

    # Multi-tiered cast-iron spire + Ayrton Light (Z = 33.4 .. 43.2 m)
    spire_pts, _ = proj2([
        [tx - tw_w * 0.98, ty, 33.4], [tx + tw_w * 0.98, ty, 33.4],
        [tx + tw_w * 0.44, ty, 37.2], [tx + tw_w * 0.24, ty, 39.4],
        [tx + tw_w * 0.16, ty, 41.2], [tx, ty, 43.4],
        [tx - tw_w * 0.16, ty, 41.2], [tx - tw_w * 0.24, ty, 39.4],
        [tx - tw_w * 0.44, ty, 37.2],
    ])
    tower_l.fill_poly_linear(
        spire_pts, tuple(spire_pts[0]), tuple(spire_pts[5]),
        [(0.0, (0.14, 0.18, 0.24, 1.0)), (0.6, (0.24, 0.29, 0.36, 1.0)), (1.0, (0.11, 0.14, 0.19, 1.0))],
    )
    tower_l.stroke_polyline_ink(spire_pts, 1.8, (0.07, 0.09, 0.13, 0.90), closed=True)

    # Glowing Ayrton Light in upper lantern stage (Z = 38.4 m)
    ayr_u, ayr_v, _ = cam.project_points(np.array([tx, ty - 0.15, 38.4]))
    ayr_poly = np.array([[ayr_u - 10, ayr_v - 13], [ayr_u + 10, ayr_v - 13], [ayr_u + 10, ayr_v + 13], [ayr_u - 10, ayr_v + 13]])
    tower_l.fill_poly_radial(
        ayr_poly, (ayr_u, ayr_v, 1.5), (ayr_u, ayr_v, 14.0),
        [(0.0, (1.0, 0.97, 0.86, 1.0)), (0.45, (0.98, 0.78, 0.40, 0.98)), (1.0, (0.73, 0.39, 0.24, 0.0))],
    )

    # Illuminated Clock Dial (radius 2.35 m centered at Z = 25.9 m)
    cx_w, cy_w, cz_w = tx, ty - 0.18, 25.9
    r_dial = 2.35
    th = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    dial_pts, dial_zc = proj2(np.column_stack([cx_w + r_dial * np.cos(th), np.full_like(th, cy_w), cz_w + r_dial * np.sin(th)]))
    dial_c_u, dial_c_v, _ = cam.project_points(np.array([cx_w, cy_w, cz_w]))
    dial_r_px = float(np.mean(np.hypot(dial_pts[:, 0] - dial_c_u, dial_pts[:, 1] - dial_c_v)))
    bezel_pts, _ = proj2(np.column_stack([cx_w + (r_dial * 1.13) * np.cos(th), np.full_like(th, cy_w + 0.01), cz_w + (r_dial * 1.13) * np.sin(th)]))
    tower_l.fill_poly(bezel_pts, (0.12, 0.12, 0.13, 1.0))
    tower_l.fill_poly_radial(
        dial_pts, (dial_c_u, dial_c_v, 2.0), (dial_c_u, dial_c_v, dial_r_px * 1.05),
        [(0.0, (1.0, 0.98, 0.90, 1.0)), (0.72, (0.97, 0.87, 0.60, 1.0)), (1.0, (0.82, 0.58, 0.31, 1.0))],
    )
    inner_pts, inner_zc = proj2(np.column_stack([cx_w + (r_dial * 0.73) * np.cos(th), np.full_like(th, cy_w - 0.01), cz_w + (r_dial * 0.73) * np.sin(th)]))
    tower_l.stroke_polyline_ink(inner_pts, 1.8, (0.16, 0.14, 0.11, 0.88), closed=True)
    tower_l.stroke_polyline_ink(dial_pts, 2.4, (0.78, 0.62, 0.30, 0.95), closed=True)
    for h_idx in range(12):
        ang = math.radians(h_idx * 30.0)
        t_pts, _ = proj2([
            [cx_w + r_dial * 0.74 * math.sin(ang), cy_w - 0.02, cz_w + r_dial * 0.74 * math.cos(ang)],
            [cx_w + r_dial * 0.94 * math.sin(ang), cy_w - 0.02, cz_w + r_dial * 0.94 * math.cos(ang)],
        ])
        tower_l.stroke_polyline_ink(t_pts, 1.9, (0.14, 0.12, 0.10, 0.92))
    m_hand, _ = proj2([[cx_w, cy_w - 0.04, cz_w], [cx_w, cy_w - 0.04, cz_w + r_dial * 0.82]])
    h_hand, _ = proj2([[cx_w, cy_w - 0.04, cz_w], [cx_w - r_dial * 0.55 * math.sin(math.radians(30)), cy_w - 0.04, cz_w + r_dial * 0.55 * math.cos(math.radians(30))]])
    tower_l.stroke_polyline_ink(m_hand, 2.8, (0.09, 0.09, 0.11, 0.96))
    tower_l.stroke_polyline_ink(h_hand, 3.4, (0.09, 0.09, 0.11, 0.96))

    tw_rgb, tw_a = tower_l.finish()
    tw_rgb = paint_isotropic_gouache_dabs(tw_rgb, tw_a, tooth, dab_strength=0.028, num_dabs=48, seed=108)
    canvas = alpha_composite(canvas, tw_rgb, tw_a)

    # 3. Midground Westminster Abbey Nave Buttresses, Stone Parapet & Pinnacles (Y = 36 m, Z = 0..8.8 m)
    abbey_l = CairoLayer2x(width, height)
    ab_y = 36.0
    ab_verts = [
        [-18.0, ab_y, 0.0], [34.0, ab_y, 0.0],
        [34.0, ab_y, 6.2], [25.0, ab_y, 6.3], [24.5, ab_y, 8.6], [24.0, ab_y, 6.3],
        [17.5, ab_y, 6.5], [16.9, ab_y, 9.4], [16.3, ab_y, 6.5],
        [9.8, ab_y, 6.8], [9.2, ab_y, 10.1], [8.6, ab_y, 6.8],
        [2.2, ab_y, 7.0], [1.5, ab_y, 10.6], [0.8, ab_y, 7.0],
        [-5.2, ab_y, 7.2], [-6.0, ab_y, 10.8], [-6.8, ab_y, 7.2],
        [-18.0, ab_y, 7.5],
    ]
    ab_pts, _ = proj2(ab_verts)
    abbey_l.fill_poly_linear(
        ab_pts, tuple(ab_pts[0]), tuple(ab_pts[10]),
        [(0.0, (0.09, 0.12, 0.18, 1.0)), (0.65, (0.14, 0.17, 0.24, 1.0)), (1.0, (0.20, 0.23, 0.30, 1.0))],
    )
    # Stone parapet coping band + vertical buttress piers for architectural rhythm
    par_pts, _ = proj2([[-18.0, ab_y - 0.06, 6.6], [34.0, ab_y - 0.06, 5.8], [34.0, ab_y - 0.06, 6.3], [-18.0, ab_y - 0.06, 7.3]])
    abbey_l.fill_poly(par_pts, (0.19, 0.22, 0.29, 0.72))
    for bx in [-6.0, 1.5, 9.2, 16.9, 24.5]:
        b_pts, _ = proj2([[bx - 0.42, ab_y - 0.08, 0.0], [bx + 0.42, ab_y - 0.08, 0.0], [bx + 0.34, ab_y - 0.08, 7.0], [bx - 0.34, ab_y - 0.08, 7.0]])
        abbey_l.fill_poly_linear(b_pts, tuple(b_pts[0]), tuple(b_pts[1]), [(0.0, (0.07, 0.09, 0.14, 0.85)), (1.0, (0.18, 0.21, 0.28, 0.75))])
    for wx in [-2.2, 5.4, 13.0, 20.6]:
        w_pts, _ = proj2([
            [wx - 0.38, ab_y - 0.1, 2.8], [wx + 0.38, ab_y - 0.1, 2.8],
            [wx + 0.38, ab_y - 0.1, 5.1], [wx, ab_y - 0.1, 5.9], [wx - 0.38, ab_y - 0.1, 5.1],
        ])
        abbey_l.fill_poly_linear(w_pts, tuple(w_pts[0]), tuple(w_pts[3]), [(0.0, (0.84, 0.54, 0.30, 0.78)), (1.0, (0.48, 0.30, 0.21, 0.58))])
        abbey_l.stroke_polyline_ink(w_pts, 1.4, (0.06, 0.08, 0.12, 0.90), closed=True)
    abbey_l.stroke_polyline_ink(ab_pts[2:], 2.2, (0.06, 0.08, 0.12, 0.90))
    ab_rgb, ab_a = abbey_l.finish()
    ab_rgb = paint_isotropic_gouache_dabs(ab_rgb, ab_a, tooth, dab_strength=0.040, warm_bounce_in_shadow=0.025, num_dabs=80, seed=125)
    canvas = alpha_composite(canvas, ab_rgb, ab_a)

    # 4. Little Dean's Yard Wet Flagstones (Z = 0 plane via exact UV homography + Fresnel clock reflection!)
    fl_w, fl_h = 1024, 1024
    span_u, span_v = 32.0, 38.0
    p0_fl = np.array([-14.0, 0.2, 0.0])
    joint, st_tone = voronoi_mortar_field(fl_h, fl_w, span_u, span_v, cell_size_m=0.72, jitter=0.33, warp_amp_m=0.024, seed=215)
    fl_base = hex_rgb("#262E3B")[None, None, :] + 0.045 * st_tone[:, :, None]
    fl_base *= (0.60 + 0.40 * smoothstep(0.012, 0.046, joint))[:, :, None]
    vv_f, uu_f = np.mgrid[0:fl_h, 0:fl_w].astype(np.float32)
    xw_f = -14.0 + (uu_f / (fl_w - 1)) * span_u
    yw_f = 0.2 + (vv_f / (fl_h - 1)) * span_v
    lantern_pool = np.exp(-(((xw_f + 3.0) / 3.4) ** 2 + ((yw_f - 6.8) / 4.2) ** 2))
    fl_base += lantern_pool[:, :, None] * hex_rgb("#E39154")[None, None, :] * 0.48
    fl_rgb, fl_a, _ = cam.warp_plane_panel(
        np.clip(fl_base, 0.0, 1.0), np.ones((fl_h, fl_w), dtype=np.float32),
        p0_fl, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        span_u, span_v, fog_col, fog_start=10.0, fog_density=0.016,
    )
    fl_rgb = paint_isotropic_gouache_dabs(fl_rgb, fl_a, tooth, dab_strength=0.044, warm_bounce_in_shadow=0.032, num_dabs=96, seed=233)

    # Puddle Fresnel reflection across wet courtyard flagstones
    puddle_mask = smoothstep(-0.1, 0.65, fbm_gouache(height, width, octaves=((115, 70, 0.6), (36, 24, 0.4)), seed=277)) * fl_a
    y_src = np.clip(2.0 * cam.horizon_y - yy, 0.0, height - 1.0)
    rip = 2.6 * np.sin(0.24 * yy + 0.03 * xx) + 1.4 * np.sin(0.51 * yy - 0.045 * xx)
    y_rip = np.clip(y_src + rip * puddle_mask, 0.0, height - 1.0).astype(np.float32)
    x_rip = np.clip(xx + 1.3 * np.cos(0.19 * yy + 0.02 * xx) * puddle_mask, 0.0, width - 1.0).astype(np.float32)
    refl = cv2.remap(canvas, x_rip, y_rip, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # Mirrored golden clock dial + sky reflection column on wet stone
    clock_refl_streak = np.exp(-((xx - dial_c_u) / 52.0) ** 2) * smoothstep(cam.horizon_y - 10, height * 0.96, yy)
    refl += clock_refl_streak[:, :, None] * hex_rgb("#F4C872")[None, None, :] * 0.58
    dist_t = np.clip((yy - cam.horizon_y) / max(1.0, height - cam.horizon_y), 0.0, 1.0)
    fresnel = (0.22 + 0.58 * ((1.0 - dist_t) ** 1.4)) * puddle_mask
    fl_final = fl_rgb * (1.0 - fresnel[:, :, None]) + np.clip(refl, 0.0, 1.0) * fresnel[:, :, None]
    canvas = alpha_composite(canvas, fl_final, fl_a)

    # 5. Left Courtyard Stone Wall, Tudor Mullioned Window & Wrought-Iron Wall Lantern (X = -4.5 m)
    wl_w, wl_h = 768, 768
    w_span_u, w_span_v = 18.0, 9.0
    p0_wl = np.array([-4.5, 0.6, 0.0])
    w_joint, w_tone = voronoi_mortar_field(wl_h, wl_w, w_span_u, w_span_v, cell_size_m=0.48, jitter=0.22, warp_amp_m=0.015, seed=312)
    wl_rgb_uv = hex_rgb("#353944")[None, None, :] + 0.04 * w_tone[:, :, None]
    wl_rgb_uv *= (0.66 + 0.34 * smoothstep(0.010, 0.038, w_joint))[:, :, None]
    vw_f, uw_f = np.mgrid[0:wl_h, 0:wl_w].astype(np.float32)
    um_w = (uw_f / (wl_w - 1)) * w_span_u
    vm_w = (vw_f / (wl_h - 1)) * w_span_v
    l_halo = np.exp(-(((um_w - 5.8) / 2.6) ** 2 + ((vm_w - 3.3) / 2.2) ** 2))
    wl_rgb_uv += l_halo[:, :, None] * hex_rgb("#E89254")[None, None, :] * 0.62
    # Warm study window with stone surround & leaded diamond-pane grid
    win_box = ((um_w >= 10.2) & (um_w <= 12.6) & (vm_w >= 2.2) & (vm_w <= 5.2)).astype(np.float32)
    mullion = (np.abs(um_w - 11.4) < 0.09) | (np.abs(vm_w - 3.7) < 0.08) | (np.abs(um_w - 10.8) < 0.03) | (np.abs(um_w - 12.0) < 0.03)
    win_glow = win_box * (~mullion).astype(np.float32)
    wl_rgb_uv = wl_rgb_uv * (1.0 - 0.88 * win_glow[:, :, None]) + hex_rgb("#F2BA6C")[None, None, :] * (0.88 * win_glow[:, :, None])
    wl_rgb, wl_a, _ = cam.warp_plane_panel(
        np.clip(wl_rgb_uv, 0.0, 1.0), np.ones((wl_h, wl_w), dtype=np.float32),
        p0_wl, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
        w_span_u, w_span_v, fog_col, fog_start=5.0, fog_density=0.020,
    )
    wl_rgb = paint_isotropic_gouache_dabs(wl_rgb, wl_a, tooth, dab_strength=0.044, warm_bounce_in_shadow=0.025, num_dabs=80, seed=341)
    canvas = alpha_composite(canvas, wl_rgb, wl_a)

    # Wrought-iron lantern housing + stone cornice on left wall
    wall_ink = CairoLayer2x(width, height)
    lu, lv, _ = cam.project_points(np.array([-4.25, 6.4, 3.3]))
    lantern_poly = np.array([[lu - 24, lv - 28], [lu + 24, lv - 28], [lu + 16, lv + 24], [lu - 16, lv + 24]])
    wall_ink.fill_poly_radial(
        lantern_poly, (lu, lv, 3.0), (lu, lv, 28.0),
        [(0.0, (1.0, 0.97, 0.84, 1.0)), (0.50, (0.96, 0.70, 0.36, 0.96)), (1.0, (0.55, 0.28, 0.18, 0.0))],
    )
    cap_pts = np.array([[lu - 26, lv - 28], [lu, lv - 44], [lu + 26, lv - 28]])
    wall_ink.fill_poly(cap_pts, (0.08, 0.08, 0.09, 0.95))
    wall_ink.stroke_polyline_ink(lantern_poly, 2.2, (0.07, 0.07, 0.08, 0.92), closed=True)
    # Attach wrought-iron bracket strictly to the cap apex (never crossing inside glass!)
    wall_attach_u, wall_attach_v, _ = cam.project_points(np.array([-4.5, 6.4, 3.85]))
    bracket_pts = np.array([[wall_attach_u, wall_attach_v], [lu - 14.0, lv - 58.0], [lu, lv - 44.0]])
    wall_ink.stroke_catmull_rom(bracket_pts, 6.0, 4.2, (0.07, 0.07, 0.08, 0.96), seed=355, tremor=0.05)
    wi_rgb, wi_a = wall_ink.finish()
    canvas = alpha_composite(canvas, wi_rgb, wi_a)

    # 6. Out-of-focus foreground winter plane-tree boughs on right edge (disc bokeh)
    fg_l = CairoLayer2x(width, height)
    boughs = [
        np.array([[width * 1.02, height * 0.22], [width * 0.92, height * 0.30], [width * 0.85, height * 0.27], [width * 0.79, height * 0.34]]),
        np.array([[width * 0.93, height * 0.30], [width * 0.88, height * 0.16], [width * 0.83, height * 0.09]]),
        np.array([[width * 1.03, height * 0.84], [width * 0.93, height * 0.91], [width * 0.85, height * 1.02]]),
    ]
    for bi, bpts in enumerate(boughs):
        fg_l.stroke_catmull_rom(bpts, 1.5, 22.0 - bi * 4.0, (0.07, 0.08, 0.11, 0.96), seed=520 + bi, tremor=1.4, taper_ends=0.35)
    fg_rgb, fg_a = fg_l.finish()
    canvas = apply_foreground_disc_bokeh(canvas, fg_rgb, fg_a, radius=8.2, chroma_shift_px=1.1)

    # 7. Specular glints mask for 35mm halation
    glints = np.zeros((height, width), dtype=np.float32)
    glints += np.exp(-(((xx - dial_c_u) ** 2 + (yy - dial_c_v) ** 2) / (dial_r_px * 1.15) ** 2)) * 0.88
    glints += np.exp(-(((xx - ayr_u) ** 2 + (yy - ayr_v) ** 2) / 16.0 ** 2)) * 0.95
    glints += np.exp(-(((xx - lu) ** 2 + (yy - lv) ** 2) / 26.0 ** 2)) * 0.92
    final = finish_35mm_optics(canvas, specular_glints=np.clip(glints, 0.0, 1.2), seed=1859)

    crops = [
        ("crop_1_tower_clock.png", (dial_c_u, dial_c_v - 90.0)),
        ("crop_2_courtyard_lantern.png", (lu + 90.0, lv + 20.0)),
        ("crop_3_wet_flagstones.png", (dial_c_u - 40.0, height * 0.88)),
    ]
    return final, crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Paint Little Dean's Yard & Elizabeth Tower dusk still.")
    parser.add_argument("--out", required=True, type=Path, help="Output image path (.webp or .png)")
    parser.add_argument("--crops-dir", type=Path, default=None, help="Optional directory for 600x450 1:1 audit crops")
    args = parser.parse_args()

    final, crops = render_scene(2048, 1152)
    save_image(final, args.out)
    if args.crops_dir is not None:
        save_crop_boxes(final, args.crops_dir, crops)


if __name__ == "__main__":
    main()
