#!/usr/bin/env python3
"""
Deterministic 48 kHz modal bell synthesizer driven by measured 1969 BBC Big Ben
and Whitechapel Quarter Bell eigenmodes. Emits WAV master + sample-accurate events.json.
"""

from __future__ import annotations

import json
import math
import types
import wave
from pathlib import Path

import numpy as np
try:
    import soundfile as sf
except ModuleNotFoundError:
    def _write_wav(path, stereo, sr, subtype=None):
        pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(pcm16.tobytes())
    sf = types.SimpleNamespace(write=_write_wav)

SR = 48000
DURATION_S = 33.60

# Measured 1969 BBC WAV modes: (freq_hz, t60_s, rel_db)
BB_MODES = [
    (198.48, 29.03, -11.0),
    (200.26, 29.03, -6.2),
    (332.735, 14.17, -4.2),
    (335.440, 14.17, -1.1),
    (494.828, 17.33, -5.0),
    (499.666, 17.33, 0.0),
    (677.60, 9.28, -3.2),
    (883.34, 7.96, -0.4),
    (1088.24, 7.11, -8.5),
    (1315.45, 8.53, -12.8),
    (1748.04, 5.44, -16.4),
]

QUARTER_BELLS = {
    "G#4": [(869.75, 3.0, 0.0), (513.66, 3.8, -6.2), (1297.93, 2.1, -9.5), (374.93, 3.4, -6.2)],
    "F#4": [(770.97, 3.0, 0.0), (454.50, 3.8, -1.8), (1157.19, 2.1, -3.9), (363.69, 3.4, -10.4)],
    "E4": [(678.99, 3.2, 0.0), (402.60, 4.0, -8.5), (1015.97, 2.1, 2.0), (312.03, 3.6, -11.0)],
    "B3": [(505.14, 4.4, -5.0), (298.72, 5.2, 0.0), (627.78, 2.5, -10.5), (222.21, 5.0, -1.5)],
}


def synth_modes(buf: np.ndarray, t0_s: float, modes: list[tuple[float, float, float]], gain: float,
                abbey_filter: bool = False, dur_cap_s: float = 10.0) -> None:
    i0 = int(round(t0_s * SR))
    if i0 >= len(buf):
        return
    n_len = min(len(buf) - i0, int(dur_cap_s * SR))
    t = np.arange(n_len, dtype=np.float64) / SR
    sig = np.zeros(n_len, dtype=np.float64)
    # 4 ms smooth hammer impact attack to avoid digital click
    attack = 1.0 - np.exp(-t / 0.004)
    for freq, t60, rel_db in modes:
        amp = (10.0 ** (rel_db / 20.0))
        if abbey_filter:
            # Maekawa + winter air path attenuation at 354.5 m
            loss_db = 4.2 * (freq / 500.0) ** 1.35 + (12.0 if freq > 900.0 else 0.0)
            amp *= 10.0 ** (-loss_db / 20.0)
        tau = t60 / 6.907755
        env = attack * np.exp(-t / tau)
        sig += amp * env * np.sin(2.0 * math.pi * freq * t)
    # Add short bronze impact transient (filtered scaled noise burst 15 ms)
    n_tr = min(n_len, int(0.03 * SR))
    rng = np.random.default_rng(int(round(t0_s * 1000)) + 17)
    clang = rng.normal(0.0, 0.14 if not abbey_filter else 0.02, n_tr) * np.exp(-t[:n_tr] / 0.006)
    sig[:n_tr] += clang
    buf[i0 : i0 + n_len] += gain * sig


def build_master(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(round(DURATION_S * SR))
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    events = [
        {"t_s": 0.50, "frame": 15, "act": 1, "kind": "belfry_full", "label": "Big Ben Full Belfry Strike"},
        {"t_s": 6.80, "frame": 204, "act": 2, "kind": "solo_a5", "label": "883 Hz A5 -> A4 Attack Clash"},
        {"t_s": 8.40, "frame": 252, "act": 2, "kind": "solo_doublet", "label": "332.7 / 335.4 Hz E4 Doublet (2.71 Hz)"},
        {"t_s": 10.60, "frame": 318, "act": 2, "kind": "solo_hum_fifth", "label": "200 Hz G3 Hum + 500 Hz B4 Fifth"},
        {"t_s": 13.80, "frame": 414, "act": 3, "kind": "belfry_cmp", "label": "10 m Belfry Bright Strike"},
        {"t_s": 16.50, "frame": 495, "act": 3, "kind": "abbey_shadow", "label": "354.5 m Little Dean's Yard (Abbey Shadow)"},
        {"t_s": 20.80, "frame": 624, "act": 4, "kind": "q_gs4", "label": "Q1 Bell 1 (G#4 870 Hz)"},
        {"t_s": 21.60, "frame": 648, "act": 4, "kind": "q_fs4", "label": "Q1 Bell 2 (F#4 771 Hz)"},
        {"t_s": 22.40, "frame": 672, "act": 4, "kind": "q_e4", "label": "Q1 Bell 3 (E4 679 Hz)"},
        {"t_s": 23.20, "frame": 696, "act": 4, "kind": "q_b3", "label": "Q1 Bell 4 (B3 505 Hz)"},
        {"t_s": 24.60, "frame": 738, "act": 4, "kind": "era_bell", "label": "Big Ben 1890->2024 Invariant Blow"},
        {"t_s": 27.80, "frame": 834, "act": 5, "kind": "coda_abbey", "label": "Little Dean's Yard Night Coda"},
    ]

    # Synthesize each event cleanly with slight binaural stereo dispersion
    synth_modes(left, 0.50, BB_MODES, 0.28, abbey_filter=False, dur_cap_s=5.9)
    synth_modes(right, 0.5003, BB_MODES, 0.28, abbey_filter=False, dur_cap_s=5.9)

    synth_modes(left, 6.80, [m for m in BB_MODES if abs(m[0] - 883.34) < 5], 0.34, dur_cap_s=2.0)
    synth_modes(right, 6.8002, [m for m in BB_MODES if abs(m[0] - 883.34) < 5], 0.34, dur_cap_s=2.0)

    synth_modes(left, 8.40, [m for m in BB_MODES if 330 < m[0] < 340], 0.36, dur_cap_s=3.5)
    synth_modes(right, 8.4002, [m for m in BB_MODES if 330 < m[0] < 340], 0.36, dur_cap_s=3.5)

    synth_modes(left, 10.60, [m for m in BB_MODES if m[0] < 205 or 490 < m[0] < 505], 0.32, dur_cap_s=3.0)
    synth_modes(right, 10.6003, [m for m in BB_MODES if m[0] < 205 or 490 < m[0] < 505], 0.32, dur_cap_s=3.0)

    synth_modes(left, 13.80, BB_MODES, 0.26, abbey_filter=False, dur_cap_s=2.6)
    synth_modes(right, 13.8002, BB_MODES, 0.26, abbey_filter=False, dur_cap_s=2.6)

    synth_modes(left, 16.50, BB_MODES, 0.42, abbey_filter=True, dur_cap_s=4.2)
    synth_modes(right, 16.5004, BB_MODES, 0.42, abbey_filter=True, dur_cap_s=4.2)

    for ev_t, qk in [(20.80, "G#4"), (21.60, "F#4"), (22.40, "E4"), (23.20, "B3")]:
        synth_modes(left, ev_t, QUARTER_BELLS[qk], 0.24, abbey_filter=False, dur_cap_s=2.8)
        synth_modes(right, ev_t + 0.0002, QUARTER_BELLS[qk], 0.24, abbey_filter=False, dur_cap_s=2.8)

    synth_modes(left, 24.60, BB_MODES, 0.28, abbey_filter=False, dur_cap_s=3.1)
    synth_modes(right, 24.6003, BB_MODES, 0.28, abbey_filter=False, dur_cap_s=3.1)

    synth_modes(left, 27.80, BB_MODES, 0.44, abbey_filter=True, dur_cap_s=5.6)
    synth_modes(right, 27.8004, BB_MODES, 0.44, abbey_filter=True, dur_cap_s=5.6)

    # Fade last 0.7 s smoothly to zero
    fade_n = int(0.7 * SR)
    fade = 0.5 * (1.0 + np.cos(math.pi * np.arange(fade_n) / fade_n))
    left[-fade_n:] *= fade
    right[-fade_n:] *= fade

    stereo = np.column_stack([left, right])
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.90:
        stereo *= 0.88 / peak

    wav_path = out_dir / "master_audio.wav"
    events_path = out_dir / "events.json"
    sf.write(wav_path, stereo, SR, subtype="PCM_24")
    events_path.write_text(json.dumps({"sr": SR, "fps": 30, "duration_s": DURATION_S, "events": events}, indent=2) + "\n")
    print(f"Synthesized {wav_path} ({DURATION_S:.2f} s, peak={float(np.max(np.abs(stereo))):.3f})")
    return wav_path, events_path


if __name__ == "__main__":
    build_master(Path(__file__).resolve().parent / "build")
