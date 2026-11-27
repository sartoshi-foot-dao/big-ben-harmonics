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
FPS = 30
DURATION_S = 56.00

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


def _end_taper(n_len: int, fade_s: float = 0.25) -> np.ndarray:
    fade_n = min(n_len, int(round(fade_s * SR)))
    win = np.ones(n_len, dtype=np.float64)
    if fade_n > 1:
        win[-fade_n:] = 0.5 * (1.0 + np.cos(math.pi * np.arange(fade_n) / fade_n))
    return win


def synth_pure(
    buf: np.ndarray,
    t0_s: float,
    freq_hz: float,
    gain: float,
    dur_s: float = 2.5,
    t60_s: float = 3.2,
    attack_s: float = 0.004,
) -> None:
    i0 = int(round(t0_s * SR))
    if i0 >= len(buf):
        return
    n_len = min(len(buf) - i0, int(round(dur_s * SR)))
    t = np.arange(n_len, dtype=np.float64) / SR
    attack = 1.0 - np.exp(-t / attack_s)
    tau = t60_s / 6.907755
    env = attack * np.exp(-t / tau) * _end_taper(n_len, fade_s=0.22)
    sig = env * np.sin(2.0 * math.pi * freq_hz * t)
    buf[i0 : i0 + n_len] += gain * sig


def synth_modes(
    buf: np.ndarray,
    t0_s: float,
    modes: list[tuple[float, float, float]],
    gain: float,
    abbey_filter: bool = False,
    dur_cap_s: float = 10.0,
) -> None:
    i0 = int(round(t0_s * SR))
    if i0 >= len(buf):
        return
    n_len = min(len(buf) - i0, int(round(dur_cap_s * SR)))
    t = np.arange(n_len, dtype=np.float64) / SR
    sig = np.zeros(n_len, dtype=np.float64)
    attack = 1.0 - np.exp(-t / 0.004)
    for freq, t60, rel_db in modes:
        amp = 10.0 ** (rel_db / 20.0)
        if abbey_filter:
            loss_db = 4.2 * (freq / 500.0) ** 1.35 + (12.0 if freq > 900.0 else 0.0)
            amp *= 10.0 ** (-loss_db / 20.0)
        tau = t60 / 6.907755
        env = attack * np.exp(-t / tau)
        sig += amp * env * np.sin(2.0 * math.pi * freq * t)
    n_tr = min(n_len, int(0.03 * SR))
    rng = np.random.default_rng(int(round(t0_s * 1000)) + 17)
    clang = rng.normal(0.0, 0.14 if not abbey_filter else 0.02, n_tr) * np.exp(-t[:n_tr] / 0.006)
    sig[:n_tr] += clang
    sig *= _end_taper(n_len, fade_s=0.35)
    buf[i0 : i0 + n_len] += gain * sig


def build_master(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(round(DURATION_S * SR))
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    events = [
        {"t_s": 0.70, "frame": int(round(0.70 * FPS)), "act": 1, "kind": "pure_string", "label": "Pure 335.5 Hz mode (single pitch)"},
        {"t_s": 3.90, "frame": int(round(3.90 * FPS)), "act": 1, "kind": "belfry_full", "label": "1969 BBC belfry strike (five voices)"},
        {"t_s": 13.10, "frame": int(round(13.10 * FPS)), "act": 2, "kind": "steady_single", "label": "Single unsplit 335.4 Hz mode"},
        {"t_s": 16.10, "frame": int(round(16.10 * FPS)), "act": 2, "kind": "solo_doublet", "label": "Split 332.74 + 335.44 Hz doublet (0.370 s swell)"},
        {"t_s": 25.20, "frame": int(round(25.20 * FPS)), "act": 3, "kind": "solo_a5", "label": "883.34 Hz A5 super-third (T60 = 7.96 s)"},
        {"t_s": 27.80, "frame": int(round(27.80 * FPS)), "act": 3, "kind": "solo_hum_fifth", "label": "200.26 Hz hum + 499.67 Hz B4 (T60 = 29.0 s)"},
        {"t_s": 30.50, "frame": int(round(30.50 * FPS)), "act": 3, "kind": "chord_shift_strike", "label": "14-second chord rotation (A4 clash to B4 + G3)"},
        {"t_s": 37.70, "frame": int(round(37.70 * FPS)), "act": 4, "kind": "belfry_close", "label": "10 m close belfry strike (unfiltered)"},
        {"t_s": 41.40, "frame": int(round(41.40 * FPS)), "act": 4, "kind": "abbey_yard", "label": "354.5 m Little Dean's Yard behind Abbey (-49.5 dB @ 5 kHz)"},
        {"t_s": 48.70, "frame": int(round(48.70 * FPS)), "act": 5, "kind": "coda_1890_lock", "label": "1890 cylinder speed recovery & night coda"},
    ]

    # Act 1: Pure note -> Full Big Ben strike (end cleanly before 11.35 s transition)
    synth_pure(left, 0.70, 335.51, gain=0.24, dur_s=2.8, t60_s=3.4)
    synth_pure(right, 0.7002, 335.51, gain=0.24, dur_s=2.8, t60_s=3.4)

    synth_modes(left, 3.90, BB_MODES, 0.30, abbey_filter=False, dur_cap_s=7.3)
    synth_modes(right, 3.9003, BB_MODES, 0.30, abbey_filter=False, dur_cap_s=7.3)

    # Act 2: Steady single ring -> Split doublet (332.735 @ -5.1 dB vs 335.440 @ 0 dB -> 0.557 ratio)
    synth_pure(left, 13.10, 335.440, gain=0.24, dur_s=2.6, t60_s=5.0)
    synth_pure(right, 13.1002, 335.440, gain=0.24, dur_s=2.6, t60_s=5.0)

    doublet_modes = [
        (332.735, 12.16, -5.08),
        (335.440, 13.58, 0.0),
    ]
    synth_modes(left, 16.10, doublet_modes, 0.34, abbey_filter=False, dur_cap_s=7.2)
    synth_modes(right, 16.1002, doublet_modes, 0.34, abbey_filter=False, dur_cap_s=7.2)

    # Act 3: 883 Hz fast fade vs 200 Hz + 500 Hz hum sustain -> Full 14s decay strike
    synth_modes(left, 25.20, [(883.34, 7.96, -0.4)], 0.32, abbey_filter=False, dur_cap_s=2.3)
    synth_modes(right, 25.2002, [(883.34, 7.96, -0.4)], 0.32, abbey_filter=False, dur_cap_s=2.3)

    synth_modes(left, 27.80, [(200.26, 29.03, -4.0), (499.666, 17.33, 0.0)], 0.30, abbey_filter=False, dur_cap_s=2.4)
    synth_modes(right, 27.8003, [(200.26, 29.03, -4.0), (499.666, 17.33, 0.0)], 0.30, abbey_filter=False, dur_cap_s=2.4)

    synth_modes(left, 30.50, BB_MODES, 0.30, abbey_filter=False, dur_cap_s=5.4)
    synth_modes(right, 30.5003, BB_MODES, 0.30, abbey_filter=False, dur_cap_s=5.4)

    # Act 4: Close 10m belfry vs 354.5m Abbey shadow
    synth_modes(left, 37.70, BB_MODES, 0.28, abbey_filter=False, dur_cap_s=3.3)
    synth_modes(right, 37.7002, BB_MODES, 0.28, abbey_filter=False, dur_cap_s=3.3)

    synth_modes(left, 41.40, BB_MODES, 0.44, abbey_filter=True, dur_cap_s=5.6)
    synth_modes(right, 41.4004, BB_MODES, 0.44, abbey_filter=True, dur_cap_s=5.6)

    # Act 5: 1890 lock & Little Dean's Yard coda
    synth_modes(left, 48.70, BB_MODES, 0.40, abbey_filter=True, dur_cap_s=7.2)
    synth_modes(right, 48.7004, BB_MODES, 0.40, abbey_filter=True, dur_cap_s=7.2)

    fade_n = int(0.9 * SR)
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
    events_path.write_text(json.dumps({"sr": SR, "fps": FPS, "duration_s": DURATION_S, "events": events}, indent=2) + "\n")
    print(f"Synthesized {wav_path} ({DURATION_S:.2f} s, peak={float(np.max(np.abs(stereo))):.3f})")
    return wav_path, events_path


if __name__ == "__main__":
    build_master(Path(__file__).resolve().parent / "build")
