#!/usr/bin/env python3
"""
Deterministic acoustic, modal, propagation, and multi-era forensic analysis
of Big Ben and the Westminster Quarter Bells (1890-2024).
"""
import json
import math
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, hilbert, find_peaks
from scipy.signal.windows import hann, dpss
from sklearn.decomposition import NMF

CORPUS_DIR = os.environ.get(
    "BIG_BEN_CORPUS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"),
)
OUT_DIR = os.environ.get(
    "BIG_BEN_OUT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out"),
)
os.makedirs(OUT_DIR, exist_ok=True)


def load_audio_mono(fname, sr_target=None):
    fp = os.path.join(CORPUS_DIR, fname)
    if fname.endswith((".wav", ".flac", ".ogg")) and sr_target is None:
        try:
            x, sr = sf.read(fp, always_2d=True)
            return x.mean(axis=1).astype(np.float64), int(sr)
        except Exception:
            pass
    with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
        cmd = ["ffmpeg", "-y", "-i", fp, "-ac", "1"]
        if sr_target is not None:
            cmd += ["-ar", str(sr_target)]
        cmd += [tf.name]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        x, sr = sf.read(tf.name)
        return x.astype(np.float64), int(sr)


def hz_to_cents_a4(f):
    return 1200.0 * math.log2(f / 440.0)


def cents_to_note_name(cents_from_a4):
    names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    midi = round(69 + cents_from_a4 / 100.0)
    err = cents_from_a4 - (midi - 69) * 100.0
    name = names[(midi - 69) % 12]
    octave = (midi // 12) - 1
    return f"{name}{octave} ({err:+.0f}c)"


def parabolic_peak(mag, k):
    if k <= 0 or k >= len(mag) - 1:
        return float(k), float(mag[k])
    y0, y1, y2 = np.log(mag[k - 1] + 1e-30), np.log(mag[k] + 1e-30), np.log(mag[k + 1] + 1e-30)
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-18:
        return float(k), float(mag[k])
    delta = 0.5 * (y0 - y2) / denom
    delta = max(-0.5, min(0.5, delta))
    return k + delta, float(np.exp(y1 - 0.25 * (y0 - y2) * delta))


def narrowband_peak_and_t60(seg, sr, f_center, bw_hz=12.0, nfft=262144):
    """Sub-Hz parabolic peak + narrowband Hilbert envelope T60 fit."""
    win = hann(len(seg), sym=False)
    spec = np.abs(np.fft.rfft(seg * win, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    mask = (freqs >= f_center - bw_hz) & (freqs <= f_center + bw_hz)
    idxs = np.where(mask)[0]
    k_loc = idxs[np.argmax(spec[idxs])]
    k_sub, amp = parabolic_peak(spec, k_loc)
    f_peak = float(k_sub * (sr / nfft))

    # T60 via bandpass + Hilbert log-envelope linear regression over [0.25s, 4.5s] (or available len)
    lo = max(20.0, f_peak - 8.0)
    hi = min(sr * 0.45, f_peak + 8.0)
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    xf = sosfilt(sos, seg)
    env = np.abs(hilbert(xf))
    # smooth over 80 ms to average out doublet beating when estimating global T60
    sm_len = max(1, int(0.08 * sr))
    kernel = np.ones(sm_len) / sm_len
    env_sm = np.convolve(env, kernel, mode="same")
    t = np.arange(len(env_sm)) / sr
    t_end = min(t[-1] - 0.15, 6.0)
    fit_mask = (t >= 0.25) & (t <= t_end)
    if np.sum(fit_mask) > 100:
        db = 20.0 * np.log10(env_sm[fit_mask] + 1e-12)
        # Keep points above noise floor (within 38 dB of local start)
        valid = db > (np.max(db) - 38.0)
        if np.sum(valid) > 50:
            slope, _ = np.polyfit(t[fit_mask][valid], db[valid], 1)
            t60 = float(-60.0 / slope) if slope < -0.05 else None
        else:
            t60 = None
    else:
        t60 = None
    return f_peak, amp, t60


def esprit_doublet(seg, sr, f_center, lp_hz=15.0, dec_sr=60.0):
    """Two-pole ESPRIT / Matrix Pencil around f_center to extract doublet splitting."""
    t = np.arange(len(seg)) / sr
    z = seg * np.exp(-2j * np.pi * f_center * t)
    sos = butter(4, lp_hz, btype="low", fs=sr, output="sos")
    zf = sosfilt(sos, z)
    step = max(1, int(round(sr / dec_sr)))
    zd = zf[::step]
    fs_d = sr / step
    # Skip initial 0.12 s transient, take up to 3.5 s
    i0 = int(0.12 * fs_d)
    i1 = min(len(zd), i0 + int(3.5 * fs_d))
    y = zd[i0:i1]
    N = len(y)
    L = N // 3
    if L < 10:
        return None
    H = np.column_stack([y[i : i + N - L] for i in range(L)])
    U, S, _ = np.linalg.svd(H, full_matrices=False)
    U2 = U[:, :2]
    # TLS / LS shift invariance
    U_top = U2[:-1, :]
    U_bot = U2[1:, :]
    psi = np.linalg.lstsq(U_top, U_bot, rcond=None)[0]
    eigs = np.linalg.eigvals(psi)
    freqs = f_center + (np.angle(eigs) * fs_d) / (2.0 * np.pi)
    decays = -np.log(np.clip(np.abs(eigs), 1e-6, 1.5)) * fs_d
    order = np.argsort(freqs)
    freqs = freqs[order]
    decays = decays[order]
    split = float(freqs[1] - freqs[0])
    # Fit complex amplitudes
    tt = np.arange(N) / fs_d
    V = np.column_stack([np.exp((1j * 2 * np.pi * (f - f_center) - d) * tt) for f, d in zip(freqs, decays)])
    amps = np.abs(np.linalg.lstsq(V, y, rcond=None)[0])
    return {
        "f1_hz": round(float(freqs[0]), 3),
        "f2_hz": round(float(freqs[1]), 3),
        "split_hz": round(split, 3),
        "beat_period_s": round(1.0 / split, 3) if split > 0.05 else None,
        "amp_ratio": round(float(min(amps) / max(amps)), 3),
        "t60_1_s": round(float(6.9078 / decays[0]), 2) if decays[0] > 0.02 else None,
        "t60_2_s": round(float(6.9078 / decays[1]), 2) if decays[1] > 0.02 else None,
    }


def terhardt_virtual_pitch(freqs_hz, amps_lin, f_cands=None):
    """
    Harmonic-sieve / subharmonic coincidence virtual-pitch salience model (Terhardt / Schroeder):
    Each partial f_m votes for subharmonics f_m / n (n = 1..6) with harmonic weight 1/sqrt(n)
    and perceptual spectral-dominance weighting peaking around 600-900 Hz.
    """
    if f_cands is None:
        f_cands = np.linspace(120.0, 520.0, 2001)
    salience = np.zeros_like(f_cands)
    sigma_cents = 22.0
    max_amp = max(amps_lin) + 1e-12
    for fm, am in zip(freqs_hz, amps_lin):
        if am <= 0 or fm < 140:
            continue
        # Spectral dominance peaking near 700 Hz (Terhardt 1979)
        dom = math.exp(-0.5 * ((math.log2(fm / 720.0)) / 1.15) ** 2)
        w = (am / max_amp) * dom
        for n in range(1, 7):
            sub = fm / n
            if sub < 110.0 or sub > 540.0:
                continue
            cents_diff = 1200.0 * np.log2(f_cands / sub)
            salience += (w / math.sqrt(n)) * np.exp(-0.5 * (cents_diff / sigma_cents) ** 2)
    if np.max(salience) > 0:
        salience /= np.max(salience)
    return f_cands, salience


def detect_exact_onsets(x, sr, min_dist_s=0.42, thresh_frac=0.15):
    hop = int(0.01 * sr)
    n_f = len(x) // hop
    rms = np.sqrt(np.mean(x[: n_f * hop].reshape(n_f, hop) ** 2, axis=1) + 1e-12)
    diff = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    peaks, _ = find_peaks(diff, height=np.max(diff) * thresh_frac, distance=int(min_dist_s / 0.01))
    return [float(p * 0.01) for p in peaks]


def refine_onset(x, sr, approx_t, search_win=0.25):
    i0 = max(0, int((approx_t - search_win) * sr))
    i1 = min(len(x), int((approx_t + search_win) * sr))
    seg = np.abs(x[i0:i1])
    hop = int(0.005 * sr)
    n_f = len(seg) // hop
    env = np.sqrt(np.mean(seg[: n_f * hop].reshape(n_f, hop) ** 2, axis=1) + 1e-12)
    d = np.diff(env, prepend=env[0])
    k = int(np.argmax(d))
    return (i0 + k * hop) / sr


def analyze_modal_and_quarters():
    # 1. Load matched 1969 BBC WAVs + 1953 BBC 12 o'clock + 2024 NYE
    x_q1, sr = load_audio_mono("07002139.wav")
    x_q2, _ = load_audio_mono("07002140.wav")
    x_q3, _ = load_audio_mono("07002141.wav")
    x_q4, _ = load_audio_mono("07002142.wav")
    x_53, sr53 = load_audio_mono("bbc_1953_12oclock.wav")
    x_1o, sr1o = load_audio_mono("commons_1oclock.ogg")
    x_24, sr24 = load_audio_mono("fs_720696_nye2024.mp3")

    # Find exact 1 o'clock strike in 07002142.wav (last low-band peak around 24.8 s)
    sos_low = butter(4, [170, 360], btype="band", fs=sr, output="sos")
    e_q4 = np.abs(sosfilt(sos_low, x_q4))
    t_bb_1969 = refine_onset(x_q4, sr, 24.80, 0.5)
    # 12th strike in 1953 (~70.5 s)
    t_bb_1953 = refine_onset(x_53, sr53, 70.55, 0.6)
    # 1 o'clock strike in commons_1oclock (~16.7 s)
    t_bb_1oc = refine_onset(x_1o, sr1o, 16.70, 0.5)
    # 12th strike in 2024 NYE (~65.5 s)
    t_bb_2024 = refine_onset(x_24, sr24, 65.45, 0.6)

    # Big Ben partial targets (nominal ~335 + lower hum ~200 + upper partials)
    bb_targets = [
        ("Hum (G3)", 200.0),
        ("Lower / Prime?", 327.0),
        ("Nominal (E4 +27c)", 335.0),
        ("Quint-region (B4)", 497.0),
        ("Octave Nominal (E5)", 677.0),
        ("Super-Third (A5)", 883.0),
        ("Upper Partial 6 (C#6)", 1088.0),
        ("Upper Partial 7 (E6)", 1306.0),
        ("Upper Partial 8 (A6)", 1746.0),
    ]

    seg_69 = x_q4[int(t_bb_1969 * sr) : int((t_bb_1969 + 14.0) * sr)]
    seg_53 = x_53[int(t_bb_1953 * sr53) : int((t_bb_1953 + 14.0) * sr53)]
    seg_24 = x_24[int(t_bb_2024 * sr24) : int((t_bb_2024 + 14.0) * sr24)]

    # Measure partials on 1969 (0..14s) and 1953 (0..14s) and 2024 (0..14s)
    bb_partials = []
    ref_amps_69 = []
    for label, fc in bb_targets:
        f69, a69, t60_69 = narrowband_peak_and_t60(seg_69, sr, fc, bw_hz=12.0 if fc != 327.0 else 5.0)
        f53, a53, t60_53 = narrowband_peak_and_t60(seg_53, sr53, fc, bw_hz=12.0 if fc != 327.0 else 5.0)
        f24, a24, t60_24 = narrowband_peak_and_t60(seg_24, sr24, fc, bw_hz=12.0 if fc != 327.0 else 5.0)
        ref_amps_69.append(a69)
        bb_partials.append({
            "label": label,
            "f_1969_hz": round(f69, 2),
            "f_1953_hz": round(f53, 2),
            "f_2024_hz": round(f24, 2),
            "note_1969": cents_to_note_name(hz_to_cents_a4(f69)),
            "ratio_to_nom_1969": round(f69 / 335.4, 3),
            "t60_1969_s": round(t60_69, 2) if t60_69 else None,
            "t60_1953_s": round(t60_53, 2) if t60_53 else None,
            "amp_1969_raw": a69,
            "amp_2024_raw": a24,
        })
    max_a69 = max(p["amp_1969_raw"] for p in bb_partials)
    max_a24 = max(p["amp_2024_raw"] for p in bb_partials)
    for p in bb_partials:
        p["rel_db_1969"] = round(20.0 * math.log10(p["amp_1969_raw"] / max_a69 + 1e-12), 2)
        p["rel_db_2024"] = round(20.0 * math.log10(p["amp_2024_raw"] / max_a24 + 1e-12), 2)
        del p["amp_1969_raw"], p["amp_2024_raw"]

    # Doublets on Big Ben (1969 & 1953 & 2024) at 200, 335, 500, 678, 883 Hz
    bb_doublets = {}
    for band in [200.0, 335.0, 500.0, 678.0, 883.0]:
        bb_doublets[f"{int(band)}Hz"] = {
            "1969": esprit_doublet(seg_69, sr, band, lp_hz=12.0),
            "1953": esprit_doublet(seg_53, sr53, band, lp_hz=12.0),
            "2024": esprit_doublet(seg_24, sr24, band, lp_hz=12.0),
        }

    # Time-evolving virtual pitch across 4 windows of the 1969 1 o'clock strike
    windows = [(0.0, 0.5), (0.5, 3.0), (3.0, 8.0), (8.0, 14.0)]
    vp_evolution = []
    f_targets_vp = [200.0, 335.0, 499.0, 677.0, 883.0, 1088.0, 1306.0, 1746.0]
    for w0, w1 in windows:
        sub = seg_69[int(w0 * sr) : int(w1 * sr)]
        win = hann(len(sub), sym=False)
        sp = np.abs(np.fft.rfft(sub * win, n=131072)) / (np.sum(win) / 2.0)
        fr = np.fft.rfftfreq(131072, 1.0 / sr)
        f_meas, a_meas = [], []
        for fc in f_targets_vp:
            m = (fr >= fc - 10) & (fr <= fc + 10)
            f_meas.append(float(fr[m][np.argmax(sp[m])]))
            a_meas.append(float(np.max(sp[m])))
        f_cands, sal = terhardt_virtual_pitch(f_meas, a_meas)
        # Read salience specifically near E3 (165-172 Hz), G3 (196-204 Hz), E4 (330-340 Hz), A4 (436-446 Hz)
        def peak_near(lo, hi):
            mm = (f_cands >= lo) & (f_cands <= hi)
            idx = np.where(mm)[0][np.argmax(sal[mm])]
            return round(float(f_cands[idx]), 2), round(float(sal[idx]), 3)
        vp_evolution.append({
            "window_s": [w0, w1],
            "top_virtual_pitch_hz": round(float(f_cands[np.argmax(sal)]), 2),
            "top_note": cents_to_note_name(hz_to_cents_a4(float(f_cands[np.argmax(sal)]))),
            "E3_168_salience": peak_near(162, 173),
            "G3_200_salience": peak_near(195, 205),
            "E4_335_salience": peak_near(328, 342),
            "A4_441_salience": peak_near(434, 448),
            "partial_amps_db": {str(int(fc)): round(20 * math.log10(am / (max(a_meas) + 1e-12) + 1e-12), 1) for fc, am in zip(f_targets_vp, a_meas)},
        })

    # Extract all 4 Quarter Bells from held cadential endings in 07002139..07002141.wav
    # Q1 (07002139): P1 = [G#4, F#4, E4, B3] -> check onsets
    q1_on = detect_exact_onsets(x_q1, sr, 0.45, 0.16)
    q2_on = detect_exact_onsets(x_q2, sr, 0.45, 0.16)
    q3_on = detect_exact_onsets(x_q3, sr, 0.45, 0.16)
    q4_on = detect_exact_onsets(x_q4, sr, 0.45, 0.16)

    # Known quarter-bell approximate nominal / strike frequencies from Whitechapel 1857-58 cast:
    # Bell 1 (G#4 ~830-870 nominal -> strike ~415-435 Hz), Bell 2 (F#4 ~740-771 nominal -> strike ~370-385),
    # Bell 3 (E4 ~660-680 nominal -> strike ~330-340), Bell 4 (B3 ~494-505 nominal -> strike ~247-252)
    # Grab solo/held notes from Q1/Q2/Q3 where each bell ends a phrase cleanly!
    # Wait: let's inspect spectral peaks around each onset in Q1, Q2, Q3 to assign exact onset times to Bell 1..4
    def extract_quarter_bell_partials():
        # In Q1 (07002139.wav): 4 notes [G#4, F#4, E4, B3]. Note 4 (B3) rings alone for > 14s!
        # In Q2 (07002140.wav): 8 notes [E4, G#4, F#4, B3 | E4, F#4, G#4, E4]. Note 8 (E4) rings alone for > 12s!
        # Find exact 4 onsets in Q1 (first 4 prominent peaks) and 8 in Q2
        hop = int(0.01 * sr)
        def note_peaks(x, n_notes):
            on = detect_exact_onsets(x, sr, min_dist_s=0.55, thresh_frac=0.14)
            # Filter to cluster of quarter notes before long decay
            return on[:n_notes]
        o1 = note_peaks(x_q1, 4)
        o2 = note_peaks(x_q2, 8)
        o3 = note_peaks(x_q3, 12)
        o4 = note_peaks(x_q4, 16)

        # Measure each bell using its cleanest isolated window:
        # Bell 4 (B3): Q1 note 4 (t = o1[3] .. o1[3]+6s)
        # Bell 3 (E4): Q2 note 8 (t = o2[7] .. o2[7]+6s)
        # Bell 1 (G#4): Q1 note 1 (0..0.9s before note 2) + Q2 note 7 (0..0.9s)
        # Bell 2 (F#4): Q1 note 2 (subtracting residual? Use short 0.1..0.85s FFT around F#4 bands!)
        specs = [
            ("Quarter Bell 1 (G#4)", 870.0, [435.0, 520.0, 650.0, 870.0, 1080.0, 1300.0], x_q1, o1[0], 0.90),
            ("Quarter Bell 2 (F#4)", 771.0, [385.0, 460.0, 580.0, 771.0, 960.0, 1155.0], x_q1, o1[1], 0.90),
            ("Quarter Bell 3 (E4)", 679.0, [339.0, 405.0, 508.0, 679.0, 845.0, 1018.0], x_q2, o2[7], 5.00),
            ("Quarter Bell 4 (B3)", 504.0, [252.0, 302.0, 378.0, 504.0, 628.0, 756.0], x_q1, o1[3], 6.00),
        ]
        # Let's actually discover the true strongest peaks in [150, 2600] Hz for each bell segment!
        bells_out = []
        for name, nom_hint, _, x_src, t0, dur in specs:
            seg = x_src[int(t0 * sr) : int((t0 + dur) * sr)]
            w = hann(len(seg), sym=False)
            sp = np.abs(np.fft.rfft(seg * w, n=262144))
            fr = np.fft.rfftfreq(262144, 1.0 / sr)
            band = (fr >= 150) & (fr <= 2600)
            pks, _ = find_peaks(sp[band], height=np.max(sp[band]) * 0.10, distance=int(25.0 / (sr / 262144)))
            found = []
            b_idxs = np.where(band)[0]
            for pk in pks:
                k_sub, amp = parabolic_peak(sp, b_idxs[pk])
                found.append((float(k_sub * sr / 262144), float(amp)))
            # Pick nominal closest to nom_hint
            nom_f, nom_a = min(found, key=lambda z: abs(z[0] - nom_hint))
            max_a = max(z[1] for z in found)
            found_sorted = sorted(found, key=lambda z: z[0])
            partials = [
                {
                    "f_hz": round(f, 2),
                    "ratio_to_nom": round(f / nom_f, 3),
                    "rel_db": round(20 * math.log10(a / max_a + 1e-12), 1),
                    "note": cents_to_note_name(hz_to_cents_a4(f)),
                }
                for f, a in found_sorted[:10]
            ]
            bells_out.append({
                "name": name,
                "onset_s": round(t0, 3),
                "window_s": dur,
                "nominal_hz": round(nom_f, 2),
                "implied_strike_hz": round(nom_f / 2.0, 2),
                "implied_strike_note": cents_to_note_name(hz_to_cents_a4(nom_f / 2.0)),
                "partials": partials,
            })
        return bells_out, {"Q1": o1, "Q2": o2, "Q3": o3, "Q4": o4}

    q_bells, q_onsets = extract_quarter_bell_partials()

    # Check 1969 Q4 timing vs Hour strike and 1953 12-blow cadence
    h_peaks_53, _ = find_peaks(
        np.maximum(0, np.diff(np.sqrt(np.mean(sosfilt(sos_low, x_53)[: (len(x_53) // int(0.01 * sr53)) * int(0.01 * sr53)].reshape(-1, int(0.01 * sr53)) ** 2, axis=1) + 1e-12), prepend=0)),
        height=0.0,
    )
    # Use clean low-band onset times around 25..71s in bbc_1953
    hop53 = int(0.01 * sr53)
    xb53 = sosfilt(sos_low, x_53)
    eb53 = np.sqrt(np.mean(xb53[: (len(xb53) // hop53) * hop53].reshape(-1, hop53) ** 2, axis=1) + 1e-12)
    db53 = np.maximum(0, np.diff(eb53, prepend=eb53[0]))
    bp53, _ = find_peaks(db53, height=np.max(db53) * 0.25, distance=int(2.8 / 0.01))
    blows_53 = [round(float(p * 0.01), 2) for p in bp53 if p * 0.01 > 23.0]
    ioi_53 = [round(blows_53[i + 1] - blows_53[i], 3) for i in range(len(blows_53) - 1)]

    # Combinatorial invariant proof
    perms = {
        "P1": ["G#4", "F#4", "E4", "B3"],
        "P2": ["E4", "G#4", "F#4", "B3"],
        "P3": ["E4", "F#4", "G#4", "E4"],
        "P4": ["G#4", "E4", "F#4", "B3"],
        "P5": ["B3", "F#4", "G#4", "E4"],
    }
    quarters_map = {
        "Q1 (:15)": ["P1"],
        "Q2 (:30)": ["P2", "P3"],
        "Q3 (:45)": ["P4", "P5", "P1"],
        "Q4 (:00)": ["P2", "P3", "P4", "P5"],
    }
    hourly_seq = [n for q in quarters_map.values() for p in q for n in perms[p]]
    counts = {bell: hourly_seq.count(bell) for bell in ["G#4", "F#4", "E4", "B3"]}

    out = {
        "combinatorics": {
            "permutations": perms,
            "quarters": quarters_map,
            "total_quarter_notes_per_hour": len(hourly_seq),
            "cycle_modulus": 5,
            "strikes_per_bell_per_hour": counts,
        },
        "observed_quarter_onsets_1969_s": q_onsets,
        "bbc_1953_12_hour_blows_s": blows_53,
        "bbc_1953_hour_blow_ioi_mean_s": round(float(np.mean(ioi_53)), 3) if ioi_53 else None,
        "bbc_1953_hour_blow_ioi_std_s": round(float(np.std(ioi_53)), 3) if ioi_53 else None,
        "big_ben_partials": bb_partials,
        "big_ben_doublets": bb_doublets,
        "virtual_pitch_evolution_1969": vp_evolution,
        "quarter_bells_modal": q_bells,
    }
    with open(os.path.join(OUT_DIR, "quarters_and_bells_modal.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved quarters_and_bells_modal.json")
    return out


def iso9613_alpha_db_per_m(f_hz, temp_c, rh_pct, p_kpa=101.325):
    """ISO 9613-1 atmospheric sound absorption coefficient alpha in dB/m."""
    T = temp_c + 273.15
    T0 = 293.15
    T01 = 273.16
    p0 = 101.325
    pr = p_kpa / p0
    Tr = T / T0
    # Saturation vapor pressure ratio
    C = -6.8346 * (T01 / T) ** 1.261 + 4.6151
    psat_p0 = 10.0 ** C
    h = rh_pct * psat_p0 / pr
    fro = pr * (24.0 + 4.04e4 * h * ((0.02 + h) / (0.391 + h)))
    frn = pr * (Tr ** -0.5) * (9.0 + 280.0 * h * math.exp(-4.170 * (Tr ** (-1.0 / 3.0) - 1.0)))
    f2 = f_hz ** 2
    alpha_np_m = (
        f2
        * (
            1.84e-11 * (1.0 / pr) * math.sqrt(Tr)
            + (Tr ** -2.5)
            * (
                0.01275 * math.exp(-2239.1 / T) / (fro + f2 / fro)
                + 0.1068 * math.exp(-3352.0 / T) / (frn + f2 / frn)
            )
        )
    )
    return 8.686 * alpha_np_m


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def analyze_little_deans_yard():
    # Coordinates (WGS84):
    # Elizabeth Tower belfry center: 51.500729 N, -0.124625 W, z_source = 65 m AGL
    # Little Dean's Yard center (Westminster School): 51.49855 N, -0.12825 W, z_rx = 1.7 m AGL
    # Westminster Abbey nave / Henry VII Chapel roof barrier along the ray path (~165 m from tower)
    lat_t, lon_t, z_t = 51.500729, -0.124625, 65.0
    lat_y, lon_y, z_y = 51.498550, -0.128250, 1.7
    d_horiz = haversine_m(lat_t, lon_t, lat_y, lon_y)
    d_slant = math.sqrt(d_horiz ** 2 + (z_t - z_y) ** 2)

    # Bearing from Tower to Yard
    dl = math.radians(lon_y - lon_t)
    y_b = math.sin(dl) * math.cos(math.radians(lat_y))
    x_b = math.cos(math.radians(lat_t)) * math.sin(math.radians(lat_y)) - math.sin(math.radians(lat_t)) * math.cos(math.radians(lat_y)) * math.cos(dl)
    bearing_deg = (math.degrees(math.atan2(y_b, x_b)) + 360.0) % 360.0

    # Ray height at Abbey crossing d1 = 165 m from tower, d2 = d_horiz - 165 m to yard
    d1 = 165.0
    d2 = d_horiz - d1
    z_los_at_abbey = z_t + (z_y - z_t) * (d1 / d_horiz)
    # Test two realistic Abbey roof / pinnacle barrier heights:
    # (a) Nave roof ridge z_bar = 31.0 m (barely grazing / shadow edge, h_eff = -4.0 m below LOS? Wait: z_los ~35 m!)
    # (b) South transept / Ashburnham House / Abbey clerestory parapet + courtyard wall closer to yard (say d1_wall = 310 m, d2_wall = d_horiz - 310, z_wall = 14 m; LOS at 310 m is 8.6 m -> +5.4 m ABOVE LOS!)
    # Actually Little Dean's Yard is bounded on the NE by Ashburnham House / Abbey Cloister & College Hall (~15-18 m) and the Abbey nave (~31-42 m)!
    # Let's compute exact ray geometry for BOTH:
    #   Barrier A (Mid-path Abbey nave parapet/pinnacle h = 38.0 m at d1 = 165 m, LOS = 35.0 m -> excess +3.0 m)
    #   Barrier B (Yard NE cloister/Ashburnham gable h = 16.5 m at d1 = 305 m, LOS = 9.5 m -> excess +7.0 m)
    def maekawa_att_db(f_hz, c_ms, d1_b, d2_b, z_bar):
        z_los = z_t + (z_y - z_t) * (d1_b / d_horiz)
        # Direct path length and diffracted path length
        da = math.sqrt(d1_b ** 2 + (z_t - z_bar) ** 2)
        db = math.sqrt(d2_b ** 2 + (z_bar - z_y) ** 2)
        d_dir = math.sqrt(d_horiz ** 2 + (z_t - z_y) ** 2)
        delta = da + db - d_dir
        # Sign of Fresnel number: positive if z_bar > z_los (blocked), negative if grazing
        sign = 1.0 if z_bar >= z_los else -1.0
        lam = c_ms / f_hz
        NF = sign * 2.0 * delta / lam
        if NF <= -0.1:
            return 0.0, NF, delta
        return 10.0 * math.log10(3.0 + 20.0 * max(-0.09, NF)), NF, delta

    freqs_test = [200.0, 335.0, 500.0, 678.0, 883.0, 1088.0, 1306.0, 1746.0, 3000.0, 5000.0]
    winter = {"temp_c": 0.0, "rh_pct": 85.0}
    summer = {"temp_c": 28.0, "rh_pct": 45.0}
    c_win = 331.3 * math.sqrt(1.0 + winter["temp_c"] / 273.15)
    c_sum = 331.3 * math.sqrt(1.0 + summer["temp_c"] / 273.15)
    lag_win = d_slant / c_win
    lag_sum = d_slant / c_sum

    # Hibbert thermal pitch shift: -2.78 cents / 10 C
    bronze_shift_cents = -2.78 * ((summer["temp_c"] - winter["temp_c"]) / 10.0)
    bronze_shift_hz_335 = 335.4 * (2.0 ** (bronze_shift_cents / 1200.0) - 1.0)

    # Geometric spreading loss relative to 10 m belfry reference: 20*log10(354/10) = 30.98 dB
    geom_loss_db = 20.0 * math.log10(d_slant / 10.0)

    transfer_curve = []
    for f in freqs_test:
        att_cloister, nf_c, del_c = maekawa_att_db(f, 340.0, 305.0, d_horiz - 305.0, 16.5)
        att_abbey, nf_a, del_a = maekawa_att_db(f, 340.0, 165.0, d_horiz - 165.0, 38.5)
        air_win = iso9613_alpha_db_per_m(f, **winter) * d_slant
        air_sum = iso9613_alpha_db_per_m(f, **summer) * d_slant
        transfer_curve.append({
            "f_hz": f,
            "diff_cloister_db": round(att_cloister, 2),
            "diff_abbey_pinnacle_db": round(att_abbey, 2),
            "air_winter_0c_db": round(air_win, 2),
            "air_summer_28c_db": round(air_sum, 2),
            "total_excess_loss_cloister_winter_db": round(att_cloister + air_win, 2),
            "total_excess_loss_cloister_summer_db": round(att_abbey + air_sum, 2),
        })

    # Empirical belfry vs ground-level check on 1969 BBC matched recordings: 07002139.wav vs 07002154.wav
    x_bel, sr_b = load_audio_mono("07002139.wav")
    x_gnd, sr_g = load_audio_mono("07002154.wav")
    # Measure octave-band PSD during the 4 quarter strikes vs pre-strike ambient
    def band_db(sig, srate, t0, t1, flo, fhi):
        s = sig[int(t0 * srate) : int(t1 * srate)]
        sp = np.abs(np.fft.rfft(s * hann(len(s), sym=False))) ** 2
        fr = np.fft.rfftfreq(len(s), 1.0 / srate)
        m = (fr >= flo) & (fr <= fhi)
        return 10.0 * math.log10(float(np.sum(sp[m])) + 1e-18)

    # First 4 notes in 07002139 (0.7..4.8s) vs 07002154 (2.0..6.2s)
    bel_low = band_db(x_bel, sr_b, 0.7, 5.5, 300, 950)
    bel_high = band_db(x_bel, sr_b, 0.7, 5.5, 2000, 6000)
    gnd_low = band_db(x_gnd, sr_g, 2.0, 6.8, 300, 950)
    gnd_high = band_db(x_gnd, sr_g, 2.0, 6.8, 2000, 6000)

    # Audible tail duration in Little Dean's Yard: day (62 dBA traffic floor) vs night (44 dBA quiet yard)
    # Using Big Ben's measured 335 Hz T60 (~13.5-15s) and 200 Hz hum T60 (~16-20s):
    # At 354 m behind cloister (-31 dB spreading - 10 dB diffraction = -41 dB from 10 m ~114 dB -> ~73 dB SPL peak in yard)
    # Daytime masking threshold ~60 dB SPL (13 dB headroom); Nighttime ~43 dB SPL (30 dB headroom)
    t60_e4 = 13.8
    audible_day_s = t60_e4 * (13.0 / 60.0)
    audible_night_s = t60_e4 * (30.0 / 60.0)

    out = {
        "geometry": {
            "tower_lat": lat_t,
            "tower_lon": lon_t,
            "belfry_height_m": z_t,
            "yard_lat": lat_y,
            "yard_lon": lon_y,
            "horizontal_distance_m": round(d_horiz, 1),
            "slant_distance_m": round(d_slant, 1),
            "bearing_tower_to_yard_deg": round(bearing_deg, 1),
            "los_height_at_abbey_165m_m": round(z_los_at_abbey, 1),
            "spreading_loss_vs_10m_db": round(geom_loss_db, 2),
        },
        "seasonal_physics": {
            "c_winter_0c_ms": round(c_win, 2),
            "c_summer_28c_ms": round(c_sum, 2),
            "lag_winter_0c_s": round(lag_win, 4),
            "lag_summer_28c_s": round(lag_sum, 4),
            "seasonal_lag_shift_ms": round((lag_win - lag_sum) * 1000.0, 1),
            "hibbert_bronze_shift_cents_0_to_28c": round(bronze_shift_cents, 2),
            "hibbert_bronze_shift_hz_at_335hz": round(bronze_shift_hz_335, 2),
            "audible_tail_day_60db_s": round(audible_day_s, 1),
            "audible_tail_night_43db_s": round(audible_night_s, 1),
        },
        "transfer_curve": transfer_curve,
        "empirical_1969_belfry_vs_ground_hf_rolloff": {
            "belfry_high_minus_low_db": round(bel_high - bel_low, 2),
            "ground_high_minus_low_db": round(gnd_high - gnd_low, 2),
            "measured_hf_drop_db": round((bel_high - bel_low) - (gnd_high - gnd_low), 2),
        },
    }
    with open(os.path.join(OUT_DIR, "little_deans_yard_acoustics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved little_deans_yard_acoustics.json")
    return out


def analyze_forensic_1890_2024(modal_ref):
    # Canonical Big Ben dimensionless ratios from 1969 WAV
    nom_ref = next(p["f_1969_hz"] for p in modal_ref["big_ben_partials"] if "Nominal" in p["label"])
    ref_freqs = [p["f_1969_hz"] for p in modal_ref["big_ben_partials"]]

    corpus_files = [
        ("1890 Edison Cylinder", "commons_1890_edison.ogg", 65.77),
        ("1945 BBC Broadcast WAV", "commons_1945_broadcast.wav", 49.65),
        ("1953 BBC 12 o'clock WAV", "bbc_1953_12oclock.wav", 70.57),
        ("1969 BBC 1 o'clock WAV", "07002142.wav", 24.85),
        ("2016 Bridge St Binaural WAV", "aporee_2016_bridge_st.wav", None),
        ("2017 Midnight HQ MP3", "midnight-hq.mp3", 42.09),
        ("2017 Midnight LQ MP3", "midnight-lq.mp3", 42.09),
        ("2024 NYE Post-Restoration MP3", "fs_720696_nye2024.mp3", 65.49),
        ("2024 Milsom Post-Restoration MP3", "fs_791991_milsom.mp3", None),
        ("Parliament Sq LQ MP3", "square-lq.mp3", 46.97),
    ]

    # 1. 1890 Edison Cylinder invariant modal-ratio speed recovery
    x_1890, sr_90 = load_audio_mono("commons_1890_edison.ogg")
    # Isolate the actual Big Ben hour strikes (88..141 s), avoiding 53..83 s quarter chimes
    seg_90 = x_1890[int(88 * sr_90) :]
    w90 = hann(len(seg_90), sym=False)
    sp90 = np.abs(np.fft.rfft(seg_90 * w90, n=524288))
    fr90 = np.fft.rfftfreq(524288, 1.0 / sr_90)
    # Canonical 1969 Big Ben strike modes (incl. 200.26 Hz Hum + 332.7/335.5 doublet center)
    tmpl_f = np.array([200.26, 334.1, 499.75, 677.60, 883.34, 1088.24])
    alphas = np.linspace(0.80, 1.30, 5001)
    scores_alpha = []
    # Pre-whiten 1890 spectrum locally (divide by 35 Hz median smooth) to suppress horn resonance tilt
    from scipy.ndimage import median_filter
    df90 = sr_90 / 524288.0
    m_band = (fr90 >= 150) & (fr90 <= 2200)
    fr_b = fr90[m_band]
    sp_b = sp90[m_band]
    sp_white = sp_b / (median_filter(sp_b, size=int(35.0 / df90)) + 1e-12)
    for a in alphas:
        sc = 0.0
        for tf in tmpl_f:
            idx = int(round((tf * a - 150.0) / df90))
            if 5 <= idx < len(sp_white) - 5:
                sc += math.log1p(float(np.max(sp_white[idx - 6 : idx + 7])))
        scores_alpha.append(sc)
    best_idx = int(np.argmax(scores_alpha))
    alpha_hat = float(alphas[best_idx])
    # Refine peaks around alpha_hat * ref_1969_hz
    recovered_1890_peaks = []
    for tf in [200.26, 335.51, 499.75, 677.60, 883.34, 1088.24, 1315.45]:
        fc_pred = tf * alpha_hat
        m = (fr_b >= fc_pred - 14.0) & (fr_b <= fc_pred + 14.0)
        if np.any(m):
            f_obs = float(fr_b[m][np.argmax(sp_white[m])])
            snr_loc = float(np.max(sp_white[m]))
            recovered_1890_peaks.append({
                "ref_1969_hz": tf,
                "raw_1890_hz": round(f_obs, 2),
                "speed_corrected_hz": round(f_obs / alpha_hat, 2),
                "residual_cents": round(1200.0 * math.log2((f_obs / alpha_hat) / tf), 1),
                "local_whitened_snr": round(snr_loc, 2),
            })

    # 2. Bandwidth / ENF / Partial preservation across all 10 recordings
    forensic_rows = []
    nmf_results = {}
    for label, fn, strike_t in corpus_files:
        x, sr = load_audio_mono(fn)
        # Global Welch-like FFT for bandwidth & 50/100Hz ENF
        nfft = min(len(x), 131072)
        seg = x[:nfft]
        sp = np.abs(np.fft.rfft(seg * hann(len(seg), sym=False))) ** 2
        fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
        cum = np.cumsum(sp) / (np.sum(sp) + 1e-18)
        f99 = float(fr[np.searchsorted(cum, 0.99)])
        f995 = float(fr[np.searchsorted(cum, 0.995)])

        # ENF 50 Hz / 100 Hz narrowband prominence vs [46..48, 52..54] Hz sidebands
        def enf_snr(f0):
            sig_m = (fr >= f0 - 0.35) & (fr <= f0 + 0.35)
            bg_m = ((fr >= f0 - 3.5) & (fr <= f0 - 1.2)) | ((fr >= f0 + 1.2) & (fr <= f0 + 3.5))
            if not np.any(sig_m) or not np.any(bg_m):
                return None
            return round(10.0 * math.log10((np.max(sp[sig_m]) + 1e-18) / (np.median(sp[bg_m]) + 1e-18)), 2)

        enf50 = enf_snr(50.0)
        enf100 = enf_snr(100.0)

        # If recording has an hour strike, measure 335, 500, 678, 883, 1306 Hz + spectral centroid of Big Ben
        bb_meas = None
        if strike_t is not None and strike_t + 2.0 < len(x) / sr:
            st_t = refine_onset(x, sr, strike_t, 0.5)
            s_win = x[int((st_t + 0.05) * sr) : int(min(len(x), (st_t + 3.5) * sr))]
            f_335, a_335, _ = narrowband_peak_and_t60(s_win, sr, 335.4, bw_hz=12.0, nfft=131072)
            f_500, a_500, _ = narrowband_peak_and_t60(s_win, sr, 499.5, bw_hz=12.0, nfft=131072)
            f_678, a_678, _ = narrowband_peak_and_t60(s_win, sr, 677.3, bw_hz=12.0, nfft=131072)
            f_883, a_883, _ = narrowband_peak_and_t60(s_win, sr, 883.1, bw_hz=12.0, nfft=131072)
            f_1306, a_1306, _ = narrowband_peak_and_t60(s_win, sr, 1306.0, bw_hz=15.0, nfft=131072)
            amps = np.array([a_335, a_500, a_678, a_883, a_1306])
            freqs_v = np.array([f_335, f_500, f_678, f_883, f_1306])
            centroid_5p = float(np.sum(freqs_v * amps) / (np.sum(amps) + 1e-12))
            bb_meas = {
                "f_335": round(f_335, 2),
                "f_500": round(f_500, 2),
                "f_678": round(f_678, 2),
                "f_883": round(f_883, 2),
                "f_1306": round(f_1306, 2),
                "partial_centroid_hz": round(centroid_5p, 1),
                "ratio_883_to_335_db": round(20.0 * math.log10(a_883 / (a_335 + 1e-12) + 1e-12), 2),
            }

        forensic_rows.append({
            "label": label,
            "file": fn,
            "sr_hz": sr,
            "f99_hz": round(f99, 0),
            "f995_hz": round(f995, 0),
            "enf_50hz_snr_db": enf50,
            "enf_100hz_snr_db": enf100,
            "big_ben_strike": bb_meas,
        })

        # Unsupervised KL-NMF (k=6) on first 30s (or full clip) on magnitude STFT [150..2200 Hz]
        if fn in ("07002142.wav", "commons_1945_broadcast.wav", "aporee_2016_bridge_st.wav", "fs_720696_nye2024.mp3", "square-lq.mp3"):
            x_nmf = x[: min(len(x), int(40 * sr))]
            n_fft = 4096
            hop = 1024
            frames = []
            w_h = hann(n_fft, sym=False)
            for start in range(0, len(x_nmf) - n_fft, hop):
                mag = np.abs(np.fft.rfft(x_nmf[start : start + n_fft] * w_h))
                frames.append(mag)
            V = np.column_stack(frames)
            fr_nmf = np.fft.rfftfreq(n_fft, 1.0 / sr)
            band = (fr_nmf >= 150) & (fr_nmf <= 2200)
            V_b = V[band, :] + 1e-6
            fr_b_nmf = fr_nmf[band]
            model = NMF(n_components=6, init="nndsvda", beta_loss="kullback-leibler", solver="mu", max_iter=300, random_state=42)
            W = model.fit_transform(V_b)
            H = model.components_
            V_hat = W @ H
            # Explained variance / KL reconstruction SNR
            err = np.sum((V_b - V_hat) ** 2)
            tot = np.sum(V_b ** 2)
            r2 = 1.0 - err / (tot + 1e-12)
            # Dominant frequency of each NMF basis atom
            atom_peaks = [round(float(fr_b_nmf[np.argmax(W[:, j])]), 1) for j in range(6)]
            nmf_results[fn] = {
                "r2_variance_explained": round(float(r2), 4),
                "atom_peak_hz": sorted(atom_peaks),
            }

    out = {
        "edison_1890_speed_recovery": {
            "estimated_speed_multiplier_alpha": round(alpha_hat, 4),
            "implied_pitch_shift_semitones": round(12.0 * math.log2(alpha_hat), 2),
            "recovered_partials": recovered_1890_peaks,
        },
        "corpus_forensic_metrics": forensic_rows,
        "kl_nmf_k6_separation": nmf_results,
    }
    with open(os.path.join(OUT_DIR, "forensic_1890_2024.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved forensic_1890_2024.json")
    return out


if __name__ == "__main__":
    m = analyze_modal_and_quarters()
    analyze_little_deans_yard()
    analyze_forensic_1890_2024(m)
