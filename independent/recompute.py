#!/usr/bin/env python3
"""Independent blind numerical verification of Big Ben audio & OSINT findings."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch


def get_paths() -> tuple[Path, Path]:
    base = Path(__file__).resolve().parent.parent
    corpus_dir = Path(os.environ.get("BIG_BEN_CORPUS_DIR", base / "corpus"))
    out_dir = Path(os.environ.get("BIG_BEN_OUT_DIR", base / "out"))
    return corpus_dir, out_dir


def parabolic_peak(
    freqs: np.ndarray, mag: np.ndarray, mag_db: np.ndarray, f_target: float, bw: float = 4.0
) -> tuple[float, float]:
    mask = (freqs >= f_target - bw) & (freqs <= f_target + bw)
    idxs = np.where(mask)[0]
    k = int(idxs[int(np.argmax(mag[idxs]))])
    if 0 < k < len(mag) - 1:
        y0, y1, y2 = mag_db[k - 1], mag_db[k], mag_db[k + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
    else:
        delta = 0.0
    return float(freqs[k] + delta * (freqs[1] - freqs[0])), float(mag_db[k])


def band_ratio_db(path: Path, lo_band: tuple[float, float], hi_band: tuple[float, float]) -> float:
    x, sr = sf.read(path)
    mono = x.mean(axis=1) if x.ndim > 1 else x
    f, p = welch(mono, fs=sr, window="blackman", nperseg=min(len(mono), 65536))
    elo = np.sum(p[(f >= lo_band[0]) & (f <= lo_band[1])])
    ehi = np.sum(p[(f >= hi_band[0]) & (f <= hi_band[1])])
    return float(10.0 * np.log10((ehi + 1e-20) / (elo + 1e-20)))


def verify_combinatorics(ref_modal: dict) -> dict:
    perms = {
        "P1": ["G#4", "F#4", "E4", "B3"],
        "P2": ["E4", "G#4", "F#4", "B3"],
        "P3": ["E4", "F#4", "G#4", "E4"],
        "P4": ["G#4", "E4", "F#4", "B3"],
        "P5": ["B3", "F#4", "G#4", "E4"],
    }
    quarters = {
        "Q1 (:15)": ["P1"],
        "Q2 (:30)": ["P2", "P3"],
        "Q3 (:45)": ["P4", "P5", "P1"],
        "Q4 (:00)": ["P2", "P3", "P4", "P5"],
    }
    unique_perms = len({tuple(v) for v in perms.values()})
    perm_seq = [p for q in quarters.values() for p in q]
    notes_seq = [note for p in perm_seq for note in perms[p]]
    strike_counts = {b: notes_seq.count(b) for b in ["G#4", "F#4", "E4", "B3"]}
    notes_per_quarter = {q: sum(len(perms[p]) for p in ps) for q, ps in quarters.items()}
    return {
        "unique_permutations": unique_perms,
        "total_permutation_plays_per_hour": len(perm_seq),
        "cycle_modulus": unique_perms,
        "cycle_closes_per_hour": (len(perm_seq) % unique_perms == 0),
        "cycles_per_hour": len(perm_seq) // unique_perms,
        "notes_per_quarter": notes_per_quarter,
        "total_quarter_notes_per_hour": len(notes_seq),
        "strikes_per_bell_per_hour": strike_counts,
        "matches_reference_exact": (
            unique_perms == ref_modal["combinatorics"]["cycle_modulus"]
            and len(notes_seq) == ref_modal["combinatorics"]["total_quarter_notes_per_hour"]
            and strike_counts == ref_modal["combinatorics"]["strikes_per_bell_per_hour"]
        ),
    }


def verify_1969_isolated_strike(corpus_dir: Path, ref_modal: dict) -> tuple[dict, np.ndarray, int]:
    x69, sr69 = sf.read(corpus_dir / "07002142.wav")
    m69 = x69.mean(axis=1) if x69.ndim > 1 else x69

    s0, s1 = int(22.0 * sr69), int(28.0 * sr69)
    w10ms = np.ones(int(0.01 * sr69)) / int(0.01 * sr69)
    sm69 = np.convolve(np.abs(m69[s0:s1]), w10ms, mode="same")
    i_on = int(np.where(sm69 > 0.25 * sm69.max())[0][0])
    t_onset = (s0 + i_on) / sr69

    seg69 = m69[int(t_onset * sr69) :]
    win69 = np.blackman(len(seg69))
    nfft69 = 1 << 20
    spec69 = np.abs(np.fft.rfft(seg69 * win69, n=nfft69))
    freqs69 = np.fft.rfftfreq(nfft69, 1.0 / sr69)
    spec69_db = 20.0 * np.log10(spec69 + 1e-15)

    ref_lookup = {
        200: 200.26,
        335: 335.51,
        500: 499.75,
        678: 677.60,
        883: 883.34,
        1088: 1088.24,
        1315: 1315.45,
        1748: 1748.04,
    }
    partials = {}
    for ft, ref_v in ref_lookup.items():
        fp, _ = parabolic_peak(freqs69, spec69, spec69_db, float(ft), bw=4.0)
        partials[str(ft)] = {
            "measured_hz": round(fp, 3),
            "reference_hz": ref_v,
            "delta_hz": round(fp - ref_v, 3),
            "abs_delta_hz": round(abs(fp - ref_v), 3),
        }

    f1_fft, _ = parabolic_peak(freqs69, spec69, spec69_db, 332.7, bw=1.5)
    f2_fft, _ = parabolic_peak(freqs69, spec69, spec69_db, 335.5, bw=1.5)
    split_fft = f2_fft - f1_fft

    sos_d = butter(4, [328.0, 340.0], btype="bandpass", fs=sr69, output="sos")
    seg_bp = sosfiltfilt(sos_d, seg69[: int(15.0 * sr69)])
    env_h = np.abs(hilbert(seg_bp))
    t_h = np.arange(len(env_h)) / sr69
    m_h = (t_h >= 0.3) & (t_h <= 10.0)
    t_sub = t_h[m_h]
    log_env = np.log(env_h[m_h] + 1e-12)
    poly = np.polyfit(t_sub, log_env, 1)
    detrended = np.exp(log_env - np.polyval(poly, t_sub))
    w25 = np.ones(int(0.025 * sr69)) / int(0.025 * sr69)
    det_sm = np.convolve(detrended, w25, mode="same")
    epks, _ = find_peaks(det_sm, distance=int(0.22 * sr69), prominence=0.05)
    periods = np.diff(t_sub[epks])
    med_period = float(np.median(periods))
    split_hilbert = 1.0 / med_period

    win_e = np.hanning(len(det_sm))
    spec_e = np.abs(np.fft.rfft((det_sm - det_sm.mean()) * win_e, n=1 << 18))
    f_e = np.fft.rfftfreq(1 << 18, 1.0 / sr69)
    m_fe = (f_e >= 1.0) & (f_e <= 6.0)
    df_env_fft = float(f_e[m_fe][np.argmax(spec_e[m_fe])])

    ref_dbl = ref_modal["big_ben_doublets"]["335Hz"]["1969"]
    res = {
        "strike_onset_s": round(t_onset, 3),
        "tail_duration_s": round(len(seg69) / sr69, 3),
        "partials_hz": partials,
        "max_partial_abs_delta_hz_6_core": round(
            max(partials[str(k)]["abs_delta_hz"] for k in [200, 335, 500, 678, 883, 1088]), 3
        ),
        "doublet_335hz": {
            "fft_f1_hz": round(f1_fft, 3),
            "fft_f2_hz": round(f2_fft, 3),
            "fft_split_hz": round(split_fft, 3),
            "fft_beat_period_s": round(1.0 / split_fft, 4),
            "hilbert_n_peaks": int(len(epks)),
            "hilbert_median_beat_period_s": round(med_period, 4),
            "hilbert_mean_beat_period_s": round(float(np.mean(periods)), 4),
            "hilbert_std_beat_period_s": round(float(np.std(periods)), 4),
            "hilbert_implied_split_hz": round(split_hilbert, 3),
            "hilbert_env_fft_freq_hz": round(df_env_fft, 4),
            "ref_f1_hz": ref_dbl["f1_hz"],
            "ref_f2_hz": ref_dbl["f2_hz"],
            "ref_split_hz": ref_dbl["split_hz"],
            "ref_beat_period_s": ref_dbl["beat_period_s"],
            "delta_fft_split_hz": round(split_fft - ref_dbl["split_hz"], 3),
            "delta_hilbert_split_hz": round(split_hilbert - ref_dbl["split_hz"], 3),
            "delta_hilbert_period_s": round(med_period - ref_dbl["beat_period_s"], 4),
        },
    }
    return res, seg69, sr69


def verify_1890_edison(
    corpus_dir: Path, seg69: np.ndarray, sr69: int, ref_modal: dict, ref_forensic: dict
) -> dict:
    x90, sr90 = sf.read(corpus_dir / "commons_1890_edison.ogg")
    m90 = x90.mean(axis=1) if x90.ndim > 1 else x90
    f69_w, p69_w = welch(seg69, fs=sr69, window="blackman", nperseg=min(len(seg69), 131072), nfft=262144)
    ref_alpha = ref_forensic["edison_1890_speed_recovery"]["estimated_speed_multiplier_alpha"]

    def log_freq_xcorr(seg: np.ndarray, sr: int):
        f90, p90 = welch(seg, fs=sr, window="blackman", nperseg=min(len(seg), 131072), nfft=262144)
        u = np.linspace(np.log(170.0), np.log(1600.0), 4000)
        fu = np.exp(u)
        lp69 = np.interp(fu, f69_w, np.log10(p69_w + 1e-20))
        lp90 = np.interp(fu, f90, np.log10(p90 + 1e-20))
        w69 = np.maximum(0.0, lp69 - median_filter(lp69, size=181))
        w90 = np.maximum(0.0, lp90 - median_filter(lp90, size=181))
        alphas = np.linspace(0.80, 1.35, 2201)
        scores = np.empty_like(alphas)
        n69 = np.sqrt(np.sum(w69**2)) + 1e-12
        for idx, a in enumerate(alphas):
            w90s = np.interp(fu * a, fu, w90, left=0.0, right=0.0)
            scores[idx] = np.sum(w69 * w90s) / (n69 * (np.sqrt(np.sum(w90s**2)) + 1e-12))
        ib = int(np.argmax(scores))
        iref = int(np.argmin(np.abs(alphas - ref_alpha)))
        return float(alphas[ib]), float(scores[ib]), float(scores[iref]), f90, p90

    a_88, c_88, c_88_ref, f90_88, p90_88 = log_freq_xcorr(m90[int(88 * sr90) :], sr90)
    a_full, c_full, c_full_ref, _, _ = log_freq_xcorr(m90, sr90)

    sos_90 = butter(4, [175.0, 330.0], btype="bandpass", fs=sr90, output="sos")
    bp90 = sosfiltfilt(sos_90, m90[int(85 * sr90) :])
    hop90 = int(0.05 * sr90)
    rms90 = np.array([np.sqrt(np.mean(bp90[i : i + hop90] ** 2)) for i in range(0, len(bp90) - hop90, hop90)])
    pks90, _ = find_peaks(rms90, distance=int(3.0 / 0.05), prominence=0.05)
    med_ioi90 = float(np.median(np.diff(pks90 * 0.05)))
    alpha_ioi = ref_modal["bbc_1953_hour_blow_ioi_mean_s"] / med_ioi90

    lp90_88 = 10.0 * np.log10(p90_88 + 1e-20)
    exc90_88 = lp90_88 - median_filter(lp90_88, size=301)
    matched = []
    for r_hz in [200.26, 335.51, 499.75, 677.60, 883.34, 1088.24]:
        ft = r_hz * a_88
        m = (f90_88 >= ft - 15.0) & (f90_88 <= ft + 15.0)
        idxs = np.where(m)[0]
        k = int(idxs[int(np.argmax(exc90_88[idxs]))])
        fp = float(f90_88[k])
        corr_hz = fp / a_88
        cents = 1200.0 * math.log2(corr_hz / r_hz)
        matched.append(
            {
                "ref_1969_hz": r_hz,
                "raw_1890_hz": round(fp, 2),
                "corrected_hz": round(corr_hz, 2),
                "residual_cents": round(cents, 1),
                "local_snr_db": round(float(exc90_88[k]), 2),
            }
        )

    return {
        "hour_strikes_88_141s": {
            "alpha": round(a_88, 4),
            "semitones": round(12.0 * math.log2(a_88), 2),
            "xcorr": round(c_88, 4),
            "xcorr_at_ref_alpha": round(c_88_ref, 4),
            "matched_partials": matched,
        },
        "full_recording_0_141s": {
            "alpha": round(a_full, 4),
            "semitones": round(12.0 * math.log2(a_full), 2),
            "xcorr": round(c_full, 4),
            "xcorr_at_ref_alpha": round(c_full_ref, 4),
        },
        "independent_ioi_check_85_141s": {
            "n_blows_detected": int(len(pks90)),
            "median_ioi_s": round(med_ioi90, 3),
            "ref_1953_ioi_s": ref_modal["bbc_1953_hour_blow_ioi_mean_s"],
            "implied_alpha_from_ioi": round(alpha_ioi, 4),
        },
        "reference_forensic_alpha": ref_alpha,
        "reference_forensic_semitones": ref_forensic["edison_1890_speed_recovery"]["implied_pitch_shift_semitones"],
        "alpha_discrepancy_vs_ref": round(a_88 - ref_alpha, 4),
    }


def verify_little_deans_yard(corpus_dir: Path, ref_ldy: dict) -> dict:
    lat1, lon1 = ref_ldy["geometry"]["tower_lat"], ref_ldy["geometry"]["tower_lon"]
    lat2, lon2 = ref_ldy["geometry"]["yard_lat"], ref_ldy["geometry"]["yard_lon"]
    h_belfry = ref_ldy["geometry"]["belfry_height_m"]

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a_hav = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    d_hav = 2.0 * 6371000.0 * math.asin(math.sqrt(a_hav))
    slant_hav = math.hypot(d_hav, h_belfry)

    a_wgs, f_wgs = 6378137.0, 1.0 / 298.257223563
    e2 = f_wgs * (2.0 - f_wgs)
    phi_m = 0.5 * (phi1 + phi2)
    m_rad = a_wgs * (1.0 - e2) / (1.0 - e2 * math.sin(phi_m) ** 2) ** 1.5
    n_rad = a_wgs / math.sqrt(1.0 - e2 * math.sin(phi_m) ** 2)
    dn, de = dphi * m_rad, dlmb * n_rad * math.cos(phi_m)
    d_wgs = math.hypot(dn, de)
    slant_wgs = math.hypot(d_wgs, h_belfry)
    bearing = (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0

    c0 = 331.3 * math.sqrt(1.0 + 0.0 / 273.15)
    c28 = 331.3 * math.sqrt(1.0 + 28.0 / 273.15)
    lag0_hav, lag28_hav = slant_hav / c0, slant_hav / c28
    lag0_wgs, lag28_wgs = slant_wgs / c0, slant_wgs / c28

    belfry_hi_lo = band_ratio_db(corpus_dir / "07002139.wav", (200, 1000), (2000, 5000))
    ground_hi_lo = band_ratio_db(corpus_dir / "07002154.wav", (200, 1000), (2000, 5000))
    drop_db = belfry_hi_lo - ground_hi_lo

    return {
        "haversine_horiz_m": round(d_hav, 2),
        "haversine_slant_m": round(slant_hav, 2),
        "wgs84_horiz_m": round(d_wgs, 2),
        "wgs84_slant_m": round(slant_wgs, 2),
        "bearing_deg": round(bearing, 2),
        "ref_horiz_m": ref_ldy["geometry"]["horizontal_distance_m"],
        "ref_slant_m": ref_ldy["geometry"]["slant_distance_m"],
        "delta_horiz_haversine_m": round(d_hav - ref_ldy["geometry"]["horizontal_distance_m"], 2),
        "delta_slant_haversine_m": round(slant_hav - ref_ldy["geometry"]["slant_distance_m"], 2),
        "c_0c_ms": round(c0, 2),
        "c_28c_ms": round(c28, 2),
        "lag_0c_hav_s": round(lag0_hav, 4),
        "lag_28c_hav_s": round(lag28_hav, 4),
        "lag_diff_hav_ms": round((lag0_hav - lag28_hav) * 1000.0, 2),
        "lag_diff_wgs_ms": round((lag0_wgs - lag28_wgs) * 1000.0, 2),
        "ref_lag_diff_ms": ref_ldy["seasonal_physics"]["seasonal_lag_shift_ms"],
        "delta_lag_diff_ms": round((lag0_hav - lag28_hav) * 1000.0 - ref_ldy["seasonal_physics"]["seasonal_lag_shift_ms"], 2),
        "bands_hz": {"low": [200, 1000], "high": [2000, 5000]},
        "belfry_07002139_hi_lo_db": round(belfry_hi_lo, 2),
        "ground_07002154_hi_lo_db": round(ground_hi_lo, 2),
        "measured_hf_drop_db": round(drop_db, 2),
        "ref_measured_hf_drop_db": ref_ldy["empirical_1969_belfry_vs_ground_hf_rolloff"]["measured_hf_drop_db"],
        "delta_hf_drop_db": round(drop_db - ref_ldy["empirical_1969_belfry_vs_ground_hf_rolloff"]["measured_hf_drop_db"], 2),
    }


def verify_codec_and_tilt(corpus_dir: Path, ref_osint: dict) -> dict:
    xh, srh = sf.read(corpus_dir / "midnight-hq.mp3")
    xl, srl = sf.read(corpus_dir / "midnight-lq.mp3")
    mh = xh.mean(axis=1) if xh.ndim > 1 else xh
    ml = xl.mean(axis=1) if xl.ndim > 1 else xl

    fh_w, ph_w = welch(mh, fs=srh, window="blackman", nperseg=4 * srh, nfft=16 * srh)
    fl_w, pl_w = welch(ml, fs=srl, window="blackman", nperseg=4 * srl, nfft=16 * srl)
    ph_db, pl_db = 10.0 * np.log10(ph_w + 1e-20), 10.0 * np.log10(pl_w + 1e-20)

    codec_freqs, max_f_full = {}, 0.0
    for ft in [335, 500, 678, 883, 1306]:
        fph, _ = parabolic_peak(fh_w, ph_w, ph_db, float(ft), bw=6.0)
        fpl, _ = parabolic_peak(fl_w, pl_w, pl_db, float(ft), bw=6.0)
        df_hl = abs(fph - fpl)
        max_f_full = max(max_f_full, df_hl)
        codec_freqs[str(ft)] = {"hq_hz": round(fph, 4), "lq_hz": round(fpl, 4), "abs_drift_hz": round(df_hl, 4)}

    sh42, sl42 = mh[int(42 * srh) : int(50 * srh)], ml[int(42 * srl) : int(50 * srl)]
    fh42, ph42 = welch(sh42, fs=srh, window="blackman", nperseg=4 * srh, nfft=16 * srh)
    fl42, pl42 = welch(sl42, fs=srl, window="blackman", nperseg=4 * srl, nfft=16 * srl)
    codec_42_50, max_f_42_50 = {}, 0.0
    for ft in [333, 500, 678, 883, 1306]:
        fph, _ = parabolic_peak(fh42, ph42, 10.0 * np.log10(ph42 + 1e-20), float(ft), bw=6.0)
        fpl, _ = parabolic_peak(fl42, pl42, 10.0 * np.log10(pl42 + 1e-20), float(ft), bw=6.0)
        df_hl = abs(fph - fpl)
        max_f_42_50 = max(max_f_42_50, df_hl)
        codec_42_50[str(ft)] = {"hq_hz": round(fph, 4), "lq_hz": round(fpl, 4), "abs_drift_hz": round(df_hl, 4)}

    tilt_hq = band_ratio_db(corpus_dir / "midnight-hq.mp3", (200, 900), (1000, 4000))
    tilt_lq = band_ratio_db(corpus_dir / "midnight-lq.mp3", (200, 900), (1000, 4000))
    tilt_drift = abs(tilt_hq - tilt_lq)

    tilt_belfry = band_ratio_db(corpus_dir / "07002139.wav", (200, 900), (1000, 4000))
    tilt_aporee = band_ratio_db(corpus_dir / "aporee_2016_bridge_st.wav", (200, 900), (1000, 4000))
    qt_belfry = band_ratio_db(corpus_dir / "07002139.wav", (450, 950), (1000, 2600))
    qt_aporee = band_ratio_db(corpus_dir / "aporee_2016_bridge_st.wav", (450, 950), (1000, 2600))

    by_id = {x["id"]: x for x in ref_osint["site_osint_observables"]}
    return {
        "midnight_hq_vs_lq": {
            "full_recording_eigenfreqs": codec_freqs,
            "max_eigenfreq_drift_full_hz": round(max_f_full, 4),
            "window_42_50s_eigenfreqs": codec_42_50,
            "max_eigenfreq_drift_42_50s_hz": round(max_f_42_50, 4),
            "eigenfreq_threshold_0_05_hz_pass": bool(max_f_full < 0.05 and max_f_42_50 < 0.05),
            "tilt_hq_200_900_vs_1000_4000_db": round(tilt_hq, 3),
            "tilt_lq_200_900_vs_1000_4000_db": round(tilt_lq, 3),
            "tilt_abs_drift_db": round(tilt_drift, 3),
            "tilt_threshold_0_2_db_pass": bool(tilt_drift < 0.2),
            "ref_quarter_tilt_drift_db": round(
                abs(by_id["midnight_hq_2017"]["quarter_tilt_db"] - by_id["midnight_lq_2017"]["quarter_tilt_db"]), 3
            ),
            "ref_big_ben_tilt_drift_db": round(
                abs(by_id["midnight_hq_2017"]["big_ben_tilt_db"] - by_id["midnight_lq_2017"]["big_ben_tilt_db"]), 3
            ),
        },
        "aporee_2016_vs_bbc_1969_belfry_q1": {
            "broadband_200_900_vs_1000_4000_db": {
                "bbc_1969_belfry_q1": round(tilt_belfry, 2),
                "aporee_2016_bridge_st": round(tilt_aporee, 2),
                "difference_db": round(tilt_aporee - tilt_belfry, 2),
            },
            "quarter_band_450_950_vs_1000_2600_db": {
                "bbc_1969_belfry_q1": round(qt_belfry, 2),
                "aporee_2016_bridge_st": round(qt_aporee, 2),
                "difference_db": round(qt_aporee - qt_belfry, 2),
            },
            "ref_osint_quarter_tilt_db": {
                "bbc_1969_belfry_q1": by_id["bbc_1969_belfry_q1"]["quarter_tilt_db"],
                "aporee_2016_bridge_st": by_id["aporee_2016_bridge_st"]["quarter_tilt_db"],
                "difference_db": round(
                    by_id["aporee_2016_bridge_st"]["quarter_tilt_db"] - by_id["bbc_1969_belfry_q1"]["quarter_tilt_db"], 2
                ),
            },
        },
    }


def main() -> None:
    corpus_dir, out_dir = get_paths()
    ref_modal = json.loads((out_dir / "quarters_and_bells_modal.json").read_text())
    ref_ldy = json.loads((out_dir / "little_deans_yard_acoustics.json").read_text())
    ref_forensic = json.loads((out_dir / "forensic_1890_2024.json").read_text())
    ref_osint = json.loads((out_dir / "osint_geolocation.json").read_text())

    comb = verify_combinatorics(ref_modal)
    bb69, seg69, sr69 = verify_1969_isolated_strike(corpus_dir, ref_modal)
    ed1890 = verify_1890_edison(corpus_dir, seg69, sr69, ref_modal, ref_forensic)
    ldy = verify_little_deans_yard(corpus_dir, ref_ldy)
    codec_tilt = verify_codec_and_tilt(corpus_dir, ref_osint)

    recomputed = {
        "combinatorics": comb,
        "big_ben_1969_isolated_strike": bb69,
        "edison_1890_speed_recovery": ed1890,
        "little_deans_yard": ldy,
        "codec_invariance_and_tilt": codec_tilt,
    }
    (out_dir / "RECOMPUTED.json").write_text(json.dumps(recomputed, indent=2) + "\n")
    print(json.dumps(recomputed, indent=2))


if __name__ == "__main__":
    main()
