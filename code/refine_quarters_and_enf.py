#!/usr/bin/env python3
"""
Refine Quarter Bell modal isolation via pre-onset subtraction and verify exact 50.00 Hz ENF
plus multi-recording restoration comparison.
"""
import json, math, os
import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import hann
from analyze_big_ben import (
    load_audio_mono, detect_exact_onsets, parabolic_peak,
    hz_to_cents_a4, cents_to_note_name, refine_onset, narrowband_peak_and_t60, OUT_DIR
)

def clean_quarter_bells():
    x_q1, sr = load_audio_mono("07002139.wav")
    x_q2, _ = load_audio_mono("07002140.wav")
    o1 = detect_exact_onsets(x_q1, sr, min_dist_s=0.55, thresh_frac=0.14)[:4]
    o2 = detect_exact_onsets(x_q2, sr, min_dist_s=0.55, thresh_frac=0.14)[:8]

    def diff_spectrum(x, t_on, pre_s=0.75, post_s=0.95, nfft=262144):
        i_on = int(t_on * sr)
        i_post = min(len(x), i_on + int(post_s * sr))
        s_post = x[i_on:i_post]
        sp_post = np.abs(np.fft.rfft(s_post * hann(len(s_post), sym=False), n=nfft)) ** 2
        sp_post /= (np.sum(hann(len(s_post), sym=False)) ** 2)
        if t_on > pre_s + 0.05:
            i_pre0 = max(0, i_on - int(pre_s * sr))
            s_pre = x[i_pre0:i_on]
            sp_pre = np.abs(np.fft.rfft(s_pre * hann(len(s_pre), sym=False), n=nfft)) ** 2
            sp_pre /= (np.sum(hann(len(s_pre), sym=False)) ** 2)
            # Attenuated residual bleed subtraction (decay factor ~0.85)
            sp_net = np.maximum(0.0, sp_post - 0.85 * sp_pre)
        else:
            sp_net = sp_post
        return np.sqrt(sp_net), np.fft.rfftfreq(nfft, 1.0 / sr)

    # Bell 1 (G#4): Q1 note 0 (clean first note!)
    # Bell 2 (F#4): Q1 note 1 (with Bell 1 pre-subtraction)
    # Bell 3 (E4):  Q2 note 7 (with Q2 note 6 pre-subtraction) AND Q1 note 2 (with pre-subtraction)
    # Bell 4 (B3):  Q1 note 3 (with pre-subtraction, long 4s window)
    configs = [
        ("Quarter Bell 1 (G#4)", 870.0, x_q1, o1[0], 0.50, 0.85),
        ("Quarter Bell 2 (F#4)", 771.0, x_q1, o1[1], 0.70, 0.85),
        ("Quarter Bell 3 (E4)",  679.0, x_q2, o2[0], 0.40, 0.90),
        ("Quarter Bell 4 (B3)",  505.0, x_q1, o1[3], 0.70, 1.80),
    ]
    bells = []
    for name, nom_hint, x_src, t0, pre_s, post_s in configs:
        mag, fr = diff_spectrum(x_src, t0, pre_s=pre_s, post_s=post_s)
        band = (fr >= 170) & (fr <= 2600)
        pks, _ = find_peaks(mag[band], height=np.max(mag[band]) * 0.14, distance=int(30.0 / (sr / 262144)))
        b_idxs = np.where(band)[0]
        found = []
        for pk in pks:
            k_sub, amp = parabolic_peak(mag, b_idxs[pk])
            found.append((float(k_sub * sr / 262144), float(amp)))
        nom_f, _ = min(found, key=lambda z: abs(z[0] - nom_hint))
        max_a = max(z[1] for z in found)
        partials = [
            {
                "f_hz": round(f, 2),
                "ratio_to_nom": round(f / nom_f, 3),
                "rel_db": round(20.0 * math.log10(a / max_a + 1e-12), 1),
                "note": cents_to_note_name(hz_to_cents_a4(f)),
            }
            for f, a in sorted(found, key=lambda z: z[0])[:10]
        ]
        bells.append({
            "name": name,
            "onset_s": round(t0, 3),
            "nominal_hz": round(nom_f, 2),
            "implied_strike_hz": round(nom_f / 2.0, 2),
            "implied_strike_note": cents_to_note_name(hz_to_cents_a4(nom_f / 2.0)),
            "partials": partials,
        })

    qpath = os.path.join(OUT_DIR, "quarters_and_bells_modal.json")
    qdata = json.load(open(qpath))
    qdata["quarter_bells_modal"] = bells
    json.dump(qdata, open(qpath, "w"), indent=2)
    print("Updated clean quarter_bells_modal:")
    for b in bells:
        print(" ", b["name"], "nom=", b["nominal_hz"], "partials=", [(p["f_hz"], p["ratio_to_nom"], p["rel_db"]) for p in b["partials"][:6]])

def check_enf_and_extra_recordings():
    fpath = os.path.join(OUT_DIR, "forensic_1890_2024.json")
    fdata = json.load(open(fpath))
    for row in fdata["corpus_forensic_metrics"]:
        x, sr = load_audio_mono(row["file"])
        nfft = min(len(x), 262144)
        seg = x[:nfft]
        sp = np.abs(np.fft.rfft(seg * hann(len(seg), sym=False))) ** 2
        fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
        m50 = (fr >= 49.5) & (fr <= 50.5)
        bg50 = ((fr >= 46.0) & (fr <= 48.5)) | ((fr >= 51.5) & (fr <= 54.0))
        idx50 = np.where(m50)[0][np.argmax(sp[m50])]
        f50_pk = float(fr[idx50])
        snr50 = 10.0 * math.log10((float(sp[idx50]) + 1e-18) / (float(np.median(sp[bg50])) + 1e-18))
        # Narrow-line sharpness: peak vs 0.25 Hz neighbors
        nbr = (fr >= f50_pk - 0.6) & (fr <= f50_pk + 0.6) & (np.abs(fr - f50_pk) > 0.15)
        line_sharp_db = 10.0 * math.log10((float(sp[idx50]) + 1e-18) / (float(np.median(sp[nbr])) + 1e-18))
        row["enf_peak_hz"] = round(f50_pk, 3)
        row["enf_line_sharpness_db"] = round(line_sharp_db, 2)
        row["true_50hz_mains_lock"] = bool(abs(f50_pk - 50.0) < 0.22 and line_sharp_db > 6.0)

    # Also add commons_1oclock.ogg & fs_656683_11am.mp3 measurements
    extras = []
    for label, fn, st_guess in [
        ("Archive 1 o'clock Ogg", "commons_1oclock.ogg", 16.73),
        ("Modern 11am Post-2017 MP3", "fs_656683_11am.mp3", 66.98),
    ]:
        x, sr = load_audio_mono(fn)
        st_t = refine_onset(x, sr, st_guess, 0.5)
        s_win = x[int((st_t + 0.05) * sr) : int(min(len(x), (st_t + 3.5) * sr))]
        f_335, a_335, _ = narrowband_peak_and_t60(s_win, sr, 335.4, bw_hz=12.0, nfft=131072)
        f_500, a_500, _ = narrowband_peak_and_t60(s_win, sr, 499.5, bw_hz=12.0, nfft=131072)
        f_678, a_678, _ = narrowband_peak_and_t60(s_win, sr, 677.3, bw_hz=12.0, nfft=131072)
        f_883, a_883, _ = narrowband_peak_and_t60(s_win, sr, 883.1, bw_hz=12.0, nfft=131072)
        f_1306, a_1306, _ = narrowband_peak_and_t60(s_win, sr, 1306.0, bw_hz=15.0, nfft=131072)
        amps = np.array([a_335, a_500, a_678, a_883, a_1306])
        freqs_v = np.array([f_335, f_500, f_678, f_883, f_1306])
        extras.append({
            "label": label, "file": fn, "sr_hz": sr,
            "f_335": round(f_335, 2), "f_500": round(f_500, 2),
            "f_678": round(f_678, 2), "f_883": round(f_883, 2),
            "partial_centroid_hz": round(float(np.sum(freqs_v * amps) / (np.sum(amps) + 1e-12)), 1),
            "ratio_883_to_335_db": round(20.0 * math.log10(a_883 / (a_335 + 1e-12) + 1e-12), 2),
        })
    fdata["extra_hour_checks"] = extras
    json.dump(fdata, open(fpath, "w"), indent=2)
    print("ENF line sharpness:")
    for r in fdata["corpus_forensic_metrics"]:
        print(f"  {r['file']:32s} f_pk={r['enf_peak_hz']:6.3f} sharp={r['enf_line_sharpness_db']:5.2f}dB mains={r['true_50hz_mains_lock']}")
    print("Extra hour checks:", extras)

if __name__ == "__main__":
    clean_quarter_bells()
    check_enf_and_extra_recordings()
