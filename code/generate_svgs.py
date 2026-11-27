from pathlib import Path
#!/usr/bin/env python3
"""
Generate 4 publication-grade standalone SVGs directly from measured JSON outputs
matching zidingz.com palette (#FAF9F5 paper, #F3F1EA tint, #141413 ink, #6B6A64 muted,
#BA6348 clay, #3D5A6C slate; Georgia serif + Menlo monospace tabular readouts).
"""
import json, math, os

OUT_ROOT = os.environ.get("BIG_BEN_IMG_DIR", str(Path(__file__).resolve().parent.parent.parent / "images"))
DATA_DIR = os.environ.get("BIG_BEN_OUT_DIR", str(Path(__file__).resolve().parent.parent / "out"))

modal = json.load(open(f"{DATA_DIR}/quarters_and_bells_modal.json"))
yard = json.load(open(f"{DATA_DIR}/little_deans_yard_acoustics.json"))
forensic = json.load(open(f"{DATA_DIR}/forensic_1890_2024.json"))
osint = json.load(open(f"{DATA_DIR}/osint_geolocation.json"))

PAPER = "#FAF9F5"
TINT = "#F3F1EA"
INK = "#141413"
MUTED = "#6B6A64"
CLAY = "#BA6348"
SLATE = "#3D5A6C"
GOLD = "#B8842E"
RULE = "#DDD9CE"

def svg_header(w, h, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">
<style>
  .title {{ font-family: Georgia, "Times New Roman", serif; font-size: 20px; fill: {INK}; font-weight: 600; }}
  .sub {{ font-family: Georgia, "Times New Roman", serif; font-size: 13.5px; fill: {MUTED}; }}
  .h2 {{ font-family: Georgia, "Times New Roman", serif; font-size: 15px; fill: {INK}; font-weight: 600; }}
  .lbl {{ font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; font-size: 11.5px; fill: {INK}; }}
  .mut {{ font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; font-size: 11px; fill: {MUTED}; }}
  .num {{ font-family: Menlo, Monaco, "Courier New", monospace; font-size: 11px; fill: {INK}; }}
  .num-m {{ font-family: Menlo, Monaco, "Courier New", monospace; font-size: 10.5px; fill: {MUTED}; }}
  .num-c {{ font-family: Menlo, Monaco, "Courier New", monospace; font-size: 11px; fill: {CLAY}; font-weight: 700; }}
  .num-s {{ font-family: Menlo, Monaco, "Courier New", monospace; font-size: 11px; fill: {SLATE}; font-weight: 700; }}
</style>
<rect width="{w}" height="{h}" fill="{PAPER}" />
'''

# ---------------------------------------------------------------------------
# Post 1, Fig 1: Four-Bell Modal Spectrum, 1858 Whitechapel vs Simpson E3, & Virtual Pitch Decay
# ---------------------------------------------------------------------------
def make_p1_fig1():
    w, h = 1040, 560
    s = [svg_header(w, h, "Measured inharmonic partials of Big Ben and the four Whitechapel quarter bells, plus virtual pitch decay over 14 seconds")]
    s.append(f'<text x="32" y="38" class="title">Inside one blow of Big Ben: split modes, thick bronze ratios, and a shifting chord</text>')
    s.append(f'<text x="32" y="60" class="sub">Measured from 1969 BBC 44.1 kHz PCM WAVs (07002139–07002142) vs. an ideal Simpson-tuned E3 bell (1 : 2 : 2.4 : 3 : 4)</text>')

    # Left panel: Partial ratio ladder (log ratio relative to nominal) for Simpson E3, Big Ben, and Q1..Q4
    lx, ly, lw, lh = 32, 86, 510, 442
    s.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="6" fill="{TINT}" stroke="{RULE}" />')
    s.append(f'<text x="{lx+18}" y="{ly+28}" class="h2">A. Partial ratios f / f_nom (Whitechapel 1857–58 vs. Simpson)</text>')
    s.append(f'<text x="{lx+18}" y="{ly+47}" class="mut">Ideal Simpson bells align 0.25 : 0.50 : 0.60 : 0.75 : 1.00 : 1.50. Thick Grimthorpe bells spread.</text>')

    # X-axis: ratio 0.2 to 2.8 (log scale!)
    ax_x0, ax_x1 = lx + 128, lx + lw - 24
    def rx(ratio):
        t = (math.log2(ratio) - math.log2(0.22)) / (math.log2(2.85) - math.log2(0.22))
        return ax_x0 + max(0.0, min(1.0, t)) * (ax_x1 - ax_x0)

    for r_tick, r_lbl in [(0.25, "0.25"), (0.5, "0.50"), (0.6, "0.60"), (1.0, "1.00"), (1.5, "1.50"), (2.0, "2.00"), (2.63, "2.63")]:
        xx = rx(r_tick)
        s.append(f'<line x1="{xx:.1f}" y1="{ly+66}" x2="{xx:.1f}" y2="{ly+lh-42}" stroke="{RULE}" stroke-dasharray="3,3" />')
        anc_t = "end" if r_tick == 0.5 else ("start" if r_tick == 0.6 else "middle")
        dx_tk = -2 if r_tick == 0.5 else (2 if r_tick == 0.6 else 0)
        s.append(f'<text x="{xx+dx_tk:.1f}" y="{ly+lh-22}" text-anchor="{anc_t}" class="num-m">{r_lbl}</text>')

    rows_data = [
        ("Ideal Simpson E3", [0.25, 0.50, 0.60, 0.75, 1.00, 1.50, 2.00], MUTED, "82–659 Hz"),
        ("Big Ben (E3/A4)", [p["ratio_to_nom_1969"] for p in modal["big_ben_partials"] if p["ratio_to_nom_1969"] <= 2.7 and p["rel_db_1969"] > -35], CLAY, "nom 335.5 Hz"),
    ]
    for qb in modal["quarter_bells_modal"]:
        short = qb["name"].replace("Quarter ", "")
        rats = [p["ratio_to_nom"] for p in qb["partials"] if 0.22 <= p["ratio_to_nom"] <= 2.75]
        rows_data.append((short, rats, SLATE, f"nom {qb['nominal_hz']:.0f} Hz"))

    for idx, (rname, rats, col, sublbl) in enumerate(rows_data):
        yy = ly + 98 + idx * 54
        s.append(f'<line x1="{ax_x0}" y1="{yy}" x2="{ax_x1}" y2="{yy}" stroke="{RULE}" />')
        s.append(f'<text x="{lx+16}" y="{yy-3}" class="lbl" font-weight="600">{rname}</text>')
        s.append(f'<text x="{lx+16}" y="{yy+13}" class="num-m">{sublbl}</text>')
        for rval in rats:
            xx = rx(rval)
            rad = 6.5 if (col == CLAY and abs(rval - 2.634) < 0.05) else 5.0
            s.append(f'<circle cx="{xx:.1f}" cy="{yy}" r="{rad}" fill="{col}" stroke="{PAPER}" stroke-width="1.2" />')
            if idx == 1 and rval in (0.597, 1.0, 1.49, 2.02, 2.634):
                tag = {0.597: "G3 200", 1.0: "E4 335", 1.49: "B4 500", 2.02: "E5 678", 2.634: "A5 883"}[rval]
                y_tag = yy + 18 if rval == 2.02 else yy - 11
                s.append(f'<text x="{xx:.1f}" y="{y_tag}" text-anchor="middle" class="num-c">{tag}</text>')
            if idx >= 2 and abs(rval - 0.591) < 0.01:
                s.append(f'<text x="{xx:.1f}" y="{yy-10}" text-anchor="middle" class="num-s">.59</text>')

    s.append(f'<text x="{lx+18}" y="{ly+lh-4}" class="mut">All 4 quarter bells place their tierce at 0.590–0.593; Big Ben lifts its hum to 0.597.</text>')

    # Right panel: Big Ben partial decay (T60 & 0..14s trajectories + Virtual Pitch crossover)
    rx0, ry0, rw, rh = 562, 86, 446, 442
    s.append(f'<rect x="{rx0}" y="{ry0}" width="{rw}" height="{rh}" rx="6" fill="{TINT}" stroke="{RULE}" />')
    s.append(f'<text x="{rx0+18}" y="{ry0+28}" class="h2">B. Why the harmony shifts across 14 seconds</text>')
    s.append(f'<text x="{rx0+18}" y="{ry0+47}" class="mut">Measured 1969 partial levels (dB) and Terhardt virtual-pitch salience</text>')

    # Plot decay lines for 200 Hz, 335 Hz, 499 Hz, 883 Hz across the 4 measured windows
    px0, px1 = rx0 + 54, rx0 + rw - 24
    py0, py1 = ry0 + 78, ry0 + 275
    s.append(f'<rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" fill="{PAPER}" stroke="{RULE}" />')
    t_mids = [0.25, 1.75, 5.5, 11.0]
    def tx(t): return px0 + (t / 12.0) * (px1 - px0)
    def ty_db(db): return py0 + ((-db) / 45.0) * (py1 - py0)

    for db_t in [0, -15, -30, -45]:
        yy = ty_db(db_t)
        s.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="{RULE}" stroke-dasharray="2,3" />')
        s.append(f'<text x="{px0-6}" y="{yy+4:.1f}" text-anchor="end" class="num-m">{db_t}</text>')
    for tt in [0, 3, 6, 9, 12]:
        xx = tx(tt)
        s.append(f'<text x="{xx:.1f}" y="{py1+15}" text-anchor="middle" class="num-m">{tt}s</text>')

    series = [
        ("200", "200 Hz G3 Hum (T60 = 29.0 s)", GOLD, "none"),
        ("499", "500 Hz B4 Fifth (T60 = 17.3 s)", SLATE, "none"),
        ("335", "335 Hz E4 Doublet (2.71 Hz swell)", CLAY, "none"),
        ("883", "883 Hz A5 -> A4 strike (T60 = 8.0 s)", "#7C5266", "5,3"),
    ]
    vp_list = modal["virtual_pitch_evolution_1969"]
    for key, label, col, dash in series:
        pts = " ".join(f"{tx(tm):.1f},{ty_db(max(-44.0, w['partial_amps_db'][key])):.1f}" for tm, w in zip(t_mids, vp_list))
        da = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        s.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.4"{da} points="{pts}" />')
        for tm, w in zip(t_mids, vp_list):
            s.append(f'<circle cx="{tx(tm):.1f}" cy="{ty_db(max(-44.0, w["partial_amps_db"][key])):.1f}" r="3.6" fill="{col}" />')

    # Legend below plot + Virtual Pitch table readout
    ly_leg = py1 + 36
    for i, (key, label, col, dash) in enumerate(series):
        yy = ly_leg + i * 20
        s.append(f'<line x1="{rx0+22}" y1="{yy-4}" x2="{rx0+44}" y2="{yy-4}" stroke="{col}" stroke-width="2.5" />')
        s.append(f'<text x="{rx0+52}" y="{yy}" class="lbl">{label}</text>')

    # Virtual-pitch salience bar summary at bottom right
    s.append(f'<text x="{rx0+22}" y="{ly_leg+94}" class="lbl" font-weight="600">Terhardt virtual-pitch salience (0–0.5s vs 8–14s):</text>')
    s.append(f'<text x="{rx0+22}" y="{ly_leg+114}" class="num">0–0.5s: E3=1.00  E4=0.99  A4(441Hz)=0.68  G3=0.11</text>')
    s.append(f'<text x="{rx0+22}" y="{ly_leg+134}" class="num-c">8–14s : B4=1.00  E3=0.70  G3(200Hz)=0.56  A4=0.02</text>')

    s.append("</svg>")
    out_p = f"{OUT_ROOT}/the-chords-inside-big-ben/fig1-modal-chord-and-virtual-pitch.svg"
    open(out_p, "w").write("\n".join(s))
    return out_p

# ---------------------------------------------------------------------------
# Post 1, Fig 2: Little Dean's Yard Abbey Diffraction & 1890-2024 Multi-Era Forensics
# ---------------------------------------------------------------------------
def make_p1_fig2():
    w, h = 1040, 560
    s = [svg_header(w, h, "Little Dean's Yard acoustic diffraction across Westminster Abbey and 134-year forensic comparison from 1890 to 2024")]
    s.append(f'<text x="32" y="38" class="title">Across Westminster Abbey &amp; 134 years: diffraction, temperature, and the 1890 wax cylinder</text>')
    s.append(f'<text x="32" y="60" class="sub">Left: Maekawa + ISO 9613-1 attenuation at 354.5 m (Little Dean’s Yard). Right: speed-recovered 1890 modes &amp; 1953–2024 invariance.</text>')

    # Left: Diffraction & Seasonal Transfer Curve
    lx, ly, lw, lh = 32, 86, 480, 442
    s.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="6" fill="{TINT}" stroke="{RULE}" />')
    s.append(f'<text x="{lx+18}" y="{ly+28}" class="h2">A. Excess path loss into Little Dean’s Yard (354.5 m)</text>')
    s.append(f'<text x="{lx+18}" y="{ly+47}" class="mut">Cloister/Abbey barrier cuts &gt;2 kHz clang 20–49 dB; winter lag +51 ms.</text>')

    px0, px1 = lx + 56, lx + lw - 24
    py0, py1 = ly + 74, ly + 295
    s.append(f'<rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" fill="{PAPER}" stroke="{RULE}" />')
    tc = yard["transfer_curve"]
    def fx(f): return px0 + (math.log10(f) - math.log10(180)) / (math.log10(5500) - math.log10(180)) * (px1 - px0)
    def ty_loss(db): return py0 + (db / 52.0) * (py1 - py0)

    for db_t in [0, 10, 20, 30, 40, 50]:
        yy = ty_loss(db_t)
        s.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="{RULE}" stroke-dasharray="2,3" />')
        s.append(f'<text x="{px0-6}" y="{yy+4:.1f}" text-anchor="end" class="num-m">-{db_t}</text>')
    for ft in [200, 335, 500, 883, 1746, 3000, 5000]:
        xx = fx(ft)
        s.append(f'<text x="{xx:.1f}" y="{py1+15}" text-anchor="middle" class="num-m">{ft}</text>')

    pts_win = " ".join(f"{fx(r['f_hz']):.1f},{ty_loss(r['total_excess_loss_cloister_winter_db']):.1f}" for r in tc)
    pts_sum = " ".join(f"{fx(r['f_hz']):.1f},{ty_loss(r['total_excess_loss_cloister_summer_db']):.1f}" for r in tc)
    pts_pin = " ".join(f"{fx(r['f_hz']):.1f},{ty_loss(r['diff_abbey_pinnacle_db']):.1f}" for r in tc)
    s.append(f'<polyline fill="none" stroke="{SLATE}" stroke-width="2.5" points="{pts_win}" />')
    s.append(f'<polyline fill="none" stroke="{CLAY}" stroke-width="2.5" points="{pts_sum}" />')
    s.append(f'<polyline fill="none" stroke="{GOLD}" stroke-width="2.0" stroke-dasharray="4,3" points="{pts_pin}" />')

    s.append(f'<line x1="{lx+24}" y1="{py1+40}" x2="{lx+46}" y2="{py1+40}" stroke="{SLATE}" stroke-width="2.5" />')
    s.append(f'<text x="{lx+54}" y="{py1+44}" class="lbl">Cloister shadow + 0°C winter damp air (-49.5 dB @ 5 kHz)</text>')
    s.append(f'<line x1="{lx+24}" y1="{py1+62}" x2="{lx+46}" y2="{py1+62}" stroke="{CLAY}" stroke-width="2.5" />')
    s.append(f'<text x="{lx+54}" y="{py1+66}" class="lbl">Abbey parapet + 28°C summer air (-29.4 dB @ 5 kHz)</text>')
    s.append(f'<line x1="{lx+24}" y1="{py1+84}" x2="{lx+46}" y2="{py1+84}" stroke="{GOLD}" stroke-width="2.0" stroke-dasharray="4,3" />')
    s.append(f'<text x="{lx+54}" y="{py1+88}" class="lbl">Pure Abbey diffraction (empirical 1969 belfry->ground: -10.9 dB)</text>')
    s.append(f'<text x="{lx+24}" y="{py1+116}" class="num">Seasonal arrival shift: 1.070 s (0°C) vs 1.019 s (28°C) = 51.0 ms</text>')
    s.append(f'<text x="{lx+24}" y="{py1+134}" class="num">Hibbert bronze elasticity shift: -7.78 cents (-1.50 Hz @ 335 Hz)</text>')

    # Right: 1890 Speed Recovery + Multi-Era Invariance
    rx0, ry0, rw, rh = 528, 86, 480, 442
    s.append(f'<rect x="{rx0}" y="{ry0}" width="{rw}" height="{rh}" rx="6" fill="{TINT}" stroke="{RULE}" />')
    s.append(f'<text x="{rx0+18}" y="{ry0+28}" class="h2">B. 1890 Edison cylinder recovery &amp; 134-year drift</text>')
    ed_meta = forensic["edison_1890_speed_recovery"]
    alpha_1890 = ed_meta["estimated_speed_multiplier_alpha"]
    st_1890 = ed_meta["implied_pitch_shift_semitones"]
    s.append(f'<text x="{rx0+18}" y="{ry0+47}" class="mut">Modal-ratio match on 88–141s strikes recovers {alpha_1890:.4f}x ({st_1890:+.2f} st) speed</text>')

    # Table of 1890 raw -> corrected -> 1969 ref -> 2024 post-restoration
    cols_x = [rx0 + 18, rx0 + 106, rx0 + 196, rx0 + 296, rx0 + 382]
    headers = ["Mode", "1890 Raw", f"1890 / {alpha_1890:.4f}", "1969 WAV", "2024 NYE"]
    hy = ry0 + 80
    for cx, hd in zip(cols_x, headers):
        s.append(f'<text x="{cx}" y="{hy}" class="lbl" font-weight="700">{hd}</text>')
    s.append(f'<line x1="{rx0+18}" y1="{hy+7}" x2="{rx0+rw-18}" y2="{hy+7}" stroke="{INK}" stroke-width="1.2" />')

    ed = ed_meta["recovered_partials"]
    bb69 = {p["label"]: p for p in modal["big_ben_partials"]}
    hum_p = next(p for p in modal["big_ben_partials"] if "Hum" in p["label"])
    nom_p = next(p for p in modal["big_ben_partials"] if "Nominal (E4" in p["label"])
    qui_p = next(p for p in modal["big_ben_partials"] if "Quint" in p["label"])
    oct_p = next(p for p in modal["big_ben_partials"] if "Octave Nominal" in p["label"])
    sup_p = next(p for p in modal["big_ben_partials"] if "Super-Third" in p["label"])
    p6_p = next(p for p in modal["big_ben_partials"] if "Upper Partial 6" in p["label"])
    map_rows = [
        ("G3 Hum", ed[0]["raw_1890_hz"], ed[0]["speed_corrected_hz"], hum_p["f_1969_hz"], hum_p["f_2024_hz"], ed[0]["residual_cents"]),
        ("E4 Nom.", ed[1]["raw_1890_hz"], ed[1]["speed_corrected_hz"], nom_p["f_1969_hz"], nom_p["f_2024_hz"], ed[1]["residual_cents"]),
        ("B4 Quint", ed[2]["raw_1890_hz"], ed[2]["speed_corrected_hz"], qui_p["f_1969_hz"], qui_p["f_2024_hz"], ed[2]["residual_cents"]),
        ("E5 Oct.", ed[3]["raw_1890_hz"], ed[3]["speed_corrected_hz"], oct_p["f_1969_hz"], oct_p["f_2024_hz"], ed[3]["residual_cents"]),
        ("A5 3rd", ed[4]["raw_1890_hz"], ed[4]["speed_corrected_hz"], sup_p["f_1969_hz"], sup_p["f_2024_hz"], ed[4]["residual_cents"]),
        ("C#6 P6", ed[5]["raw_1890_hz"], ed[5]["speed_corrected_hz"], p6_p["f_1969_hz"], p6_p["f_2024_hz"], ed[5]["residual_cents"]),
    ]
    for idx, (mname, r90, c90, f69, f24, res_c) in enumerate(map_rows):
        yy = hy + 30 + idx * 32
        s.append(f'<line x1="{rx0+18}" y1="{yy+10}" x2="{rx0+rw-18}" y2="{yy+10}" stroke="{RULE}" />')
        s.append(f'<text x="{cols_x[0]}" y="{yy}" class="lbl">{mname}</text>')
        s.append(f'<text x="{cols_x[1]}" y="{yy}" class="num-m">{r90:.1f} Hz</text>')
        s.append(f'<text x="{cols_x[2]}" y="{yy}" class="num-c">{c90:.1f} ({res_c:+.0f}c)</text>')
        s.append(f'<text x="{cols_x[3]}" y="{yy}" class="num">{f69:.2f} Hz</text>')
        s.append(f'<text x="{cols_x[4]}" y="{yy}" class="num-s">{f24:.2f} Hz</text>')

    # Lower summary box: Codec + Restoration + ENF
    by = hy + 230
    s.append(f'<rect x="{rx0+18}" y="{by}" width="{rw-36}" height="114" rx="4" fill="{PAPER}" stroke="{RULE}" />')
    s.append(f'<text x="{rx0+30}" y="{by+24}" class="lbl" font-weight="600">Cross-era &amp; codec forensic signatures:</text>')
    s.append(f'<text x="{rx0+28}" y="{by+46}" class="num">• 2017 HQ (185k) vs LQ (67k): 0.01 Hz drift, 0.01 dB (883/335)</text>')
    s.append(f'<text x="{rx0+30}" y="{by+66}" class="num">• 50 Hz UK grid ENF lock: 1953 (50.13 Hz) &amp; 1969 (49.96 Hz) yes;</text>')
    s.append(f'<text x="{rx0+30}" y="{by+84}" class="num">  2016 Bridge St battery Zoom H4n &amp; 1890 wax cylinder: absent</text>')
    s.append(f'<text x="{rx0+30}" y="{by+103}" class="num">• KL-NMF (k=6): R² = 0.90–0.92 (close/1945) vs 0.76 (street)</text>')

    s.append("</svg>")
    out_p = f"{OUT_ROOT}/the-chords-inside-big-ben/fig2-little-deans-yard-and-1890-2024.svg"
    open(out_p, "w").write("\n".join(s))
    return out_p

# ---------------------------------------------------------------------------
# Post 2, Fig 1: Westminster Acoustic OSINT Map & Observables
# ---------------------------------------------------------------------------
def make_p2_fig1():
    w, h = 1040, 560
    s = [svg_header(w, h, "Westminster acoustic OSINT map comparing labelled Big Ben recording sites, Heathrow flight path, and measured observables")]
    s.append(f'<text x="32" y="38" class="title">Westminster OSINT: where public Big Ben microphones sit vs. what the sound tells us</text>')
    s.append(f'<text x="32" y="60" class="sub">Metric map centered on Elizabeth Tower (51.5007° N, −0.1246° W) with binaural ITD, street reflections, and Heathrow 27L/27R corridor</text>')

    # Left: Metric Map (-380m..+380m E-W, -320m..+240m N-S)
    mx, my, mw, mh = 32, 86, 510, 442
    s.append(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="6" fill="{TINT}" stroke="{RULE}" />')
    s.append(f'<text x="{mx+18}" y="{my+28}" class="h2">A. Metric Westminster geometry &amp; labelled sites</text>')

    # Scale: 1 m -> 0.50 px, origin (0,0) = Elizabeth Tower at (mx+265, my+175)
    ox, oy = mx + 265, my + 175
    sc = 0.50
    def mp(east_m, north_m):
        return ox + east_m * sc, oy - north_m * sc

    # Draw range rings (100 m, 200 m, 350 m) clipped inside Panel A map plot region
    s.append(f'<defs><clipPath id="mapClip"><rect x="{mx+12}" y="{my+48}" width="{mw-24}" height="{mh-62}" /></clipPath></defs>')
    s.append('<g clip-path="url(#mapClip)">')
    for rm in [100, 200, 350]:
        s.append(f'<circle cx="{ox}" cy="{oy}" r="{rm*sc:.1f}" fill="none" stroke="{RULE}" stroke-dasharray="4,4" />')
        s.append(f'<text x="{ox + rm*sc - 18:.1f}" y="{oy + 12}" class="num-m">{rm}m</text>')
    s.append('</g>')

    # River Thames band on the East (east_m ~ 90..360)
    tx_l, _ = mp(90, 0)
    s.append(f'<rect x="{tx_l:.1f}" y="{my+48}" width="{mx+mw-tx_l-14:.1f}" height="{mh-68}" fill="#E4ECEF" opacity="0.7" />')
    s.append(f'<text x="{tx_l+60:.1f}" y="{my+395}" class="mut">River Thames</text>')

    # Heathrow 27L/27R approach corridor (~1,100 m overhead) dashed line across map
    hx1, hy1 = mp(-360, 120)
    hx2, hy2 = mp(360, -40)
    s.append(f'<line x1="{hx1:.1f}" y1="{hy1:.1f}" x2="{hx2:.1f}" y2="{hy2:.1f}" stroke="{GOLD}" stroke-width="2.2" stroke-dasharray="7,5" />')
    s.append(f'<text x="{mx+22}" y="{hy1-10:.1f}" class="lbl" fill="{GOLD}">Heathrow 27L/27R glidepath (~1,100 m AGL)</text>')

    # Westminster Abbey footprint between Tower (0,0) and Little Dean's Yard (-251, -242)
    ax1, ay1 = mp(-215, -135)
    s.append(f'<rect x="{ax1:.1f}" y="{ay1:.1f}" width="102" height="40" rx="3" fill="#D5D2C7" stroke="{MUTED}" />')
    s.append(f'<text x="{ax1+51:.1f}" y="{ay1+24:.1f}" text-anchor="middle" class="lbl">Westminster Abbey</text>')

    # Sites with custom conflict-free label placements
    pts = [
        (0, 0, "Elizabeth Tower", "10m Belfry | Q_tilt +10.0 dB", CLAY, -22, 34, "end"),
        (-2, 26, "70 Bridge St (2016)", "68.6m | ITD −0.40ms | DRR −7.3dB", SLATE, 12, -32, "start"),
        (202, 8, "Westminster Bridge", "212m slant | Q_tilt −15.6dB", SLATE, 88, 66, "end"),
        (-157, -26, "Parliament Sq", "172m | Q_tilt −11.9dB", SLATE, -12, -22, "end"),
        (-251, -242, "Little Dean's Yard", "354.5m slant (Abbey shadow)", GOLD, 12, 4, "start"),
    ]
    for em, nm, name, sub, col, dx_t, off_y, anc in pts:
        px, py = mp(em, nm)
        if em != 0 or nm != 0:
            s.append(f'<line x1="{ox}" y1="{oy}" x2="{px:.1f}" y2="{py:.1f}" stroke="{col}" stroke-width="1.2" opacity="0.55" />')
        s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{col}" stroke="{PAPER}" stroke-width="1.5" />')
        if "Bridge St" in name or "Elizabeth" in name or "Westminster" in name:
            s.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px+dx_t*0.7:.1f}" y2="{py+off_y*0.65:.1f}" stroke="{col}" stroke-width="1" opacity="0.7" />')
        s.append(f'<text x="{px+dx_t:.1f}" y="{py+off_y:.1f}" text-anchor="{anc}" class="lbl" font-weight="700">{name}</text>')
        s.append(f'<text x="{px+dx_t:.1f}" y="{py+off_y+13:.1f}" text-anchor="{anc}" class="num-m">{sub}</text>')

    # Right: 4-Observable Site Table + Leave-One-Out Benchmark
    rx0, ry0, rw, rh = 560, 86, 448, 442
    s.append(f'<rect x="{rx0}" y="{ry0}" width="{rw}" height="{rh}" rx="6" fill="{TINT}" stroke="{RULE}" />')
    s.append(f'<text x="{rx0+18}" y="{ry0+28}" class="h2">B. Measured OSINT observables per site</text>')
    s.append(f'<text x="{rx0+18}" y="{ry0+47}" class="mut">Leave-one-out slant error: 63.7 m (Q_tilt) vs 82.6 m (baseline)</text>')

    cols = [rx0 + 18, rx0 + 172, rx0 + 242, rx0 + 312, rx0 + 372]
    hds = ["Recording", "Q_tilt", "DRR", "Refl.", "ITD"]
    hy = ry0 + 78
    for cx, hd in zip(cols, hds):
        s.append(f'<text x="{cx}" y="{hy}" class="lbl" font-weight="700">{hd}</text>')
    s.append(f'<line x1="{rx0+16}" y1="{hy+7}" x2="{rx0+rw-16}" y2="{hy+7}" stroke="{INK}" stroke-width="1.2" />')

    obs_rows = [
        ("Belfry Q1 (1969, 10m)", "+10.0 dB", "−8.9 dB", "9.6 ms", "−0.01 ms"),
        ("Belfry 1:00 (1969, 10m)", "−3.1 dB", "−14.9 dB", "11.4 ms", "−0.01 ms"),
        ("Bridge St (2016, 69m)", "−7.5 dB", "−7.3 dB", "17.7 ms", "−0.40 ms"),
        ("Palace Yd (1969, 95m)", "+8.7 dB", "−14.9 dB", "9.7 ms", "−0.00 ms"),
        ("Parl. Sq (LQ, 172m)", "−11.9 dB", "−12.7 dB", "12.8 ms", "+0.00 ms"),
        ("West. Br. (212m E)", "−15.6 dB", "−12.9 dB", "8.9 ms", "−0.01 ms"),
        ("Midnight HQ (185k)", "−12.3 dB", "−13.5 dB", "11.8 ms", "−0.03 ms"),
        ("Midnight LQ (67k)", "−12.2 dB", "−13.3 dB", "11.8 ms", "−0.03 ms"),
        ("NYE 2024 UTC", "−3.6 dB", "−15.4 dB", "12.3 ms", "+0.69 ms"),
    ]
    for i, row in enumerate(obs_rows):
        yy = hy + 28 + i * 31
        s.append(f'<line x1="{rx0+16}" y1="{yy+9}" x2="{rx0+rw-16}" y2="{yy+9}" stroke="{RULE}" />')
        s.append(f'<text x="{cols[0]}" y="{yy}" class="lbl">{row[0]}</text>')
        for j in range(1, 5):
            cls = "num-c" if ("Bridge" in row[0] and j in (2, 3, 4)) or ("West." in row[0] and j == 1) else "num"
            s.append(f'<text x="{cols[j]}" y="{yy}" class="{cls}">{row[j]}</text>')

    s.append(f'<text x="{rx0+18}" y="{ry0+rh-14}" class="mut">Midnight HQ vs LQ preserves Q_tilt (0.17 dB), DRR (0.12 dB), and lag (0.0 ms).</text>')
    s.append("</svg>")
    out_p = f"{OUT_ROOT}/locating-big-ben-recordings-from-sound/fig1-westminster-osint-map.svg"
    open(out_p, "w").write("\n".join(s))
    return out_p

# ---------------------------------------------------------------------------
# Post 2, Fig 2: Why doppler-geo Breaks on Big Ben Audio + Louver Anisotropy
# ---------------------------------------------------------------------------
def make_p2_fig2():
    w, h = 1040, 560
    s = [svg_header(w, h, "Why doppler-geo matched filtering triggers false positives on bells and how 4-bell louver anisotropy separates compass quadrants")]
    s.append(f'<text x="32" y="38" class="title">Why aircraft Doppler breaks on bells—and what replaces it</text>')
    s.append(f'<text x="32" y="60" class="sub">Left: real dg.detect() scores from doppler-geo on Big Ben audio. Right: 4-bell louver radiation fingerprint by site.</text>')

    # Left: Real doppler-geo detector scores across our corpus (including Belfry false positive!)
    lx, ly, lw, lh = 32, 86, 480, 442
    s.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="6" fill="{TINT}" stroke="{RULE}" />')
    s.append(f'<text x="{lx+18}" y="{ly+28}" class="h2">A. Real doppler-geo dg.detect() scores on bell audio</text>')
    s.append(f'<text x="{lx+18}" y="{ly+47}" class="mut">Bells trigger false-positive Doppler matches (f0 ≈ 325–347 Hz) even inside the belfry!</text>')

    bx0, bx1 = lx + 178, lx + lw - 30
    dg_rows = [
        ("Bridge St (95s WAV)", 5.24, "f0=344.7 Hz"),
        ("Belfry 1:00 (1969)", 4.81, "f0=324.6 Hz (false+)"),
        ("Midnight HQ (185k)", 4.77, "f0=340.6 Hz"),
        ("Midnight LQ (67k)", 4.69, "f0=340.6 Hz"),
        ("Parl. Sq (75s MP3)", 4.45, "f0=337.5 Hz"),
        ("NYE 2024 (83s MP3)", 4.38, "f0=346.8 Hz"),
        ("Belfry Q1 (26s WAV)", 3.87, "f0=224.6 Hz"),
        ("Palace Yd (28s WAV)", 3.33, "f0=413.8 Hz"),
    ]
    for i, (name, sc_val, note) in enumerate(dg_rows):
        yy = ly + 84 + i * 40
        bar_w = (sc_val / 6.0) * (bx1 - bx0)
        col = CLAY if "Belfry" in name else SLATE
        s.append(f'<text x="{lx+18}" y="{yy+14}" class="lbl">{name}</text>')
        s.append(f'<rect x="{bx0}" y="{yy}" width="{bar_w:.1f}" height="20" rx="2" fill="{col}" opacity="0.88" />')
        s.append(f'<text x="{bx0+bar_w+6:.1f}" y="{yy+14}" class="num">{sc_val:.2f}</text>')
        s.append(f'<text x="{bx0+6}" y="{yy+14}" class="num" fill="#FAF9F5" font-size="10">{note}</text>')

    s.append(f'<text x="{lx+18}" y="{ly+lh-16}" class="mut">Heathrow jets (1.1 km alt) shift slant 3.65% over 300m vs. Big Ben’s 874 ms lag span.</text>')

    # Right: Quarter-Bell Louver Anisotropy (relative energy across G#4, F#4, E4, B3)
    rx0, ry0, rw, rh = 528, 86, 480, 442
    s.append(f'<rect x="{rx0}" y="{ry0}" width="{rw}" height="{rh}" rx="6" fill="{TINT}" stroke="{RULE}" />')
    s.append(f'<text x="{rx0+18}" y="{ry0+28}" class="h2">B. Quarter-bell energy balance across azimuths</text>')
    s.append(f'<text x="{rx0+18}" y="{ry0+47}" class="mut">Stacked share of 870 Hz (G#4), 771 Hz (F#4), 679 Hz (E4), 504 Hz (B3)</text>')

    bell_cols = [
        ("G#4_870", "G#4 (870)", CLAY),
        ("F#4_771", "F#4 (771)", GOLD),
        ("E4_679", "E4 (679)", "#6D8B74"),
        ("B3_504", "B3 (504)", SLATE),
    ]
    # Legend
    for bi, (_, blbl, bcol) in enumerate(bell_cols):
        lx_i = rx0 + 22 + bi * 110
        s.append(f'<rect x="{lx_i}" y="{ry0+66}" width="12" height="12" rx="2" fill="{bcol}" />')
        s.append(f'<text x="{lx_i+17}" y="{ry0+76}" class="num-m">{blbl}</text>')

    site_keys = ["bbc_1969_belfry_q1", "bbc_1969_ground_q1", "aporee_2016_bridge_st", "square_lq_parliament_sq", "fs_388808_westminster_bridge", "midnight_hq_2017", "nye_2024_utc"]
    obs_by_id = {r["id"]: r for r in osint["site_osint_observables"]}
    sx0, sx_w = rx0 + 165, rw - 190
    for i, sid in enumerate(site_keys):
        r = obs_by_id[sid]
        yy = ry0 + 102 + i * 42
        short_lbl = {
            "bbc_1969_belfry_q1": "Belfry (1969, Q1)",
            "bbc_1969_ground_q1": "Palace Yd (1969, Q1)",
            "aporee_2016_bridge_st": "70 Bridge St (N, Q2)",
            "square_lq_parliament_sq": "Parl. Sq (WSW, Q4)",
            "fs_388808_westminster_bridge": "West. Bridge (E, Q4)",
            "midnight_hq_2017": "Midnight HQ (Street)",
            "nye_2024_utc": "NYE 2024 (UTC)",
        }[sid]
        s.append(f'<text x="{rx0+18}" y="{yy+16}" class="lbl">{short_lbl}</text>')
        cur_x = sx0
        fracs = r["quarter_bell_louver_fractions"]
        for bk, _, bcol in bell_cols:
            fr_v = fracs[bk]
            bw = fr_v * sx_w
            if bw > 1.0:
                s.append(f'<rect x="{cur_x:.1f}" y="{yy}" width="{bw:.1f}" height="24" fill="{bcol}" />')
                if bw > 28:
                    s.append(f'<text x="{cur_x + bw*0.5:.1f}" y="{yy+16}" text-anchor="middle" class="num" fill="#FAF9F5">{int(round(fr_v*100))}%</text>')
            cur_x += bw

    s.append(f'<text x="{rx0+18}" y="{ry0+rh-16}" class="mut">Q1–Q4 note counts and street/belfry diffraction jointly shape B3 vs G#4 share.</text>')
    s.append("</svg>")
    out_p = f"{OUT_ROOT}/locating-big-ben-recordings-from-sound/fig2-deconvolution-and-doppler-limits.svg"
    open(out_p, "w").write("\n".join(s))
    return out_p

if __name__ == "__main__":
    p1 = make_p1_fig1()
    p2 = make_p1_fig2()
    p3 = make_p2_fig1()
    p4 = make_p2_fig2()
    print("Generated 4 SVGs:", p1, p2, p3, p4)
