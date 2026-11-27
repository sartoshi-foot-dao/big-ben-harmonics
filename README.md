# The Chords Inside Big Ben

Modal acoustics, 14-second virtual-pitch evolution, Westminster Abbey diffraction modelling, and 1890–2024 archival forensics of Big Ben and the Westminster Quarters.

- **Research note & interactive harmonic explorer:** [The chords inside Big Ben](https://zidingz.com/the-chords-inside-big-ben/) · [Standalone modal & Abbey propagation simulator](https://zidingz.com/interactive/big-ben-harmonics/)
- **Playable Westminster Quarters score & listening essay:** [Big Ben, from Little Dean’s Yard](https://zidingz.com/big-ben-from-little-deans-yard/)
- **Single-file WebGL 2 cinema presentation:** [Fourteen Seconds of Bronze](https://zidingz.com/interactive/fourteen-seconds-of-bronze/)
- **Technical preprint (PDF):** [preprint.pdf](https://zidingz.com/research/big-ben-harmonics/preprint.pdf) (also included at `./preprint.pdf` and `./preprint.tex`)

---

## Measured chord & partial summary (1969 BBC WAVs `07002139`–`07002142`)

| Bell or partial | Measured frequency | `T60` / split | Musical pitch & role |
| --- | --- | --- | --- |
| Great Bell hum | `200.26 Hz` | `T60 = 29.03 s` | `G3 (+37c)`, ratio `0.597`; sustains late chord |
| Great Bell strike doublet | `332.735 / 335.440 Hz` | `2.705 Hz` split (`0.370 s` swell) | `E4` nominal; modal splitting creates signature warble |
| Great Bell upper quint doublet | `494.828 / 499.666 Hz` | `4.837 Hz` split (`T60 = 17.33 s`) | `B4`; becomes dominant salience (`1.00`) by `8–14 s` |
| Great Bell super-third overtone | `883.34 Hz` | `T60 = 7.96 s` (`-41.3 dB` by `8 s`) | `A5`; pulls `0–2 s` Terhardt virtual pitch toward `A4` (`440.6 Hz`) |
| Quarter bells (`×4`, John Warner 1856) | `869.75, 770.97, 678.99, 505.14 Hz` | Minor-third tierces `0.590–0.593` | `G♯4, F♯4, E4, B3` E-major Westminster Quarters |

## Key findings

1. **14-second chord evolution:** On impact (`0–2 s`), `883.34 Hz` (`A5`) pulls Terhardt virtual pitch toward a dissonant `A4` (`0.68` salience) against `E`. Because high partials decay rapidly while the `200.26 Hz` hum (`T60 = 29.03 s`) and `499.67 Hz` fifth (`T60 = 17.33 s`) linger, perceived salience flips by `8–14 s` to `B4` (`1.00`), `E3` (`0.70`), and `G3` (`0.56`).
2. **Westminster Abbey diffraction (`354.5 m` southwest along `226.0°`):** Combined Maekawa barrier loss over the Abbey gable and ISO 9613-1 air absorption attenuate `5 kHz` bronze clang by `29.4 dB` at `28°C` and `49.5 dB` at `0°C` mist, matching empirical `10.87 dB` HF rolloff between belfry (`07002139`) and Palace Yard (`07002154`) and delaying winter arrival by `51.0 ms`.
3. **1890–2024 archival forensics:** Dimensionless modal-comb matching on the hour strikes (`88–141 s`) of the July 16, 1890 Edison wax cylinder recovers a `0.9464x` (`-0.95` semitone) slow playback offset, restoring `200.0`, `331.9`, `500.7`, and `677.3 Hz`. Mains-powered 1953 (`50.132 Hz`) and 1969 (`49.964 Hz`) tapes track the 50 Hz UK grid ENF; battery-powered 2016 Bridge Street WAVs show zero grid drift.

## Repository layout

- `code/` — Analysis, figure generation, and scene scripts (`analyze_big_ben.py`, `refine_quarters_and_enf.py`, `generate_svgs.py`, `paint_scene.py`). Set `BIG_BEN_CORPUS_DIR` and `BIG_BEN_OUT_DIR` to override corpus/output paths.
- `out/` — Verified measurement JSON outputs (`corpus_manifest.json`, `quarters_and_bells_modal.json`, `little_deans_yard_acoustics.json`, `forensic_1890_2024.json`).
- `independent/` — Blind independent recomputation (`recompute.py`, `RECOMPUTED.json`, `REVIEW.md`), confirming 1890 wax-cylinder speed agreement to `0.0006` (`0.9458` vs `0.9464`).
- `video_studio/` — Deterministic `48 kHz` modal bell synthesizer (`synth_audio.py`), dual-format `16:9` / `9:16` frame renderer (`studio_engine.py`), and sync verifier (`verify_sync.py`).
- `INPUTS_sha256.txt` — SHA-256 manifest of the 20-recording corpus (raw third-party audio binaries are omitted; source URLs and hashes are listed in `out/corpus_manifest.json`).
- `preprint.tex` / `preprint.pdf` — Technical paper source and compiled PDF.

## Reproduce

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Re-run blind independent verification on existing out/ measurements
python3 independent/recompute.py

# 2. Re-run full corpus analysis (requires audio corpus in BIG_BEN_CORPUS_DIR)
BIG_BEN_CORPUS_DIR=./corpus BIG_BEN_OUT_DIR=./out python3 code/analyze_big_ben.py

# 3. Synthesize and render deterministic 16:9 & 9:16 modal summary videos
python3 video_studio/synth_audio.py
python3 video_studio/studio_engine.py --out-dir ./video_studio/build
python3 video_studio/verify_sync.py --media-dir ./video_studio/build
```

## Licence

MIT License — see [LICENSE](LICENSE). Builds on archival BBC Sound Effects (`07002139`–`07002142`, `07002154`), public Freesound / Radio Aporee captures, and Dr. Bill Hibbert's 2008 modal bell research ([oro.open.ac.uk/25305](https://oro.open.ac.uk/25305/)).
