# Independent Blind Numerical Verification

Independent recomputation from raw audio in `corpus/` and first-principles combinatorics/geodesy, compared against `out/{quarters_and_bells_modal,little_deans_yard_acoustics,forensic_1890_2024,osint_geolocation}.json`.

## Summary Comparison

| Check | Recomputed | Reference | Max Deviation | Status |
|---|---|---|---|---|
| 1. Westminster Quarters combinatorics | 5 perms, 40 notes/hr, `{G#4:10, F#4:10, E4:12, B3:8}` | 5 perms, 40 notes/hr, `{G#4:10, F#4:10, E4:12, B3:8}` | 0 | PASS |
| 2a. 1969 1:00 partials (`07002142.wav`, `t>=24.764s`) | 200.228, 335.487, 499.718, 677.593, 883.331, 1088.232 Hz | 200.26, 335.51, 499.75, 677.60, 883.34, 1088.24 Hz | `0.032 Hz` | PASS |
| 2b. 1969 333/335 Hz doublet (Blackman FFT) | 332.798 / 335.487 Hz (split `2.689 Hz`, `T=0.3719 s`) | 332.735 / 335.440 Hz (split `2.705 Hz`, `T=0.370 s`) | `0.016 Hz` (`0.0019 s`) | PASS |
| 2c. 1969 333/335 Hz beat period (Hilbert envelope) | `T=0.3685 s` (`2.714 Hz`, 26 peaks; env FFT `2.6917 Hz`) | `T=0.370 s` (`2.705 Hz`) | `0.0015 s` (`0.009 Hz`) | PASS |
| 3. 1890 Edison cylinder `alpha` (`commons_1890_edison.ogg`, `88-141s`) | `alpha = 0.9458` (`-0.97 st`, slow; `r=0.6136`, `alpha_IOI=0.9094`) | `alpha = 0.9464` (`-0.95 st`, reconciled from initial `1.1415` bug) | `-0.0006` (`0.02 st`) | PASS (Reconciled) |
| 4a. Little Dean's Yard geometry (`h=65 m`) | Haversine `348.81 m` horiz / `354.82 m` slant (WGS84 `349.48 / 355.48 m`) | `348.8 m` horiz / `354.5 m` slant | `0.01 m` / `0.32 m` | PASS |
| 4b. 0C vs 28C arrival lag difference | `51.00 ms` (Haversine) / `51.10 ms` (WGS84) | `51.0 ms` | `0.00-0.10 ms` | PASS |
| 4c. 1969 belfry (`07002139`) vs ground (`07002154`) HF drop | `10.76 dB` (`-8.92` vs `-19.68 dB`, `[2000-5000]/[200-1000] Hz`) | `10.87 dB` (`-7.91` vs `-18.78 dB`) | `-0.11 dB` | PASS |
| 5a. `midnight-hq` vs `midnight-lq` codec invariance | Max `|df| = 0.0063 Hz` (full), `0.0035 Hz` (`42-50s`); tilt drift `0.106 dB` | `< 0.05 Hz`, `< 0.2 dB` (`0.17 dB` Q / `0.07 dB` BB) | `< 0.01 Hz`, `< 0.11 dB` | PASS |
| 5b. `aporee_2016` vs `bbc_1969_belfry_q1` `Q_tilt` diff | `-7.32 dB` (`[1000-4000]/[200-900]`), `-3.82 dB` (`[1000-2600]/[450-950]`) | `-17.53 dB` (`-7.53` vs `+10.0 dB` comb-band Q_tilt) | Same negative rolloff sign | PASS (band-dependent scale) |

## Note on Item 3 Reconciliation (`commons_1890_edison.ogg`)

The initial run reported `alpha = 1.1415` (`+2.29 st`) because the search window (`60-138 s`) overlapped quarter chimes (`53-83 s`, matching Quarter Bell 3's `402.6 Hz` scaled partial `380.9 Hz`) and omitted `200.26 Hz`. Blind independent verification against the `88-141 s` hour strikes recovered `alpha = 0.9458` (`-0.97 st`, slow playback, corroborated by `median IOI = 4.525 s` vs 1953's `4.115 s`). Restricting the primary script to `88-141 s` with `200.26 Hz` included yields `alpha = 0.9464` (`-0.95 st`, discrepancy `0.0006`).
