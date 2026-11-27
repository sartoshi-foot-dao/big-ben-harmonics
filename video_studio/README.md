# Big Ben Dual-Format Video Studio

Deterministic Python studio producing synchronized `16:9` (`1920x1080`) and native vertical `9:16` (`1080x1920`) videos directly from measured modal frequencies and acoustic propagation models—no prompt-by-prompt guesswork.

## Files

- `brief.md` — Editorial brief and X-bookmark design rules (`@0xwhrrari`, `@rincyXIII`, `@hnshah`).
- `shotlist.md` — 5-act, `33.60 s` (`1,008` frames at `30 fps`) timeline.
- `synth_audio.py` — Pure `48 kHz` modal bell synthesizer driven by measured 1969 BBC eigenmodes and Westminster Abbey diffraction curves; emits `build/master_audio.wav` and `build/events.json`.
- `studio_engine.py` — Pure stateless `render_frame(t, aspect)` compositor piping RGB frames to `ffmpeg` for H.264/AAC MP4 + WebP posters.
- `verify_sync.py` — Automated frame-accurate audio/visual onset alignment check (`max_offset < 1.0 frame`) and contact-sheet generator.

## Reproduce

```bash
python3 synth_audio.py
python3 studio_engine.py --out-dir /path/to/public/media/big-ben
python3 verify_sync.py --media-dir /path/to/public/media/big-ben
```
