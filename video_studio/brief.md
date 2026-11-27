# Video Studio Architecture: Social Videos from Research Posts

- **Source thesis:** Big Ben does not ring one note (`E3`); inside the Elizabeth Tower belfry, thick cast bronze builds an evolving chord (`A4 441 Hz` clash at impact -> `2.71 Hz` E4 doublet swell -> `200 Hz G3 + 500 Hz B4` hum by `8–14 s`), while Westminster Abbey filters `20–49 dB` of upper clang before the sound reaches Little Dean's Yard (`354.5 m`).
- **Studio principles (X bookmark stack + `one-letter`):**
  1. `@0xwhrrari` ("build a studio, not a prompt"): Deterministic parameterized scenes rendered from measured JSON + `events.json` timestamps — never one-off video prompting.
  2. `@rincyXIII` (dual-format distribution): One canonical `synth_audio.py` master + native `16:9` (`1920x1080`) landscape composition AND separate native `9:16` (`1080x1920`) vertical layout with vertical safe zones (never cropped `16:9`).
  3. `@hnshah` (comprehension rules): Immediate concrete sound hook (`t = 0.50 s`), one thesis per act, readable hold plateaus, zero simultaneous competing widgets.
  4. `one-letter` engine contract: Pure stateless `render_frame(t, spec)` seekable to any millisecond, zero cumulative simulation state, `< 1 frame` audio-visual onset sync verification (`verify_sync.py`).
