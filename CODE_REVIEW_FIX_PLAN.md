# Current Code Review Fix Plan

This document tracks the engineering fixes from the project owner review. The first implementation pass focuses on correctness, train/eval credibility, service observability, and local developer workflow.

## P0: Correctness And Model Credibility

1. Fix control client response handling.
   - `ServiceClient.connect()` consumes and stores the initial greeting status as `initial_status`.
   - `ServiceClient.command()` now waits for a matching `ack`, command-specific `status`, `pong`, or `error`.
   - Broadcast messages received before the command response are skipped.
   - Connection closed and timeout cases now raise clear exceptions.

2. Prevent train/validation leakage.
   - Positive word blocks are split before augmentation.
   - User-recorded samples are split by source item when enough samples exist.
   - Negative feature streams are split into non-overlapping continuous regions before window sampling.
   - Fixed seeds remain deterministic.

3. Improve model selection metrics.
   - Validation now reports precision, recall, F1, accuracy, and false-positive rate.
   - Best checkpoint selection uses F1 at threshold `0.5` instead of accuracy.
   - Training writes `artifacts/model_output/metrics.json` with class balance, threshold scan, recommended threshold, and recommended metrics.

## P1: Service, Dependencies, And Safety

1. Add service health state.
   - `status` now includes `ready`, `worker_alive`, and `error`.
   - The worker marks the service ready only after detector/model load succeeds.
   - `start` returns an error when the detector is not ready.

2. Clean dependency groups.
   - `pyproject.toml` extras now include `runtime`, `train`, `vad`, `edge`, `export`, and `dev`.
   - `requirements.txt` is ASCII-only and includes the dependencies used by current CLI commands.
   - `edge-tts` is included for `wakeup gen-voices`.

3. Harden download and extraction.
   - `download()` supports optional SHA256 verification.
   - Existing cached files are verified when a checksum is configured.
   - Bad checksum files are removed before raising.
   - TTS tar extraction uses `safe_extract_tar()` to reject path traversal.

## P2: Maintainability

1. Move openWakeWord support model downloads to a project cache.
   - Default cache: `artifacts/oww_models`.
   - A best-effort mirror into openWakeWord's package resource directory is kept for compatibility with library internals.

2. Validate config values.
   - Numeric ranges and enum fields are checked when config loads.
   - Unknown config keys still fail fast.

3. Local validation commands.
   - Preferred project environment:
     `D:\APP\Anaconda3\envs\wakeup\python.exe`
   - Compile check:
     `D:\APP\Anaconda3\envs\wakeup\python.exe -m compileall -q src tests main.py`
   - Tests:
     `D:\APP\Anaconda3\envs\wakeup\python.exe -m pytest -q`
   - CLI smoke test:
     `D:\APP\Anaconda3\envs\wakeup\python.exe -m wakeup.cli --help`

## P3: Voice Auto-Generation (No Recording Required)

The wake word audio is fully synthesized; manual recording is optional.

1. `wakeup train` auto-synthesizes positive "小元" samples via sherpa-onnx
   `vits-zh-aishell3` (174 speakers, random speed). No recording needed.
2. `wakeup train --gen-voices` (new) additionally expands positives with Edge TTS
   multi-voice synthesis before training. Network failure is non-fatal: it logs a
   warning and continues with existing samples.
3. `wakeup gen-voices` / `wakeup record` outputs are auto-merged on the next
   `wakeup train`. `record` is optional few-shot personalization only.
4. README documents the auto-generation workflow and marks recording as optional.

## P4: Split Pipeline Into Cached Stages

Re-running training no longer repeats the slow data/feature work.

1. `wakeup prepare` (new) — stage one (slow): synthesize/record samples, download
   negatives, extract and cache all positive features to `.npy`. Re-run only when
   samples change. Supports `--gen-voices` / `--voices-count` / `--force-tts` /
   `--force-features`.
2. `wakeup fit` (new) — stage two (fast): load cached features, train, export ONNX.
   Re-run this for hyperparameter tuning. Supports `--epochs` override for quick
   trials, plus `--export-tf` / `--no-simplify` / `--device`.
3. `wakeup train` remains the one-shot `prepare` + `fit`; gained `--force-features`
   and `--epochs`.
4. Edge TTS and user-recording features are now cached to `edge_features.npy` /
   `user_features.npy` (previously re-extracted every run). `fit` errors with a
   clear hint if the positive/negative caches are missing.
5. `pipeline.run_training` now delegates to `prepare_data()` + `fit()`; both are
   exported from `wakeup.training`.

## P5: Real-Voice Sensitivity (Domain Gap)

Symptom: training F1 0.996 but live `wakeup listen --show-score` ~0.014 on a real
voice; the wake word almost never fires.

Diagnosis (measured, not guessed):

- Inference is not buggy. A TTS "小元" embedded in a continuous stream scores
  mean ~0.5 / max ~0.64 through openWakeWord streaming.
- The validation F1 is over-optimistic: it is computed on the offline windows
  built by `dataset.augment_positives` (word block over high-energy openWakeWord
  negative background), which score higher than real streaming. So the
  recommended threshold (0.85) never fires in practice; real ceiling is ~0.64.
- Real human voice (~0.014) vs TTS (~0.5) is a TTS-to-real domain gap. Lowering
  the threshold cannot bridge a ~35x gap.

Fix 1 — audio-level augmentation (`data/augment.py`):

- Adds colored noise (random SNR), synthetic reverb, random gain, and mic-style
  band-pass to each TTS clip before feature extraction.
- Applied to the word audio only (silence padding stays constant) so downstream
  `word_blocks` boundary detection still works.
- Controlled by `data.audio_augment` / `data.audio_augment_variants` (default on,
  2 variants). Re-extract with `wakeup prepare --force-features`.

Fix 2 — few-shot real recordings (already supported): `wakeup record` then
`prepare --force-features` + `fit`. `train.real_sample_prob` up-weights real word
blocks during augmentation. Recommended when augmentation alone is insufficient.

Evaluate real-world triggering with streaming scores, not the validation F1.

## P6: Detector Misses (Cold Buffer + VAD Trigger Gate)

Symptom: even after recording 40 real samples (model scores them 0.98), live
recognition was intermittent ("有时识别有时不识别"). `--debug` (model every frame)
hit 1.0; normal mode missed.

Root causes (traced on the user's own recordings through the full detector):

1. Cold streaming buffer. The detector skipped the model during silence and, on
   speech onset, reset openWakeWord and replayed only `preroll_frames` (~1.3s).
   openWakeWord needs ~2s of continuous audio to fill its 16-frame embedding
   window, so the score stayed 0.000 during the short word. Longer preroll did
   not help (replay after reset never refilled in time).
2. VAD trigger gate vs score latency. openWakeWord's score peaks ~1s AFTER the
   word (window must fill), but VAD fires during the word and ends before the
   peak — so gating triggering on VAD dropped >90% of true hits.

Fix (`service/detector.py`): run the model every frame while listening (buffer
always warm; ~1-2ms/frame) and trigger on model score + threshold + cooldown
only. VAD is no longer used for gating (kept for `wakeup listen --debug`).
Verified: the user's 40 recordings now fire 40/40 through `detector.process`,
mean score 0.984. `preroll_frames` / `hangover_frames` are deprecated (kept for
config compatibility). Added `wakeup listen --debug` (per-frame VAD/score/mic-RMS)
to diagnose VAD-vs-model issues.

## Verification (this pass)

- `D:\APP\Anaconda3\envs\wakeup\python.exe -m compileall -q src tests`: pass.
- `D:\APP\Anaconda3\envs\wakeup\python.exe -m pytest -q`: 14 passed.
- `wakeup --help` lists `prepare` and `fit`; `fit --help` shows `--epochs`.
- End-to-end: `wakeup train --gen-voices` produced `models/xiaoyuan.onnx`
  (recommended threshold 0.85, val F1 0.996). `wakeup prepare` then `wakeup fit`
  reproduces it from cache (cache load ~1s, no re-synthesis/extraction).
- Fixed a corrupted log line in `data/_download.py` (`"?? %s -> %s"` -> `"Downloading %s -> %s"`).
- Fixed ONNX export crash on Windows GBK consoles: torch 2.x dynamo exporter
  prints emoji (✅) that cp936 cannot encode. `_logging.py` now forces UTF-8 on
  stdout/stderr (`errors="replace"`, also fixes Chinese mojibake) and
  `export.py` passes `verbose=False`.

## Notes

- Some Windows PowerShell sessions may display UTF-8 Chinese text as mojibake. The source files themselves are UTF-8.
- Older Anaconda builds on this machine do not support `conda run`; activate the environment first or call the environment Python directly.
