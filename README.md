# edge-vlm-scene-copilot

A fully local, offline-first vision-language inference pipeline. It points a small
open VLM at your webcam and logs a running, timestamped description of what the
camera sees — with per-frame latency — entirely on-device.

Independent personal exploration of edge AI. No cloud, no API keys, no telemetry.

## Why this is interesting

Two years ago, "describe what the camera sees" meant shipping frames to a hosted
model. It doesn't anymore. Small open VLMs (Qwen2-VL-2B and friends) now run real
scene perception on a laptop CPU or a modest GPU. This project is a minimal harness
to feel where that frontier actually is: how good the descriptions are, and how
much latency you pay per frame when nothing leaves the machine.

The only network call is the one-time model download from Hugging Face on first
run. After that, inference is 100% local — pull the ethernet cable and it still works.

## How it works

`vlm_pipeline.py`:
1. Loads a small VLM + processor via `transformers` (weights cached locally after first download).
2. Captures webcam frames with OpenCV.
3. Every N seconds, runs one frame through the model with a fixed prompt.
4. Logs `timestamp | latency | description` to the console and a rotating log file (`logs/scenes.log`).

The model is swappable — pass any HF image-text-to-text model via `--model`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads the model (~4 GB for Qwen2-VL-2B). Everything after is offline.

## Run

```bash
python vlm_pipeline.py                      # default: Qwen2-VL-2B, every 5s, camera 0
python vlm_pipeline.py --interval 10        # one description every 10 seconds
python vlm_pipeline.py --model <hf-model-id>  # swap the model
python vlm_pipeline.py --max-iters 3        # run 3 inferences then stop
```

Ctrl-C to stop. Descriptions stream to the console and append to `logs/scenes.log`.

## Self-test (no model, no camera)

Validates frame handling, timing, and logging with a mock model and synthetic
numpy frames — no weights or webcam required:

```bash
python test_pipeline_mock.py
# -> OK: 3 inferences, empty frame skipped, timing + logging verified
```

## Current limitations

- **Latency:** on CPU, a single 2B-VLM inference can take several seconds. This is a
  perception-cadence tool, not real-time video. Use a GPU (or a smaller model) for snappier output.
- **One frame per interval:** no temporal reasoning across frames — each description is independent.
- **Fixed prompt:** the prompt is hardcoded; no interactive querying of the scene yet.
- **First run needs network** to download weights; inference itself never does.
- **Model quality:** small VLMs hallucinate and miss detail. Treat descriptions as approximate.
- **Camera assumptions:** defaults to OpenCV camera index 0; adjust `--camera` for other devices.
