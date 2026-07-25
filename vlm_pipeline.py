"""Offline-first VLM scene copilot.

Captures webcam frames, runs a small local vision-language model every N seconds,
and logs timestamped scene descriptions + latency to console and a rotating file.

No network at inference time. The model downloads once on first run (HF cache),
then everything is local. Model is swappable via --model.
"""
import argparse
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
PROMPT = "Describe this scene in one concise sentence."


def make_logger(log_dir="logs"):
    Path(log_dir).mkdir(exist_ok=True)
    logger = logging.getLogger("vlm")
    if logger.handlers:  # idempotent (matters for the test)
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = RotatingFileHandler(Path(log_dir) / "scenes.log", maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_model(model_id):
    """Load a HF vision-language model + processor. Downloads once, then cached."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map=device
    )

    def infer(frame_rgb):
        from PIL import Image

        image = Image.fromarray(frame_rgb)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    return infer


def webcam_frames(cam_index=0):
    """Yield RGB frames from the webcam forever."""
    import cv2

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {cam_index}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError("Webcam frame read failed")
            yield frame_bgr[:, :, ::-1].copy()  # BGR -> RGB
    finally:
        cap.release()


def run(frames, infer, interval, logger, max_iters=None):
    """Core loop: pull a frame, describe it, log it, sleep. Injectable for tests."""
    count = 0
    for frame in frames:
        if frame is None or getattr(frame, "size", 1) == 0:
            logger.warning("empty frame, skipping")
            continue
        t0 = time.perf_counter()
        desc = infer(frame)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"latency={latency_ms:.0f}ms | {desc}")
        count += 1
        if max_iters is not None and count >= max_iters:
            break
        time.sleep(interval)
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (swappable)")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between inferences")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--max-iters", type=int, default=None, help="stop after N inferences")
    args = ap.parse_args()

    logger = make_logger()
    logger.info(f"loading model {args.model} (first run downloads weights)...")
    infer = load_model(args.model)
    logger.info("model ready; starting capture. Ctrl-C to stop.")
    try:
        run(webcam_frames(args.camera), infer, args.interval, logger, args.max_iters)
    except KeyboardInterrupt:
        logger.info("stopped by user")


if __name__ == "__main__":
    main()
