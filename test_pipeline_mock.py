"""Self-test: exercises frame-handling, timing, and logging with a mock model and
synthetic frames. No real weights, no camera. Run: python test_pipeline_mock.py
"""
import time
import numpy as np

import vlm_pipeline as vp


def synthetic_frames(n, include_empty=True):
    """Random RGB frames; slips in one empty frame to test the skip path."""
    for i in range(n):
        if include_empty and i == 1:
            yield np.empty((0, 0, 3), dtype=np.uint8)
        yield np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)


def main():
    logger = vp.make_logger(log_dir="logs")

    calls = []

    def mock_infer(frame):
        assert frame.ndim == 3 and frame.shape[2] == 3, "expected RGB frame"
        calls.append(frame.shape)
        return f"mock scene {len(calls)}"

    interval = 0.05
    t0 = time.perf_counter()
    count = vp.run(synthetic_frames(4), mock_infer, interval, logger, max_iters=3)
    elapsed = time.perf_counter() - t0

    # frame handling: empty frame skipped, only valid frames reached the model
    assert count == 3, f"expected 3 inferences, got {count}"
    assert len(calls) == 3, f"model called {len(calls)} times"
    # timing: 3 iters => 2 inter-iteration sleeps, none after the last.
    # Lower bound proves the sleeps happen; upper bound allows fixed cold-start overhead.
    assert elapsed >= 2 * interval, f"loop too fast, timing broken: {elapsed:.3f}s"
    assert elapsed < 2 * interval + 2.5, f"loop far too slow, interval ignored?: {elapsed:.3f}s"

    # logging: file handler actually wrote our lines
    for h in logger.handlers:
        h.flush()
    log_text = open("logs/scenes.log").read()
    assert "mock scene 3" in log_text, "log file missing inference output"
    assert "latency=" in log_text, "latency not logged"

    print("OK: 3 inferences, empty frame skipped, timing + logging verified")


if __name__ == "__main__":
    main()
