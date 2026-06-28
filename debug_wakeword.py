"""Wake word score diagnostic — run this and say 'hey jarvis' a few times.

Shows live scores so we can find the right sensitivity threshold.

Usage:
    .venv/bin/python debug_wakeword.py
"""
import sys, time, queue
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE   = 16_000
CHUNK_SAMPLES = 1_280   # 80 ms — what openwakeword expects

print("Loading model...", flush=True)
oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
print("Model loaded.\n")
print("Say 'hey jarvis' several times. Scores print whenever they're > 0.")
print("Press Ctrl+C to stop.\n")
print(f"{'TIME':>8}  {'SCORE':>8}  BAR")
print("-" * 55)

audio_q: queue.Queue = queue.Queue()

def callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())

max_score = 0.0
t0 = time.time()

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    blocksize=CHUNK_SAMPLES,
    callback=callback,
):
    try:
        while True:
            chunk = audio_q.get()
            chunk_1d = chunk[:, 0]  # (1280,1) → (1280,)

            # openwakeword accepts int16 directly
            pred = oww.predict(chunk_1d)
            score = float(max(pred.values())) if pred else 0.0

            if score > max_score:
                max_score = score

            # Print every chunk that scores above 0
            if score > 0.0:
                bar = "█" * int(score * 50)
                elapsed = time.time() - t0
                print(f"{elapsed:>7.1f}s  {score:>8.4f}  {bar}")

    except KeyboardInterrupt:
        pass

print(f"\nPeak score: {max_score:.4f}")
if max_score == 0.0:
    print("→ Score never exceeded 0 — audio may not be reaching the model.")
elif max_score < 0.05:
    print(f"→ Very low scores. Model may not recognise your pronunciation.")
else:
    print(f"→ Recommended sensitivity: {max_score * 0.7:.2f}")
