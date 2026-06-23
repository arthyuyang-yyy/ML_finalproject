"""Generate a small synthetic wav for pipeline smoke testing."""
import os
import numpy as np
import soundfile as sf

os.makedirs("data/demo", exist_ok=True)
sr = 16000
t = np.arange(sr * 3) / sr
audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
sf.write("data/demo/test_tone.wav", audio, sr)
print("wrote data/demo/test_tone.wav len=", audio.size / sr, "s")