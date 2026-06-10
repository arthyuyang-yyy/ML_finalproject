import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def preprocess_audio(input_path, output_path, target_sr=16000):
    """
    读取音频，转 mono，重采样到 target_sr，峰值归一化，保存 float32 wav。
    """
    waveform, sr = sf.read(input_path, dtype='float32', always_2d=True)

    mono = waveform.mean(axis=1)

    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        mono = resample_poly(mono, up, down).astype(np.float32)

    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak

    sf.write(output_path, mono, target_sr, subtype='FLOAT')
