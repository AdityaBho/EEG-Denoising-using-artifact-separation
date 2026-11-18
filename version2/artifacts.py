"""Stage 1 - explicit artifact models (EOG, EMG, motion, 50 Hz) with spatial/spectral/temporal signatures.
contaminate() follows the EEGdenoiseNet convention: SNR_dB = 20 log10(RMS(clean)/RMS(artifact))."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from .montage import XY

def _rms(a): return np.sqrt((a ** 2).mean(axis=(-2, -1), keepdims=True))
def _lp(x, fc, fs): return sosfiltfilt(butter(2, fc, fs=fs, output='sos'), x, axis=-1)

def eog(n, T, fs, rng):
    """Blinks (<=4 Hz, large, frontal topography) + slow ocular drift. Returns art (n,C,T), ref (n,1,T)."""
    src = np.zeros((n, T)); L = int(0.3 * fs); tpl = np.sin(np.linspace(0, np.pi, L)) ** 2
    for i in range(n):
        for _ in range(rng.poisson(1.5)):
            s = rng.integers(0, T - L); src[i, s:s + L] += rng.uniform(.6, 1.4) * tpl
    src += 0.3 * _lp(rng.standard_normal((n, T)), 2.0, fs) / 0.35
    w = np.exp(-np.linalg.norm(XY - np.array([0, .95]), axis=1) / 0.7)
    return src[:, None, :] * w[None, :, None], src[:, None, :]

def emg(n, T, fs, rng):
    """Broadband 20-60 Hz bursts, strongest at temporal/frontal sites."""
    sos = butter(4, [20, min(60, fs / 2 - 2)], btype='band', fs=fs, output='sos')
    base = sosfiltfilt(sos, rng.standard_normal((n, len(XY), T)), axis=-1)
    env = np.zeros((n, 1, T))
    for i in range(n):
        for _ in range(rng.integers(1, 4)):
            s = rng.integers(0, T - fs // 2); env[i, 0, s:s + rng.integers(fs // 4, fs)] = 1
    env = _lp(env, 4, fs).clip(0)
    d = np.minimum(np.linalg.norm(XY - [-1, 0], axis=1), np.linalg.norm(XY - [1, 0], axis=1))
    return base * env * np.exp(-d / 0.6)[None, :, None], None

def motion(n, T, fs, rng):
    """Electrode-movement: step/baseline jumps with exponential recovery, channel-specific gains."""
    art = np.zeros((n, len(XY), T)); t = np.arange(T) / fs
    for i in range(n):
        for _ in range(rng.integers(1, 3)):
            s = rng.integers(0, T - 1); g = rng.standard_normal(len(XY)) + 1.0
            art[i, :, s:] += g[:, None] * np.exp(-(t[s:] - t[s]) / 0.4)[None] * rng.choice([-1, 1])
    return art, None

def line(n, T, fs, rng, f0=50.0):
    t = np.arange(T) / fs; ph = rng.uniform(0, 2 * np.pi, (n, 1, 1))
    g = rng.uniform(.5, 1.5, (1, len(XY), 1))
    ph2 = ph[:, 0, :]                                            # (n,1) true mains phase seen by the reference
    ref = np.stack([np.sin(2 * np.pi * f0 * t[None] + ph2), np.cos(2 * np.pi * f0 * t[None] + ph2)], 1)  # (n,2,T)
    return g * np.sin(2 * np.pi * f0 * t[None, None] + ph), ref

GEN = dict(eog=eog, emg=emg, motion=motion, line=line)

def contaminate(clean, kind, snr_db, fs, seed=1):
    """kind in {eog,emg,motion,line,mixed}. Returns noisy, artifact, refs (EOG source + 50 Hz quadrature)."""
    rng = np.random.default_rng(seed); n, C, T = clean.shape; art = np.zeros_like(clean); refs = None
    kinds = ['eog', 'emg', 'line'] if kind == 'mixed' else [kind]
    off = 4.77 if kind == 'mixed' else 0.0                     # 3 equal-power sources -> same total SNR
    rf = {}
    for k in kinds:
        a, r = GEN[k](n, T, fs, rng); g = _rms(clean) / ((_rms(a) + 1e-12) * 10 ** ((snr_db + off) / 20))   # (n,1,1)
        art += (a * g).astype(np.float32)
        if r is not None: rf[k] = r * g
    parts = [rf[k] for k in ('eog', 'line') if k in rf]
    refs = np.concatenate(parts, 1).astype(np.float32) if parts else np.zeros((n, 1, T), np.float32)
    return (clean + art).astype(np.float32), art.astype(np.float32), refs
