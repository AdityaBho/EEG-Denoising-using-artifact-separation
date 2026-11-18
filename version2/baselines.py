"""Baselines. Every function: f(noisy (N,C,T), refs (N,R,T), fs) -> denoised (N,C,T)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from .montage import IDX

def identity(x, refs, fs): return x.copy()

def lowpass8(x, refs, fs):             # STRAW-MAN: kills EMG/line but also beta/gamma/phase -> NIPI must punish it
    return sosfiltfilt(butter(4, 8, fs=fs, output='sos'), x, axis=-1).astype(np.float32)

def notch50(x, refs, fs):
    b, a = iirnotch(50.0, 30.0, fs); return filtfilt(b, a, x, axis=-1).astype(np.float32)

def nlms(x, refs, fs, L=4, mu=0.05, reg=1.0):
    """NLMS adaptive noise canceller, streaming over all epochs; weights shared across epochs (per channel)."""
    N, C, T = x.shape; R = refs.shape[1]
    d = x.transpose(1, 0, 2).reshape(C, -1); u = refs.transpose(1, 0, 2).reshape(R, -1)
    W = np.zeros((C, R * L)); out = np.empty_like(d); buf = np.zeros(R * L)
    delta = reg * R * L * float((u ** 2).mean(1).mean()) + 1e-6      # regulariser ~ typical reference power (stabilises intermittent refs)
    for t in range(d.shape[1]):
        buf = np.concatenate([u[:, t:t + 1], buf.reshape(R, L)[:, :-1]], 1).ravel() if t else np.concatenate([u[:, :1], np.zeros((R, L - 1))], 1).ravel()
        e = d[:, t] - W @ buf
        W += mu * np.outer(e, buf) / (buf @ buf + delta); out[:, t] = e
    return out.reshape(C, N, T).transpose(1, 0, 2).astype(np.float32)

def ica(x, refs, fs, seed=0):
    """FastICA + automatic rejection: EOG (frontal & kurtotic), EMG (>20 Hz power ratio), line (49-51 Hz ratio)."""
    N, C, T = x.shape; Z = x.transpose(1, 0, 2).reshape(C, -1).T            # (samples, C)
    m = FastICA(n_components=C, whiten='unit-variance', max_iter=600, random_state=seed, tol=1e-3)
    S = m.fit_transform(Z); A = m.mixing_                                    # S (samples,comp), A (C,comp)
    f, P = welch(S.T, fs, nperseg=256, axis=-1); P = P / P.sum(-1, keepdims=True)
    hf = P[:, f > 20].sum(1); ln = P[:, (f > 49) & (f < 51)].sum(1)
    fr = (np.abs(A[IDX['Fp1']]) + np.abs(A[IDX['Fp2']])) / (np.abs(A).mean(0) * 2 + 1e-9)
    k = kurtosis(S, axis=0)
    bad = ((k > 3) & (fr > 1.5)) | (hf > 0.5) | (ln > 0.1)
    S[:, bad] = 0
    return m.inverse_transform(S).T.reshape(C, N, T).transpose(1, 0, 2).astype(np.float32)

def wavelet(x, refs, fs, wavelet='db4', level=5, k=3.0):
    """Outlier-coefficient wavelet denoising: zero coefficients > k * robust sigma of their level.
    (A universal/MAD *shrinkage* threshold destroys EEG because the brain rhythm is the bulk of every level.)"""
    import pywt
    co = pywt.wavedec(x, wavelet, level=level, axis=-1); new = []
    for c in co:
        thr = k * np.median(np.abs(c), axis=-1, keepdims=True) / 0.6745
        new.append(np.where(np.abs(c) > thr, 0.0, c))
    return pywt.waverec(new, wavelet, axis=-1)[..., :x.shape[-1]].astype(np.float32)

CLASSICAL = dict(Noisy_input=identity, Lowpass_8Hz=lowpass8, Notch_50=notch50, NLMS=nlms, ICA=ica)
try:
    import pywt; CLASSICAL['Wavelet'] = wavelet
except ImportError:
    pass
