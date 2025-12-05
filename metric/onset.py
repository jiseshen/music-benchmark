# from audio paths -> onset F-measure
from pathlib import Path
import librosa
import numpy as np


def _resolve_audio_input(src, sr_hint=None):
    if isinstance(src, (str, Path)):
        y, sr = librosa.load(str(src), sr=sr_hint, mono=True)
    elif isinstance(src, (tuple, list)) and len(src) == 2:
        y, sr = src
    else:
        raise TypeError(f"Unsupported audio input type: {type(src)}")
    y = np.asarray(y, dtype=np.float32)
    sr = int(sr)
    if sr_hint is not None and sr != sr_hint:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_hint)
        sr = sr_hint
    return y, sr

def _match_counts(ref_times: np.ndarray, est_times: np.ndarray, window: float) -> int:
    """Greedy one-to-one matching between reference and estimated onset times within a tolerance window (seconds).
    Returns the number of true positives (matches). Both inputs must be sorted.
    """
    i = 0
    j = 0
    tp = 0
    n_ref = len(ref_times)
    n_est = len(est_times)
    while i < n_ref and j < n_est:
        dt = est_times[j] - ref_times[i]
        if abs(dt) <= window:
            tp += 1
            i += 1
            j += 1
        elif ref_times[i] < est_times[j]:
            i += 1
        else:
            j += 1
    return tp


def onset_f_from_audio(ref_wav, est_wav, sr=None, hop_length=512, backtrack=True, win=0.05):
    y_ref, sr_ref = _resolve_audio_input(ref_wav, sr_hint=sr)
    y_est, sr_est = _resolve_audio_input(est_wav, sr_hint=sr_ref if sr is None else sr)
    target_sr = sr if sr is not None else sr_ref
    if sr_ref != target_sr:
        y_ref = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=target_sr)
        sr_ref = target_sr
    if sr_est != target_sr:
        y_est = librosa.resample(y_est, orig_sr=sr_est, target_sr=target_sr)
        sr_est = target_sr
    sr = target_sr

    if y_ref.size == 0 or y_est.size == 0:
        return {"F": float("nan"), "P": float("nan"), "R": float("nan"), "count_ref": 0, "count_est": 0}

    oenv_ref = librosa.onset.onset_strength(y=y_ref, sr=sr, hop_length=hop_length)
    oenv_est = librosa.onset.onset_strength(y=y_est, sr=sr, hop_length=hop_length)

    on_ref_fr = librosa.onset.onset_detect(onset_envelope=oenv_ref, sr=sr, hop_length=hop_length, backtrack=False)
    on_est_fr = librosa.onset.onset_detect(onset_envelope=oenv_est, sr=sr, hop_length=hop_length, backtrack=False)

    if backtrack:
        on_ref_fr = librosa.onset.onset_backtrack(on_ref_fr, oenv_ref)
        on_est_fr = librosa.onset.onset_backtrack(on_est_fr, oenv_est)

    on_ref_s = np.asarray(librosa.frames_to_time(on_ref_fr, sr=sr, hop_length=hop_length), dtype=float)
    on_est_s = np.asarray(librosa.frames_to_time(on_est_fr, sr=sr, hop_length=hop_length), dtype=float)

    # Ensure sorted order (should already be, but be safe)
    on_ref_s.sort()
    on_est_s.sort()

    tp = _match_counts(on_ref_s, on_est_s, window=win)
    n_ref = len(on_ref_s)
    n_est = len(on_est_s)
    fp = max(0, n_est - tp)
    fn = max(0, n_ref - tp)
    p = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    r = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"F": float(f), "P": float(p), "R": float(r), "count_ref": n_ref, "count_est": n_est}
