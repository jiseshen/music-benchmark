from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm


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

def cents_diff(a, b, eps=1e-8):
    return 1200 * np.log2(np.maximum(a, eps) / np.maximum(b, eps))

def eval_pitch_metrics(gen_files, ref_files, fmin=50, fmax=2000, hop_length=256):
    total_frames = 0
    gpe_count = 0
    fpe_errors = []
    rpa_count = 0

    for idx, (g, r) in enumerate(tqdm(zip(gen_files, ref_files), desc="F0 metrics", total=len(gen_files))):
        try:
            y_g, sr_g = _resolve_audio_input(g)
            y_r, sr_r = _resolve_audio_input(r, sr_hint=sr_g)
        except Exception as exc:
            print(f"[F0] Failed to load pair {idx}: {exc}")
            continue
        if sr_r != sr_g:
            y_r = librosa.resample(y_r, orig_sr=sr_r, target_sr=sr_g)
            sr_r = sr_g
        if y_g.size == 0 or y_r.size == 0:
            print(f"[F0] Empty audio for pair {idx}")
            continue

        minlen = min(len(y_g), len(y_r))
        y_g, y_r = y_g[:minlen], y_r[:minlen]

        f0g, v_g, _ = librosa.pyin(y_g, sr=sr_g, fmin=fmin, fmax=fmax, hop_length=hop_length)
        f0r, v_r, _ = librosa.pyin(y_r, sr=sr_g, fmin=fmin, fmax=fmax, hop_length=hop_length)

        valid = (~np.isnan(f0g)) & (~np.isnan(f0r))
        if not np.any(valid):
            continue

        fg, fr = f0g[valid], f0r[valid]
        total_frames += len(fg)

        # Compute % error
        pct_err = np.abs(fg - fr) / fr
        gpe_count += np.sum(pct_err > 0.2)

        mid_err = pct_err[(pct_err <= 0.2) & (pct_err > 0.05)]
        fpe_errors.extend(mid_err * 100)  # percent

        cd = np.abs(cents_diff(fg, fr))
        rpa_count += np.sum(cd <= 50)

    gpe = gpe_count / total_frames if total_frames else np.nan
    fpe = np.mean(fpe_errors) if fpe_errors else np.nan
    rpa = rpa_count / total_frames if total_frames else np.nan

    return {"GPE": gpe, "FPE (%)": fpe, "RPA": rpa}
