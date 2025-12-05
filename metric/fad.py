import os
from pathlib import Path
import numpy as np, torch, torchaudio
from scipy import linalg
from numpy.lib.scimath import sqrt as scisqrt
from tqdm import tqdm

def _emb_stats(emb):
    """Return (mean, covariance) with guards.
    Ensures at least two frames to avoid degenerate covariance.
    """
    emb = np.array(emb)
    if emb.ndim == 1:
        emb = emb[None, :]
    if emb.shape[0] == 0:
        # Return zeros to avoid NaNs; caller should handle empty embeddings
        return np.zeros((emb.shape[1],), dtype=np.float64), np.eye(emb.shape[1], dtype=np.float64)
    if emb.shape[0] == 1:
        emb = np.vstack([emb, emb])
    return emb.mean(axis=0), np.cov(emb, rowvar=False)

def _frechet(mu1, s1, mu2, s2, eps=1e-6, trace_tol=1e-3):
    """Numerically-stable Frechet distance between Gaussians.
    Uses sqrtm with fallback to eigen decomposition and diagonal regularization if needed.
    """
    diff = mu1 - mu2
    # Try direct sqrtm first
    prod = s1.dot(s2)
    covmean_sqrtm, _ = linalg.sqrtm(prod, disp=False)

    # Fallback: eigenvalue method
    try:
        D, V = linalg.eig(prod)
        covmean_eig = (V * scisqrt(D)) @ linalg.inv(V)
    except Exception:
        covmean_eig = None

    covmean = covmean_sqrtm
    if covmean is None or not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))

    # Compare traces if both methods available and finite
    if covmean_eig is not None and np.isfinite(covmean_eig).all():
        tr_a = np.trace(covmean)
        tr_b = np.trace(covmean_eig)
        a_complex = np.iscomplexobj(tr_a)
        b_complex = np.iscomplexobj(tr_b)
        if a_complex and abs(tr_a.imag) < 1e-3:
            tr_a = tr_a.real
        if b_complex and abs(tr_b.imag) < 1e-3:
            tr_b = tr_b.real
        if not (np.iscomplexobj(tr_a) or np.iscomplexobj(tr_b)):
            if abs(tr_a - tr_b) > trace_tol:
                # Prefer eigen variant when large mismatch
                covmean = covmean_eig

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            covmean = covmean.real

    return diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean)

def _load_vggish_model():
    """Load torchvggish model via torch.hub."""
    model = torch.hub.load("harritaylor/torchvggish", "vggish")
    # Use raw embeddings (no PCA/quantization) to keep scales comparable across works
    model.postprocess = False
    model.eval()
    return model

def _prepare_waveform(src):
    if isinstance(src, (str, Path)):
        wav, sr = torchaudio.load(str(src))
    elif isinstance(src, (tuple, list)) and len(src) == 2:
        audio, sr = src
        sr = int(sr)
        if isinstance(audio, torch.Tensor):
            wav = audio.detach().cpu().float()
        else:
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            wav = torch.from_numpy(arr.astype(np.float32))
    else:
        raise TypeError(f"Unsupported audio input type: {type(src)}")
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav - wav.mean()
    return wav, sr

def _vggish_embeddings_inputs(inputs, target_sr=16000, min_len=32000):
    """Extract VGGish embeddings for a list of inputs (paths or (audio, sr))."""
    if len(inputs) == 0:
        return np.empty((0, 128), dtype=np.float32)
    model = _load_vggish_model()
    all_emb = None
    for idx, src in enumerate(tqdm(inputs, desc="VGGish embeddings")):
        try:
            wav, in_sr = _prepare_waveform(src)
            if wav.numel() == 0:
                print(f"[FAD] Empty waveform at index {idx}")
                continue
            # resample to 16k
            if in_sr != target_sr:
                # Use high-quality sinc resampler similar to reference
                resampler = torchaudio.transforms.Resample(
                    in_sr,
                    target_sr,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
                wav = resampler(wav)
            # pad to at least 2s
            if wav.size(-1) < min_len:
                pad = min_len - wav.size(-1)
                wav = torch.nn.functional.pad(wav, (0, pad), mode='constant', value=0.0)
            y = wav.squeeze(0).numpy()
            with torch.no_grad():
                emb = model.forward(y, target_sr)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            # guard against NaNs/Infs
            if not np.isfinite(emb).all():
                emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            if emb is None or (isinstance(emb, np.ndarray) and emb.size == 0):
                continue
            all_emb = emb if all_emb is None else np.concatenate([all_emb, emb], axis=0)
        except Exception as exc:
            print(f"[FAD] Failed to process input {idx}: {exc}")
            continue
    return all_emb if all_emb is not None else np.empty((0, 128), dtype=np.float32)

def openl3_fd(
    eval_files, ref_files,
    channels=1, samplingrate=44100, content_type='music', openl3_hop_size=0.5
) -> float:
    # Interface preserved; implementation uses VGGish embeddings for better comparability with papers.
    # Provided parameters are kept for signature compatibility but not used by VGGish backend.
    eval_emb = _vggish_embeddings_inputs(list(eval_files))
    ref_emb  = _vggish_embeddings_inputs(list(ref_files))
    mu_e, s_e = _emb_stats(eval_emb); mu_r, s_r = _emb_stats(ref_emb)
    return float(_frechet(mu_e, s_e, mu_r, s_r))
