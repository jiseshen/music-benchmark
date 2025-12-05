# visqol_iface.py
import os
import re
import shlex, shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

_VISQOL_ROOT = os.environ.get("VISQOL_ROOT")
_VISQOL_BIN = Path(_VISQOL_ROOT) / "bazel-bin" / "visqol" if _VISQOL_ROOT else None
_VISQOL_MODEL = Path(_VISQOL_ROOT) / "model" / "libsvm_nu_svr_model.txt" if _VISQOL_ROOT else None
_SR = 48000
_MOS_RE = re.compile(r"(?:MOS[-\s]*LQO|ViSQOL score)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I)


def _ensure_48k_wav(path: str, tmpdir: Optional[Path]) -> str:
    """Return a 48k PCM16 WAV path. If conversion needed, write into tmpdir and return that path."""
    ext = Path(path).suffix.lower()
    try:
        y, sr = sf.read(path, always_2d=True)  # (n, ch)
    except Exception:
        # fallback via librosa for exotic codecs
        y_mono, sr = librosa.load(path, sr=None, mono=True)
        y = y_mono[:, None]

    if sr != _SR:
        y = np.stack([librosa.resample(y[:, c], orig_sr=sr, target_sr=_SR) for c in range(y.shape[1])], axis=1)
        sr = _SR

    # If already WAV 48k PCM16 and not float32, we can return original path; else write tmp wav.
    # For simplicity and safety, always write a clean PCM16.
    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="visqol_tmp_"))
    out = tmpdir / (Path(path).stem + "_48k.wav")
    sf.write(out, y, samplerate=sr, subtype="PCM_16")
    return str(out)


def _run_visqol(ref_wav_48k: str, deg_wav_48k: str, *, visqol_bin: Optional[str], speech_mode: bool, extra_args: Optional[List[str]], timeout: Optional[float]) -> float:
    vb = visqol_bin or _VISQOL_BIN
    args = [vb, "--reference_file", ref_wav_48k, "--degraded_file", deg_wav_48k, "--similarity_to_quality_model", _VISQOL_MODEL]
    if speech_mode:
        args.append("--use_speech_mode")
    if extra_args:
        args.extend(extra_args)

    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout or 300, check=False)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = _MOS_RE.search(out)
    if proc.returncode != 0 or not m:
        raise RuntimeError(f"[visqol] failed (code={proc.returncode}).\nCMD: {shlex.join(args)}\n--- OUTPUT ---\n{out[-4000:]}")
    return float(m.group(1))


def visqol_moslqo(
    gen_files: List[str],
    ref_files: List[str],
    *,
    speech_mode: bool = False,
    visqol_bin: Optional[str] = None,
    timeout: Optional[float] = 300.0,
    extra_args: Optional[List[str]] = None,
    return_per_pair: bool = False,
) -> float | Tuple[float, List[float]]:
    """
    gen_files[i] vs ref_files[i] -> MOS-LQO. Average returned; set return_per_pair=True to also get list.
    """
    assert len(gen_files) == len(ref_files) and len(gen_files) > 0
    tmpdir = Path(tempfile.mkdtemp(prefix="visqol_run_"))

    mos_vals: List[float] = []
    try:
        for g, r in tqdm(zip(gen_files, ref_files)):
            r48 = _ensure_48k_wav(r, tmpdir)
            g48 = _ensure_48k_wav(g, tmpdir)
            mos = _run_visqol(r48, g48, visqol_bin=visqol_bin, speech_mode=speech_mode, extra_args=extra_args, timeout=timeout)
            mos_vals.append(mos)
    finally:
        # best-effort cleanup
        try:
            for p in tmpdir.glob("*"):
                p.unlink(missing_ok=True)
            tmpdir.rmdir()
        except Exception:
            pass

    avg = float(np.mean(mos_vals)) if mos_vals else float("nan")
    return (avg, mos_vals) if return_per_pair else avg
