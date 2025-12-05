from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import tempfile

# ---------------------------------------------
# Formalized segmentation configuration
# ---------------------------------------------
# Each logical content segment length (seconds)
SEG_LEN: float = 5.0
# Gap of silence (or ignored region) between segments
SEG_GAP: float = 0.2

# Separation task layout (total ≈ 2*SEG_LEN + SEG_GAP = 10.4s)
#   Input segment:        [0.0, SEG_LEN)
#   Generated segment:    [SEG_LEN+SEG_GAP, 2*SEG_LEN+SEG_GAP)
SEP_INPUT_START = 0.0
SEP_INPUT_END   = SEG_LEN
SEP_GEN_START   = SEG_LEN + SEG_GAP          # 5.2
SEP_GEN_END     = 2 * SEG_LEN + SEG_GAP      # 10.2

# Infill task layout (total ≈ 3*SEG_LEN + 2*SEG_GAP = 15.4s)
#   Input segment:        [0.0, SEG_LEN)
#   Intermediate gen:     [SEG_LEN+SEG_GAP, 2*SEG_LEN+SEG_GAP)   (optional use)
#   Result segment:       [2*SEG_LEN+2*SEG_GAP, 3*SEG_LEN+2*SEG_GAP)
INFILL_INPUT_START   = 0.0
INFILL_INPUT_END     = SEG_LEN
INFILL_GEN_START     = SEG_LEN + SEG_GAP            # 5.2
INFILL_GEN_END       = 2 * SEG_LEN + SEG_GAP        # 10.2
INFILL_RESULT_START  = 2 * SEG_LEN + 2 * SEG_GAP    # 10.4
INFILL_RESULT_END    = 3 * SEG_LEN + 2 * SEG_GAP    # 15.4

def _print_seg_layout_once():
    if getattr(_print_seg_layout_once, "_printed", False):
        return
    print("[SEG] Layout formalized:")
    print(f"  Separation: input=[{SEP_INPUT_START:.1f},{SEP_INPUT_END:.1f}), gen=[{SEP_GEN_START:.1f},{SEP_GEN_END:.1f})")
    print(f"  Infill: input=[{INFILL_INPUT_START:.1f},{INFILL_INPUT_END:.1f}), gen=[{INFILL_GEN_START:.1f},{INFILL_GEN_END:.1f}), result=[{INFILL_RESULT_START:.1f},{INFILL_RESULT_END:.1f})")
    _print_seg_layout_once._printed = True

def get_segment(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Return waveform segment for [start_s, end_s).
    If bounds exceed length, will clamp; if invalid (end<=start) returns empty array."""
    if end_s <= start_s:
        return np.zeros(0, dtype=y.dtype)
    start = int(round(start_s * sr))
    end = int(round(end_s * sr))
    start = max(0, min(start, y.shape[-1]))
    end = max(0, min(end, y.shape[-1]))
    return y[start:end]

@dataclass
class Sample:
    id: str
    text: str
    gt_path: Path
    gen_path: Path
    input_path: Path | None = None


@dataclass
class AlignedAudio:
    sr: int
    ref: np.ndarray
    gen_tail: np.ndarray
    full_gen: np.ndarray


def _load_and_align_audio(sample: Sample, tail_seconds_hint: Optional[float] = None, *, 
                          crop_start: Optional[float] = None, crop_end: Optional[float] = None) -> Optional[AlignedAudio]:
    """Load gen/gt audio, resample to common SR, and return tail-aligned segments.
    
    Args:
        sample: Sample with gen_path and gt_path
        tail_seconds_hint: If set, crop both gen (last N seconds) and ref to this duration
        crop_start: If set, crop gen starting from this second (overrides tail_seconds_hint)
        crop_end: If set, crop gen ending at this second (used with crop_start)
    
    Returns:
        AlignedAudio with matched gen_tail and ref segments, or None if failed
    """
    gen_path, ref_path = sample.gen_path, sample.gt_path
    if not gen_path.exists() or not ref_path.exists():
        print(f"[Align] Missing file(s) for sample {sample.id}: gen={gen_path.exists()} ref={ref_path.exists()}")
        return None
    try:
        y_gen, sr_gen = librosa.load(str(gen_path), sr=None, mono=True)
        y_ref, sr_ref = librosa.load(str(ref_path), sr=None, mono=True)
    except Exception as exc:
        print(f"[Align] Failed to load audio for sample {sample.id}: {exc}")
        return None
    if y_gen.size == 0 or y_ref.size == 0:
        print(f"[Align] Empty waveform for sample {sample.id}: gen={y_gen.size} ref={y_ref.size}")
        return None
    
    # Resample to common SR
    if sr_gen != sr_ref:
        try:
            y_gen = librosa.resample(y_gen, orig_sr=sr_gen, target_sr=sr_ref)
            sr_gen = sr_ref
        except Exception as exc:
            print(f"[Align] Resample error for sample {sample.id}: {exc}")
            return None
    
    sr = sr_ref
    
    # Handle explicit crop_start/crop_end (for separation/infill tasks)
    if crop_start is not None and crop_end is not None:
        start_samp = int(crop_start * sr)
        end_samp = int(crop_end * sr)
        if start_samp >= len(y_gen) or end_samp > len(y_gen) or start_samp >= end_samp:
            print(f"[Align] Invalid crop range [{crop_start}s-{crop_end}s] for sample {sample.id} (gen len: {len(y_gen)/sr:.2f}s)")
            return None
        y_tail = y_gen[start_samp:end_samp]
        target_len = y_tail.shape[-1]
        
        # Crop/pad ref to match
        if y_ref.shape[-1] > target_len:
            y_ref = y_ref[:target_len]
        elif y_ref.shape[-1] < target_len:
            pad_len = target_len - y_ref.shape[-1]
            y_ref = np.pad(y_ref, (0, pad_len), mode='constant')
    
    # Otherwise use tail_seconds_hint
    elif tail_seconds_hint and tail_seconds_hint > 0:
        # Use hint to determine target length
        target_len = int(tail_seconds_hint * sr)
        
        # Crop ref to target length (from end if longer, pad if shorter)
        if y_ref.shape[-1] > target_len:
            y_ref = y_ref[-target_len:]
        elif y_ref.shape[-1] < target_len:
            # Pad ref if shorter than hint
            pad_len = target_len - y_ref.shape[-1]
            y_ref = np.pad(y_ref, (0, pad_len), mode='constant')
        
        # Crop gen to last target_len samples
        if y_gen.shape[-1] < target_len:
            print(f"[Align] Generated clip shorter than target ({y_gen.size} < {target_len}) for sample {sample.id}")
            return None
        
        start = y_gen.shape[-1] - target_len
        y_tail = y_gen[start:]
    
    else:
        # Use ref length as target
        target_len = y_ref.shape[-1]
        
        if target_len == 0:
            print(f"[Align] Target length zero for sample {sample.id}")
            return None
        
        # Crop gen to last target_len samples
        if y_gen.shape[-1] < target_len:
            print(f"[Align] Generated clip shorter than target ({y_gen.size} < {target_len}) for sample {sample.id}")
            return None
        
        start = y_gen.shape[-1] - target_len
        y_tail = y_gen[start:]
    
    # Final sanity check
    if y_tail.shape[-1] != y_ref.shape[-1]:
        min_len = min(y_tail.shape[-1], y_ref.shape[-1])
        if min_len <= 0:
            print(f"[Align] Unable to match lengths for sample {sample.id}")
            return None
        print(f"[Align] Length mismatch, trimming to {min_len} samples for sample {sample.id}")
        y_tail = y_tail[-min_len:]
        y_ref = y_ref[-min_len:]
    
    return AlignedAudio(
        sr=sr,
        ref=np.asarray(y_ref, dtype=np.float32),
        gen_tail=np.asarray(y_tail, dtype=np.float32),
        full_gen=np.asarray(y_gen, dtype=np.float32),
    )


def _crop_last_seconds(y: np.ndarray, sr: int, seconds: Optional[float]) -> np.ndarray:
    if seconds is None or seconds <= 0:
        return y
    win = int(seconds * sr)
    start = max(0, y.shape[-1] - win)
    return y[start:]

def compute_clamp3(samples: List[Sample]) -> float:
    from metric.clamp3_score import clamp3_score
    texts = [s.text for s in samples]
    gens  = [str(s.gen_path) for s in samples]
    return float(clamp3_score(texts, gens))

def compute_clap(samples: List[Sample], clap_model_name="630k-audioset-fusion-best.pt", cache_dir=Path("cache/clap")) -> float:
    from metric.clap import clap_score
    texts = [s.text for s in samples]
    gens  = [str(s.gen_path) for s in samples]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return float(clap_score(texts, gens, clap_model=clap_model_name, cache_dir=str(cache_dir)))

def compute_openl3_fad(
    samples: List[Sample],
    cfg: Dict | None = None,
    *,
    gen_inputs: Optional[List] = None,
    ref_inputs: Optional[List] = None,
) -> float:
    from metric.fad import openl3_fd
    cfg = {"channels": 1, "samplingrate": 44100, "content_type": "music", "openl3_hop_size": 0.5, **(cfg or {})}
    gens = gen_inputs if gen_inputs is not None else [str(s.gen_path) for s in samples]
    refs = ref_inputs if ref_inputs is not None else [str(s.gt_path) for s in samples]
    return float(openl3_fd(gens, refs, channels=cfg["channels"], samplingrate=cfg["samplingrate"], content_type=cfg["content_type"], openl3_hop_size=cfg["openl3_hop_size"]))

def compute_passt_kl(samples: List[Sample], cfg: Dict | None = None) -> float:
    from metric.kld import passt_kld
    cfg = {"collect": "mean", **(cfg or {})}
    gens = [str(s.gen_path) for s in samples]
    refs = [str(s.gt_path)  for s in samples]
    return float(passt_kld(gens, refs, collect=cfg["collect"]))

def compute_visqol(samples) -> float:
    from metric.visqol import visqol_moslqo
    gens = [str(s.gen_path) for s in samples]
    refs = [str(s.gt_path)  for s in samples]
    return float(visqol_moslqo(gens, refs))

def compute_aes(samples: List[Sample], aes_ckpt: Path, batch: int = 8) -> Dict[str, float]:
    try:
        from audiobox_aesthetics.infer import initialize_predictor
    except Exception as e:
        raise ImportError("audiobox_aesthetics is required for AES metrics") from e
    predictor = initialize_predictor(str(aes_ckpt))
    wavs = [{"path": str(s.gen_path)} for s in samples]
    scores = []
    for i in range(0, len(wavs), batch):
        scores.extend(predictor.forward(wavs[i:i+batch]))
    pq = sum(s["PQ"] for s in scores) / len(scores)
    ce = sum(s["CE"] for s in scores) / len(scores)
    return {"PQ": pq, "CE": ce}

def compute_aes_gain(samples: List[Sample], aes_ckpt: Path, batch: int = 8) -> Dict[str, float]:
    try:
        from audiobox_aesthetics.infer import initialize_predictor
    except Exception as e:
        raise ImportError("audiobox_aesthetics is required for AES metrics") from e
    predictor = initialize_predictor(str(aes_ckpt))
    gen_wavs = [{"path": str(s.gen_path)} for s in samples]
    input_wavs  = [{"path": str(s.input_path)}  for s in samples]
    gen_scores = []
    input_scores  = []
    for i in range(0, len(gen_wavs), batch):
        gen_scores.extend(predictor.forward(gen_wavs[i:i+batch]))
        input_scores.extend(predictor.forward(input_wavs[i:i+batch]))
    gen_pq = sum(s["PQ"] for s in gen_scores) / len(gen_scores)
    gen_ce = sum(s["CE"] for s in gen_scores) / len(gen_scores)
    input_pq  = sum(s["PQ"] for s in input_scores)  / len(input_scores)
    input_ce  = sum(s["CE"] for s in input_scores)  / len(input_scores)
    return {"PQ_gain": gen_pq - input_pq, "CE_gain": gen_ce - input_ce}

def evaluate_gen(
    samples: List[Sample],
    model_name: str,
    aes_batch_size: int = 8,
    clap_model_name: str = "630k-audioset-fusion-best.pt",
    openl3_cfg: dict | None = None,
    passt_cfg: dict | None = None,
    aes_ckpt: Path | None = None,
    output_dir: str = "result/gen_results.csv",
    **kwargs
) -> dict:
    result_path = Path(output_dir)
    if result_path.exists():
        df = pd.read_csv(result_path, index_col=0)
        if df.index.name != "model_name":
            df.index.name = "model_name"
    else:
        df = pd.DataFrame()
        df.index.name = "model_name"

    metrics = {}

    def _save(df: pd.DataFrame):
        df.to_csv(result_path, index=True, float_format="%.4f")

    # CLAP
    if "CLAP" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "CLAP"]):
        metrics["CLAP"] = float(df.loc[model_name, "CLAP"])
    else:
        metrics["CLAP"] = round(float(compute_clap(samples, clap_model_name, Path("cache/clap"))), 4)
        df.loc[model_name, "CLAP"] = metrics["CLAP"]
        _save(df)

    # CLAMP3
    if "CLAMP3" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "CLAMP3"]):
        metrics["CLAMP3"] = float(df.loc[model_name, "CLAMP3"])
    else:
        metrics["CLAMP3"] = round(float(compute_clamp3(samples)), 4)
        df.loc[model_name, "CLAMP3"] = metrics["CLAMP3"]
        _save(df)

    # FAD
    if "FAD" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "FAD"]):
        metrics["FAD"] = float(df.loc[model_name, "FAD"])
    else:
        metrics["FAD"] = round(float(compute_openl3_fad(samples, openl3_cfg or {})), 4)
        df.loc[model_name, "FAD"] = metrics["FAD"]
        _save(df)

    # KLD
    if "KLD" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "KLD"]):
        metrics["KLD"] = float(df.loc[model_name, "KLD"])
    else:
        metrics["KLD"] = round(float(compute_passt_kl(samples, passt_cfg or {})), 4)
        df.loc[model_name, "KLD"] = metrics["KLD"]
        _save(df)

    # AES
    aes_metrics = {}
    if aes_ckpt:
        aes_cols = [col for col in df.columns if col.startswith("AES_")]
        has_all_aes = all(col in df.columns and model_name in df.index and pd.notna(df.loc[model_name, col]) for col in aes_cols) and len(aes_cols) > 0
        if has_all_aes:
            for col in aes_cols:
                aes_metrics[col] = float(df.loc[model_name, col])
        else:
            aes_metrics = {f"AES_{k}": round(v, 4) for k, v in compute_aes(samples, aes_ckpt, aes_batch_size).items()}
            for k, v in aes_metrics.items():
                df.loc[model_name, k] = v
            _save(df)
        metrics.update(aes_metrics)

    # VISQOL
    if "VISQOL" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "VISQOL"]):
        metrics["VISQOL"] = float(df.loc[model_name, "VISQOL"])
    else:
        metrics["VISQOL"] = round(float(compute_visqol(samples)), 4)
        df.loc[model_name, "VISQOL"] = metrics["VISQOL"]
        _save(df)

    return metrics

def evaluate_und(
    text_pairs: List[List[str]],
    model_name: str,
    gpt_model: str = "gpt-5-mini",
    api_key: str | None = None,
    output_dir: str = "result/und_results.csv",
    **kwargs
):
    result_path = Path(output_dir)
    if result_path.exists():
        df = pd.read_csv(result_path, index_col=0)
        if df.index.name != "model_name":
            df.index.name = "model_name"
    else:
        df = pd.DataFrame()
        df.index.name = "model_name"

    metrics = {}

    def _save(df: pd.DataFrame):
        df.to_csv(result_path, index=True, float_format="%.4f")

    if "gpt-eval" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "gpt-eval"]):
        metrics["gpt-eval"] = float(df.loc[model_name, "gpt-eval"])
        metrics["f1"] = float(df.loc[model_name, "f1"])
        metrics["recall"] = float(df.loc[model_name, "recall"])
    else:
        from metric.gpt_eval import gpt_eval
        for pair in text_pairs:
            pair[0] = pair[0].split(":", 1)[1] if pair[0].startswith("Generate") else pair[0]
        refs = [pair[0] for pair in text_pairs]
        gens = [pair[1] for pair in text_pairs]
        results = gpt_eval(refs, gens, model=gpt_model, api_key=api_key)
        metrics["gpt-eval"] = round(float(results.get("score", 0)), 4)
        df.loc[model_name, "gpt-eval"] = metrics["gpt-eval"]
        metrics["f1"] = round(float(results.get("f1", 0)), 4)
        df.loc[model_name, "f1"] = metrics["f1"]
        metrics["recall"] = round(float(results.get("recall", 0)), 4)
        df.loc[model_name, "recall"] = metrics["recall"]
        _save(df)

    return metrics


# -----------------------------
# Additional task evaluators
# -----------------------------

def compute_sisnr(samples: List[Sample], *, gen_last_seconds: Optional[float] = None, verbose: bool = True,
                 crop_start: Optional[float] = None, crop_end: Optional[float] = None, 
                 skip_low_energy: bool = False, energy_threshold: float = 1e-4) -> float:
    """Compute average SI-SNR using tail-aligned waveforms.
    
    Args:
        samples: List of Sample objects with gen_path and gt_path
        gen_last_seconds: If set, crop to last N seconds of gen (and match GT length)
        verbose: If True, print per-sample SI-SNR scores
        crop_start: If set, crop gen from this second (overrides gen_last_seconds)
        crop_end: If set, crop gen to this second (used with crop_start)
        skip_low_energy: If True, skip samples where GT energy is too low (for drum separation)
        energy_threshold: RMS energy threshold for skipping low-energy GT samples
    
    Returns:
        Average SI-SNR across all valid samples
    """
    from metric.bss import bss_batch

    gen_groups: List[List[tuple]] = []
    ref_groups: List[List[tuple]] = []
    sample_ids: List[str] = []
    
    for s in samples:
        try:
            aligned = _load_and_align_audio(s, tail_seconds_hint=gen_last_seconds, 
                                          crop_start=crop_start, crop_end=crop_end)
            if aligned is None:
                if verbose:
                    print(f"[SI-SNR] Sample {s.id}: SKIPPED (alignment failed)")
                continue
            
            # Skip low-energy GT samples (for drum separation where some tracks have minimal drums)
            if skip_low_energy:
                gt_rms = np.sqrt(np.mean(aligned.ref ** 2))
                if gt_rms < energy_threshold:
                    if verbose:
                        print(f"[SI-SNR] Sample {s.id}: SKIPPED (low GT energy: RMS={gt_rms:.6f} < {energy_threshold})")
                    continue
            
            gen_tail_dur = aligned.gen_tail.shape[-1] / aligned.sr
            ref_dur = aligned.ref.shape[-1] / aligned.sr
            
            if verbose:
                gt_rms = np.sqrt(np.mean(aligned.ref ** 2))
                print(f"[SI-SNR] Sample {s.id}: gen_tail={gen_tail_dur:.2f}s, ref={ref_dur:.2f}s, sr={aligned.sr}, gt_rms={gt_rms:.6f}")
            
            gen_groups.append([(aligned.gen_tail, aligned.sr)])
            ref_groups.append([(aligned.ref, aligned.sr)])
            sample_ids.append(s.id)
        except Exception as exc:
            print(f"[SI-SNR] Sample {s.id}: ERROR - {exc}")
    
    if not gen_groups:
        print("[SI-SNR] No valid samples for SI-SNR computation")
        return float("nan")
    
    print(f"[SI-SNR] Computing SI-SNR for {len(gen_groups)} samples...")
    res = bss_batch(gen_groups, ref_groups, mono=True, permute=False)
    
    # Print per-sample results if available
    if verbose and "per_sample_si_snr" in res:
        print("\n[SI-SNR] Per-sample results:")
        per_sample = res["per_sample_si_snr"]
        for i, (sample_id, score) in enumerate(zip(sample_ids, per_sample)):
            print(f"  {sample_id}: {score:.4f} dB")
    
    global_avg = float(res.get("global_avg_si_snr", float("nan")))
    print(f"[SI-SNR] Global average: {global_avg:.4f} dB")
    
    return global_avg


def compute_onset_f(
    samples: List[Sample],
    sr: int | None = None,
    hop_length: int = 512,
    backtrack: bool = True,
    win: float = 0.05,
    gen_last_seconds: Optional[float] = None,
    crop_start: Optional[float] = None,
    crop_end: Optional[float] = None,
) -> float:
    """Compute average onset F-measure over aligned segments.

    Priority of segment selection:
      1. If crop_start/crop_end provided, use explicit segment.
      2. Else if gen_last_seconds provided, take tail of generated audio.
      3. Else use full length (matched to ref).
    """
    from metric.onset import onset_f_from_audio
    fs: List[float] = []
    for s in samples:
        try:
            aligned = _load_and_align_audio(s, tail_seconds_hint=gen_last_seconds,
                                           crop_start=crop_start, crop_end=crop_end)
            if aligned is None:
                fs.append(np.nan); continue
            res = onset_f_from_audio(
                (aligned.ref, aligned.sr),
                (aligned.gen_tail, aligned.sr),
                sr=aligned.sr if sr is None else sr,
                hop_length=hop_length,
                backtrack=backtrack,
                win=win,
            )
            fs.append(float(res.get("F", np.nan)))
        except Exception as exc:
            print(f"[OnsetF] Failed for sample {s.id}: {exc}"); fs.append(np.nan)
    valid = [x for x in fs if np.isfinite(x)]
    if not valid:
        print("[OnsetF] No valid pairs. Check GT/gen alignment.")
    return float(np.mean(valid)) if valid else float("nan")


def evaluate_sep(
    samples: List[Sample],
    model_name: str,
    output_dir: str = "result/sep_results.csv",
    skip_low_energy: bool = True,
    energy_threshold: float = 1e-4,
    **kwargs,
) -> dict:
    """Source separation evaluation (SI-SNR) using fixed segment layout.

    Segments:
      Input  : [0.0, 5.0)
      GenEval: [5.2, 10.2)  (model generated region compared against GT drum track)
    """
    result_path = Path(output_dir)
    if result_path.exists():
        df = pd.read_csv(result_path, index_col=0)
        if df.index.name != "model_name":
            df.index.name = "model_name"
    else:
        df = pd.DataFrame(); df.index.name = "model_name"

    metrics: Dict[str, float] = {}

    def _save(dframe: pd.DataFrame):
        dframe.to_csv(result_path, index=True, float_format="%.4f")

    if "SI-SNR" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "SI-SNR"]):
        metrics["SI-SNR"] = float(df.loc[model_name, "SI-SNR"])
    else:
        _print_seg_layout_once()
        val = round(float(compute_sisnr(
            samples,
            crop_start=SEP_GEN_START,
            crop_end=SEP_GEN_END,
            verbose=True,
            skip_low_energy=skip_low_energy,
            energy_threshold=energy_threshold,
        )), 4)
        df.loc[model_name, "SI-SNR"] = val
        metrics["SI-SNR"] = val
        _save(df)

    return metrics


def evaluate_infill(
    samples: List[Sample],
    model_name: str,
    aes_ckpt: Path | None = None,
    clap_model_name: str = "630k-audioset-fusion-best.pt",
    openl3_cfg: dict | None = None,
    passt_cfg: dict | None = None,
    onset_cfg: dict | None = None,
    output_dir: str = "result/infill_results.csv",
    **kwargs,
) -> dict:
    """Infill evaluation (FAD, ViSQOL, Onset-F, AES-gain) using fixed segments.

    Segments:
      Input   : [0.0, 5.0)
      Gen mid : [5.2, 10.2)  (not directly used yet)
      Result  : [10.4, 15.4) (used for metrics vs GT)
      AES diff: Input vs Result
    """
    result_path = Path(output_dir)
    if result_path.exists():
        df = pd.read_csv(result_path, index_col=0)
        if df.index.name != "model_name":
            df.index.name = "model_name"
    else:
        df = pd.DataFrame(); df.index.name = "model_name"

    metrics: Dict[str, float] = {}

    def _save(dframe: pd.DataFrame):
        dframe.to_csv(result_path, index=True, float_format="%.4f")

    aligned_cache: List[Tuple[Sample, AlignedAudio]] | None = None

    def _get_aligned_cache() -> List[Tuple[Sample, AlignedAudio]]:
        nonlocal aligned_cache
        if aligned_cache is None:
            aligned_cache = []
            for smp in samples:
                data = _load_and_align_audio(smp, crop_start=INFILL_RESULT_START, crop_end=INFILL_RESULT_END)
                if data is not None:
                    aligned_cache.append((smp, data))
        return aligned_cache

    if "FAD" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "FAD"]):
        metrics["FAD"] = float(df.loc[model_name, "FAD"])
    else:
        aligned = _get_aligned_cache()
        if not aligned:
            print("[Infill FAD] No aligned segments available.")
            metrics["FAD"] = float("nan")
        else:
            gen_inputs = [(data.gen_tail, data.sr) for _, data in aligned]
            ref_inputs = [(data.ref, data.sr) for _, data in aligned]
            metrics["FAD"] = round(
                float(compute_openl3_fad(samples, openl3_cfg or {}, gen_inputs=gen_inputs, ref_inputs=ref_inputs)),
                4,
            )
        df.loc[model_name, "FAD"] = metrics["FAD"]; _save(df)

    # ViSQOL（固定 result 段）
    if "VISQOL" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "VISQOL"]):
        metrics["VISQOL"] = float(df.loc[model_name, "VISQOL"])
    else:
        aligned = _get_aligned_cache()
        if not aligned:
            metrics["VISQOL"] = float("nan")
        else:
            with tempfile.TemporaryDirectory(prefix="infill_visqol_") as td:
                seg_samples: List[Sample] = []
                for smp, data in aligned:
                    try:
                        gen_seg_path = Path(td) / f"{smp.id}_result.wav"
                        gt_seg_path  = Path(td) / f"{smp.id}_gt.wav"
                        sf.write(str(gen_seg_path), data.gen_tail, data.sr)
                        sf.write(str(gt_seg_path), data.ref, data.sr)
                        seg_samples.append(Sample(id=smp.id, text=smp.text, gt_path=gt_seg_path, gen_path=gen_seg_path))
                    except Exception as exc:
                        print(f"[ViSQOL] Failed segment write for sample {smp.id}: {exc}")
                metrics["VISQOL"] = round(float(compute_visqol(seg_samples)) if seg_samples else float("nan"), 4)
        df.loc[model_name, "VISQOL"] = metrics["VISQOL"]; _save(df)

    # Onset-F（固定 result 段）
    if "OnsetF" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "OnsetF"]):
        metrics["OnsetF"] = float(df.loc[model_name, "OnsetF"])
    else:
        metrics["OnsetF"] = round(float(compute_onset_f(samples, crop_start=INFILL_RESULT_START, crop_end=INFILL_RESULT_END, **(onset_cfg or {}))), 4)
        df.loc[model_name, "OnsetF"] = metrics["OnsetF"]; _save(df)

    # AES-gain
    if aes_ckpt is not None:
        for col in ("AES_PQ_gain", "AES_CE_gain"):
            if col not in df.columns:
                df[col] = np.nan
        # cache check
        has_aes_cached = (
            model_name in df.index and
            all(col in df.columns and pd.notna(df.loc[model_name, col]) for col in ("AES_PQ_gain", "AES_CE_gain"))
        )
        if has_aes_cached:
            metrics["AES_PQ_gain"] = float(df.loc[model_name, "AES_PQ_gain"])
            metrics["AES_CE_gain"] = float(df.loc[model_name, "AES_CE_gain"])
        else:
            with tempfile.TemporaryDirectory(prefix="infill_aes_") as td:
                aes_samples: List[Sample] = []
                aligned = _get_aligned_cache()
                for smp, data in aligned:
                    try:
                        total = data.full_gen
                        sr0 = data.sr
                        
                        # Infill AES-gain: compare 0-5s (input) vs 10.4-15.4s (output)
                        input_len = int(5.0 * sr0)
                        output_start = int(10.4 * sr0)
                        output_end = int(15.4 * sr0)
                        
                        if total.shape[-1] < output_end:
                            print(f"[Infill AES] Generated clip too short for sample {smp.id} (need >15.4s)")
                            continue
                        
                        y_in = total[:input_len]
                        y_out = total[output_start:output_end]
                        
                        if y_in.size == 0 or y_out.size == 0:
                            print(f"[Infill AES] Empty segment for sample {smp.id}")
                            continue
                        in_p = Path(td) / (smp.gen_path.stem + "_0-5s.wav")
                        out_p = Path(td) / (smp.gen_path.stem + "_10.4-15.4s.wav")
                        sf.write(str(in_p), y_in, sr0)
                        sf.write(str(out_p), y_out, sr0)
                        aes_samples.append(Sample(id=smp.id, text=smp.text, gt_path=smp.gt_path, gen_path=out_p, input_path=in_p))
                    except Exception as exc:
                        print(f"[Infill AES] Failed for sample {smp.id}: {exc}")
                        continue
                if aes_samples:
                    gains = compute_aes_gain(aes_samples, aes_ckpt)
                    pq = round(float(gains.get("PQ_gain", np.nan)), 4)
                    ce = round(float(gains.get("CE_gain", np.nan)), 4)
                    df.loc[model_name, "AES_PQ_gain"] = pq
                    df.loc[model_name, "AES_CE_gain"] = ce
                    metrics["AES_PQ_gain"], metrics["AES_CE_gain"] = pq, ce
                    _save(df)

    return metrics


def evaluate_key(
    samples: List[Sample],
    model_name: str,
    output_dir: str = "result/key_results.csv",
    **kwargs,
) -> dict:
    """Key change evaluation: SI-SNR and F0 metrics (GPE/FPE/RPA)."""
    result_path = Path(output_dir)
    if result_path.exists():
        df = pd.read_csv(result_path, index_col=0)
        if df.index.name != "model_name":
            df.index.name = "model_name"
    else:
        df = pd.DataFrame(); df.index.name = "model_name"

    metrics: Dict[str, float] = {}

    def _save(dframe: pd.DataFrame):
        dframe.to_csv(result_path, index=True, float_format="%.4f")

    # SI-SNR
    if "SI-SNR" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "SI-SNR"]):
        metrics["SI-SNR"] = float(df.loc[model_name, "SI-SNR"])
    else:
        val = round(float(compute_sisnr(samples, gen_last_seconds=5.0)), 4)
        df.loc[model_name, "SI-SNR"] = val
        metrics["SI-SNR"] = val
        _save(df)

    # F0 metrics (cache-aware)
    existing_f0_cols = [c for c in df.columns if c.startswith("F0_")]
    f0_cached = (
        model_name in df.index and existing_f0_cols and all(pd.notna(df.loc[model_name, c]) for c in existing_f0_cols)
    )
    if not f0_cached:
        from metric.f0 import eval_pitch_metrics
        gen_inputs: List[tuple] = []
        ref_inputs: List[tuple] = []
        for s in samples:
            aligned = _load_and_align_audio(s, tail_seconds_hint=5.0)
            if aligned is None:
                continue
            gen_inputs.append((aligned.gen_tail, aligned.sr))
            ref_inputs.append((aligned.ref, aligned.sr))
        if not gen_inputs:
            print("[F0] No aligned segments; skipping F0 metrics.")
        else:
            f0_res = eval_pitch_metrics(gen_inputs, ref_inputs)
            for k, v in f0_res.items():
                if isinstance(v, float) and np.isfinite(v):
                    df.loc[model_name, f"F0_{k}"] = round(v, 4)
                    metrics[f"F0_{k}"] = round(v, 4)
        _save(df)

    return metrics


def evaluate_filter(
    samples: List[Sample],
    model_name: str,
    output_dir: str = "result/filter_results.csv",
    **kwargs,
) -> dict:
    """Filter task evaluation: SI-SNR."""
    result_path = Path(output_dir)
    if result_path.exists():
        df = pd.read_csv(result_path, index_col=0)
        if df.index.name != "model_name":
            df.index.name = "model_name"
    else:
        df = pd.DataFrame(); df.index.name = "model_name"

    metrics: Dict[str, float] = {}

    def _save(dframe: pd.DataFrame):
        dframe.to_csv(result_path, index=True, float_format="%.4f")

    if "SI-SNR" in df.columns and model_name in df.index and pd.notna(df.loc[model_name, "SI-SNR"]):
        metrics["SI-SNR"] = float(df.loc[model_name, "SI-SNR"])
    else:
        val = round(float(compute_sisnr(samples, gen_last_seconds=5.0)), 4)
        df.loc[model_name, "SI-SNR"] = val
        metrics["SI-SNR"] = val
        _save(df)

    return metrics
