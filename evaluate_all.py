from datasets import load_dataset, Dataset
from evaluate import (
    evaluate_gen,
    evaluate_und,
    evaluate_sep,
    evaluate_infill,
    evaluate_key,
    evaluate_filter,
    Sample,
)
import re
from pathlib import Path
import json
import os
import argparse

DATASET = load_dataset("b583x2/MMIO_tasks_benchmark", token=os.getenv("HF_TOKEN"), split="test_eval")
DEFAULT_BASE_PATH = Path("gt")
AES_CKPT_PATH = Path("cache/aes_checkpoint.pt")
AES_CKPT_URL = "https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt"


def _resolve_output_path(dir_path: Path, i: int | str) -> Path | None:
    """Return a Path to the generated audio file for index i within dir_path.

    Supported patterns (case-insensitive extension):
    - audio_{i}.wav
    - sample_{i}.wav
    - {i}.wav
    - 0...0{i}.wav (zero-padded numeric filename)
    If multiple candidates exist, prefer audio_*, then sample_*, then numeric-only.
    """
    try:
        idx = str(int(i))
    except Exception:
        idx = str(i)

    # Try direct, common filenames first (lowercase and uppercase extensions)
    direct_names = [
        f"audio_{idx}.wav", f"audio_{idx}.WAV",
        f"sample_{idx}.wav", f"sample_{idx}.WAV",
        f"{idx}.wav", f"{idx}.WAV",
    ]
    for name in direct_names:
        p = dir_path / name
        if p.exists():
            return p

    # Fallback: search by regex on stem with optional prefixes and zero-padding
    stem_re = re.compile(rf"^(?:audio_|sample_)?0*{re.escape(idx)}$", re.IGNORECASE)
    candidates: list[Path] = []
    for pattern in ("*.wav", "*.WAV"):
        for p in dir_path.glob(pattern):
            try:
                if stem_re.match(p.stem):
                    candidates.append(p)
            except Exception:
                continue

    if not candidates:
        return None

    # Rank: audio_* > sample_* > numeric-only; then lexicographically for stability
    def rank(path: Path) -> tuple[int, str]:
        s = path.stem.lower()
        if s.startswith("audio_"):
            pr = 0
        elif s.startswith("sample_"):
            pr = 1
        else:
            pr = 2
        return (pr, s)

    candidates.sort(key=rank)
    return candidates[0]

def _ensure_aes_ckpt():
    if not AES_CKPT_PATH.exists():
        print(f"[INFO] AES checkpoint not found at {AES_CKPT_PATH}, downloading...")
        AES_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(AES_CKPT_URL, AES_CKPT_PATH)
        print(f"[INFO] Downloaded AES checkpoint to {AES_CKPT_PATH}")
    else:
        print(f"[INFO] Using AES checkpoint: {AES_CKPT_PATH}")

def evaluate_all(
    model_name,
    gen_path=None,
    und_path=None,
    sep_path=None,
    infill_path=None,
    key_path=None,
    filter_path=None,
    gt_base=None,
    gen_gt_path=None,
    infill_gt_path=None,
    key_gt_path=None,
    filter_gt_path=None,
    sep_gt_path=None,
    und_ref_path=None,
    dataset="mutask",
    **kwargs,
):
    _ensure_aes_ckpt()
    base = Path(gt_base) if gt_base else DEFAULT_BASE_PATH / dataset
    def _normalize_gt_dir(p: Path) -> Path:
        if p.exists():
            return p
        if (p / 'wav').exists():
            return p / 'wav'
        return p

    gen_gt = _normalize_gt_dir(Path(gen_gt_path) if gen_gt_path else (base / "orig"))
    infill_gt = _normalize_gt_dir(Path(infill_gt_path) if infill_gt_path else (base / "orig"))
    key_gt = _normalize_gt_dir(Path(key_gt_path) if key_gt_path else (base / "key"))
    filter_gt = _normalize_gt_dir(Path(filter_gt_path) if filter_gt_path else (base / "filter"))
    sep_gt = _normalize_gt_dir(Path(sep_gt_path) if sep_gt_path else (base / "drum"))

    def _warn_missing(name: str, p: Path):
        if not p.exists():
            print(f"[WARN] {name} GT path not found: {p}")

    _warn_missing("GEN", gen_gt)
    _warn_missing("INFILL", infill_gt)
    _warn_missing("KEY", key_gt)
    _warn_missing("FILTER", filter_gt)
    _warn_missing("SEP", sep_gt)
    print(f"[INFO] Using GT dirs:\n  GEN: {gen_gt}\n  INFILL: {infill_gt}\n  KEY: {key_gt}\n  FILTER: {filter_gt}\n  SEP: {sep_gt}")

    if gen_path:
        base = Path(gen_path)
        samples = []
        for i in range(len(DATASET)):
            gp = _resolve_output_path(base, i)
            if gp is None:
                continue
            samples.append(
                Sample(
                    id=str(i),
                    text=DATASET[i]["text"],
                    gt_path=gen_gt / f"audio_{i}.wav",
                    gen_path=gp,
                )
            )
        print(f"[INFO] GEN matched {len(samples)}/{len(DATASET)} files in {base}")
        evaluate_gen(
            samples,
            model_name=model_name,
            aes_ckpt=AES_CKPT_PATH,
            **kwargs,
        )
    if und_path:
        json_path = Path(und_path) / "captions.json"
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Build samples for CLaMP3 if und_ref_path is provided
        samples = None
        if und_ref_path:
            und_ref = _normalize_gt_dir(Path(und_ref_path))
            samples = []
            for i, pair in enumerate(data):
                ref_audio = und_ref / f"audio_{i}.wav"
                if ref_audio.exists():
                    samples.append(Sample(
                        id=str(i),
                        text=pair[0],
                        gt_path=ref_audio,
                        gen_path=ref_audio,
                    ))
            print(f"[INFO] UND matched {len(samples)}/{len(data)} audio files for CLaMP3")
        
        evaluate_und(data, model_name=model_name, samples=samples, **kwargs)

    # Source separation (SI-SNR) — fixed segment: generated [5.2,10.2) vs GT [0.0,5.0)
    if sep_path:
        gt_base_resolved = sep_gt if sep_gt.exists() else gen_gt
        base = Path(sep_path)
        samples = []
        for i in range(len(DATASET)):
            gp = _resolve_output_path(base, i)
            if gp is None:
                continue
            samples.append(
                Sample(
                    id=str(i),
                    text=DATASET[i]["text"],
                    gt_path=gt_base_resolved / f"audio_{i}.wav",
                    gen_path=gp,
                )
            )
            print(f"[DEBUG] SEP sample {i}: GT {gt_base_resolved / f'audio_{i}.wav'}, GEN {gp}"  )
        print(f"[INFO] SEP matched {len(samples)}/{len(DATASET)} files in {base}")
        evaluate_sep(samples, model_name=model_name, **kwargs)

    # Infill (FAD, ViSQOL, OnsetF, AES-gain) — fixed segments:
    #   Result segment [10.4,15.4) vs GT [0.0,5.0); AES diff compares input [0.0,5.0) vs result [10.4,15.4)
    if infill_path:
        base = Path(infill_path)
        samples = []
        for i in range(len(DATASET)):
            gp = _resolve_output_path(base, i)
            if gp is None:
                continue
            samples.append(
                Sample(
                    id=str(i),
                    text=DATASET[i]["text"],
                    gt_path=infill_gt / f"audio_{i}.wav",
                    gen_path=gp,
                )
            )
        print(f"[INFO] INFILL matched {len(samples)}/{len(DATASET)} files in {base}")
        evaluate_infill(samples, model_name=model_name, aes_ckpt=AES_CKPT_PATH, **kwargs)

    # Key change (SI-SNR + F0 metrics)
    if key_path:
        base = Path(key_path)
        samples = []
        for i in range(len(DATASET)):
            gp = _resolve_output_path(base, i)
            if gp is None:
                continue
            samples.append(
                Sample(
                    id=str(i),
                    text=DATASET[i]["text"],
                    gt_path=key_gt / f"audio_{i}.wav",
                    gen_path=gp,
                )
            )
        print(f"[INFO] KEY matched {len(samples)}/{len(DATASET)} files in {base}")
        evaluate_key(samples, model_name=model_name, **kwargs)

    # Filter (SI-SNR)
    if filter_path:
        base = Path(filter_path)
        samples = []
        for i in range(len(DATASET)):
            gp = _resolve_output_path(base, i)
            if gp is None:
                continue
            samples.append(
                Sample(
                    id=str(i),
                    text=DATASET[i]["text"],
                    gt_path=filter_gt / f"audio_{i}.wav",
                    gen_path=gp,
                )
            )
        print(f"[INFO] FILTER matched {len(samples)}/{len(DATASET)} files in {base}")
        evaluate_filter(samples, model_name=model_name, **kwargs)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Model name for logging")
    parser.add_argument("--gen-path", type=str, default=None, help="Path to generated audio files for generation evaluation")
    parser.add_argument("--und-path", type=str, default=None, help="Path to understanding results folder for understanding evaluation")
    parser.add_argument("--und-ref-path", type=str, default=None, help="Reference audio folder for UND CLaMP3 evaluation")
    parser.add_argument("--sep-path", type=str, default=None, help="Path to separated audio outputs for source separation evaluation")
    parser.add_argument("--infill-path", type=str, default=None, help="Path to infill generated audio outputs")
    parser.add_argument("--key-path", type=str, default=None, help="Path to key-change generated audio outputs")
    parser.add_argument("--filter-path", type=str, default=None, help="Path to filter generated audio outputs")
    # Ground-truth base/paths overrides
    parser.add_argument("--gt-base", type=str, default=None, help="Override base path for GT folders (defaults to gt/{dataset}")
    parser.add_argument("--gen-gt-path", type=str, default=None, help="Override GEN GT wav folder")
    parser.add_argument("--infill-gt-path", type=str, default=None, help="Override INFILL GT wav folder")
    parser.add_argument("--key-gt-path", type=str, default=None, help="Override KEY GT wav folder")
    parser.add_argument("--filter-gt-path", type=str, default=None, help="Override FILTER GT wav folder")
    parser.add_argument("--sep-gt-path", type=str, default=None, help="Override SEP GT wav folder")
    parser.add_argument("--dataset", type=str, default="mutask", help="Dataset name (mutask or musiccaps)")
    args = parser.parse_args()
    evaluate_all(
        model_name=args.model_name,
        gen_path=args.gen_path,
        und_path=args.und_path,
        und_ref_path=args.und_ref_path,
        sep_path=args.sep_path,
        infill_path=args.infill_path,
        key_path=args.key_path,
        filter_path=args.filter_path,
        gt_base=args.gt_base,
        gen_gt_path=args.gen_gt_path,
        infill_gt_path=args.infill_gt_path,
        key_gt_path=args.key_gt_path,
        filter_gt_path=args.filter_gt_path,
        sep_gt_path=args.sep_gt_path,
        dataset=args.dataset,
    )