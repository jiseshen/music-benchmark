#!/usr/bin/env python3
"""ACE-Step batch text-to-music inference baseline.

Loads the MMIO benchmark text prompts, runs ACE-Step with default UI-like
settings, and writes each generated clip under ``gen_output``. Designed for
reproducibility comparable to the other baseline scripts in this repo.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset

from acestep.pipeline_ace_step import ACEStepPipeline


DATASET_ID = "b583x2/MMIO_tasks_benchmark"
DATASET_SPLIT = "test_eval"
PROMPT_FIELD = "text"
OUTPUT_DIR = Path("gen_output")
FORMAT = "wav"
HF_TOKEN_ENV = "HF_TOKEN_BENCH"
DEFAULT_AUDIO_DURATION = 5.0
DEFAULT_INFER_STEPS = 60
DEFAULT_GUIDANCE_SCALE = 15.0
DEFAULT_OMEGA_SCALE = 10.0
DEFAULT_GUIDANCE_INTERVAL = 0.5
DEFAULT_GUIDANCE_INTERVAL_DECAY = 0.0
DEFAULT_MIN_GUIDANCE_SCALE = 3.0
DEFAULT_GUIDANCE_SCALE_TEXT = 0.0
DEFAULT_GUIDANCE_SCALE_LYRIC = 0.0
DEFAULT_REF_AUDIO_STRENGTH = 0.5
DEFAULT_CFG_TYPE = "apg"
DEFAULT_SCHEDULER = "euler"
DEFAULT_USE_ERG_TAG = True
DEFAULT_USE_ERG_LYRIC = False
DEFAULT_USE_ERG_DIFFUSION = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for ACE-Step baseline")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-split", default=DATASET_SPLIT)
    parser.add_argument("--prompt-field", default=PROMPT_FIELD)
    parser.add_argument("--lyrics-field", default=None, help="Optional dataset column to use as lyrics")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Directory with pre-downloaded checkpoints")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--audio-duration", type=float, default=DEFAULT_AUDIO_DURATION)
    parser.add_argument("--infer-steps", type=int, default=DEFAULT_INFER_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--omega-scale", type=float, default=DEFAULT_OMEGA_SCALE)
    parser.add_argument("--guidance-interval", type=float, default=DEFAULT_GUIDANCE_INTERVAL)
    parser.add_argument("--guidance-interval-decay", type=float, default=DEFAULT_GUIDANCE_INTERVAL_DECAY)
    parser.add_argument("--min-guidance-scale", type=float, default=DEFAULT_MIN_GUIDANCE_SCALE)
    parser.add_argument("--guidance-scale-text", type=float, default=DEFAULT_GUIDANCE_SCALE_TEXT)
    parser.add_argument("--guidance-scale-lyric", type=float, default=DEFAULT_GUIDANCE_SCALE_LYRIC)
    parser.add_argument("--cfg-type", default=DEFAULT_CFG_TYPE, choices=["cfg", "apg", "cfg_star"])
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER, choices=["euler", "heun", "pingpong"])
    parser.add_argument("--manual-seed", type=int, default=None, help="Base seed (per sample offset) for determinism")
    parser.add_argument("--overlap-decode", action="store_true", help="Enable overlapped DCAE decoding")
    parser.add_argument("--cpu-offload", action="store_true", help="Only load active stage on GPU")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--lora", default="none", help="LoRA repo or path to load (default: disabled)")
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true", help="Skip prompts with existing outputs")
    parser.add_argument("--format", default=FORMAT, choices=["wav", "flac", "mp3", "ogg"])
    parser.add_argument("--hf-token-env", default=HF_TOKEN_ENV)
    return parser.parse_args()


def iter_samples(dataset, limit: Optional[int] = None) -> Iterable[tuple[int, dict]]:
    total = len(dataset)
    n = min(total, limit) if limit is not None else total
    for idx in range(n):
        yield idx, dataset[idx]


def main() -> None:
    args = parse_args()
    auth_token = os.environ.get(args.hf_token_env)
    dataset = load_dataset(args.dataset_id, split=args.dataset_split, token=auth_token)

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_dir or None,
        device_id=args.device_id,
        dtype=args.dtype,
        torch_compile=args.torch_compile,
        cpu_offload=args.cpu_offload,
        overlapped_decode=args.overlap_decode,
    )

    total = args.max_samples if args.max_samples is not None else len(dataset)
    print(f"[INFO] Running ACE-Step on {total} prompts from {args.dataset_id}:{args.dataset_split}")
    start_time = time.time()
    processed = 0
    failures = 0
    skipped = 0

    for local_idx, sample in iter_samples(dataset, args.max_samples):
        prompt = sample.get(args.prompt_field, "")
        if not prompt:
            print(f"[WARN] Sample {local_idx} missing '{args.prompt_field}', skipping")
            failures += 1
            continue

        lyrics = ""
        if args.lyrics_field:
            lyrics = sample.get(args.lyrics_field, "") or ""

        out_path = args.output_dir / f"{local_idx:05d}.{args.format}"

        if args.skip_existing and out_path.exists():
            print(f"[SKIP] {out_path} exists")
            skipped += 1
            continue

        manual_seed = None
        if args.manual_seed is not None:
            manual_seed = args.manual_seed + local_idx

        processed += 1

        try:
            outputs = pipeline(
                format=args.format,
                audio_duration=args.audio_duration,
                prompt=prompt,
                lyrics=lyrics,
                infer_step=args.infer_steps,
                guidance_scale=args.guidance_scale,
                scheduler_type=args.scheduler,
                cfg_type=args.cfg_type,
                omega_scale=args.omega_scale,
                manual_seeds=None if manual_seed is None else str(manual_seed),
                guidance_interval=args.guidance_interval,
                guidance_interval_decay=args.guidance_interval_decay,
                min_guidance_scale=args.min_guidance_scale,
                use_erg_tag=DEFAULT_USE_ERG_TAG,
                use_erg_lyric=DEFAULT_USE_ERG_LYRIC,
                use_erg_diffusion=DEFAULT_USE_ERG_DIFFUSION,
                oss_steps=None,
                guidance_scale_text=args.guidance_scale_text,
                guidance_scale_lyric=args.guidance_scale_lyric,
                audio2audio_enable=False,
                ref_audio_strength=DEFAULT_REF_AUDIO_STRENGTH,
                ref_audio_input=None,
                lora_name_or_path=args.lora,
                lora_weight=args.lora_weight,
                save_path=str(out_path),
                batch_size=1,
            )
        except Exception as exc:  # pylint: disable=broad-except
            failures += 1
            print(f"[ERROR] Sample {local_idx} failed: {exc}")
            continue

        generated_path = outputs[0]
        print(f"[OK] {local_idx} -> {generated_path}")

    elapsed = time.time() - start_time
    completed = processed - failures
    print(
        f"[DONE] success={completed} failure={failures} skipped={skipped} "
        f"time={elapsed/60:.2f} min"
    )


if __name__ == "__main__":
    main()
