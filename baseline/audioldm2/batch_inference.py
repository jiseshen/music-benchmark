#!/usr/bin/env python3
"""AudioLDM2 batch text-to-music baseline inference.

Reference: https://huggingface.co/cvssp/audioldm2#checkpoint-details

Design goals:
  - Simple, deterministic baseline over the same benchmark dataset
	(b583x2/MMIO_tasks_benchmark split 'test_eval').
  - Minimal dependencies beyond diffusers + datasets + scipy.
  - Batched generation for throughput while keeping reproducibility.
  - Saves each audio clip to output_dir/<index>.wav (float32 PCM to 16-bit WAV).
  - Provides timing + failure stats.

Key differences vs MuMu_LLaMA batch script:
  - Directly uses AudioLDM2Pipeline (text prompt only) without LLaMA wrapper.
  - Guidance scale and num_inference_steps exposed as constants.

You can adjust constants below. For quick experiments keep batch size small if VRAM limited.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from diffusers import AudioLDM2Pipeline
import scipy.io.wavfile as wavfile
import numpy as np

# ---------------- Configuration Constants ---------------- #
DATASET_ID = "b583x2/MMIO_tasks_benchmark"
DATASET_SPLIT = "test_eval"
MODEL_ID = "cvssp/audioldm2-music"  # HF model repo
OUTPUT_DIR = Path("gen_output")
SEED = 42
BATCH_SIZE = 4                 # Set to 1 if OOM
NUM_INFERENCE_STEPS = 200      # As per model card typical usage
GUIDANCE_SCALE = 3.5           # Typical classifier-free guidance value for music variant
AUDIO_SECONDS = 5              # Target length; pipeline respects length via internal config
NUM_WAVEFORMS_PER_PROMPT = 1   # Keep 1 for baseline comparability
FP16 = True                    # Use fp16 if GPU available
TOKEN_ENV = "HF_TOKEN_BENCH"   # Optional auth token env var
PRINT_EVERY = 25               # Progress logging interval
# --------------------------------------------------------- #


def prepare_pipeline() -> AudioLDM2Pipeline:
	auth_token = os.environ.get(TOKEN_ENV, None)
	torch_dtype = torch.float16 if (torch.cuda.is_available() and FP16) else torch.float32
	pipe = AudioLDM2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype, use_auth_token=auth_token)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	pipe = pipe.to(device)
	pipe.set_progress_bar_config(disable=True)
	return pipe


def save_audio(waveform: torch.Tensor, sample_rate: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # waveform 可能是 (C, T)、(T,) 或 (T, C)
    data = waveform.detach().cpu().numpy()
    if data.ndim == 2:
        if data.shape[0] <= 4:   # (C, T) 常见情况
            data = data[0]      # 取第一声道
        elif data.shape[1] <= 4: # (T, C) -> 取第一声道
            data = data[:, 0]
        else:
            # 如果通道数很多，默认取第一个通道
            data = data[:, 0]
    
    max_val = max(1e-9, np.abs(data).max())
    data16 = (data / max_val).clip(-1, 1)
    data16 = (data16 * 32767).astype(np.int16)

    wavfile.write(str(out_path), int(sample_rate), data16)


def batch_generate(pipe: AudioLDM2Pipeline, prompts: List[str], generator: torch.Generator):
	audios = pipe(
		prompts,
		num_inference_steps=NUM_INFERENCE_STEPS,
		guidance_scale=GUIDANCE_SCALE,
		audio_length_in_s=AUDIO_SECONDS,
		num_waveforms_per_prompt=NUM_WAVEFORMS_PER_PROMPT,
		generator=generator,
	).audios
	# audios: list (batch) of list (waveforms per prompt) of ndarray (float32)
	return audios


def main():
	if not torch.cuda.is_available():
		print("[WARN] CUDA not available, falling back to CPU; this will be very slow.")
	print(f"[INFO] Loading dataset {DATASET_ID}:{DATASET_SPLIT}")
	auth_token = os.environ.get(TOKEN_ENV, None)
	dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, token=auth_token)
	print(f"[INFO] Dataset size: {len(dataset)}")

	pipe = prepare_pipeline()
	sr = getattr(pipe, 'audio_encoder', getattr(pipe, 'feature_extractor', None))
	sample_rate = None
	# Robust way: pipeline outputs known sample rate (currently 16000 for music model)
	try:
		sample_rate = pipe.vae.config.sample_rate  # if available
	except Exception:
		sample_rate = 16000
	print(f"[INFO] Using sample rate: {sample_rate}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	generator = torch.Generator(device=pipe.device).manual_seed(SEED)

	total = len(dataset)
	failures = 0
	start = time.time()
	for start_idx in range(0, total, BATCH_SIZE):
		end_idx = min(total, start_idx + BATCH_SIZE)
		batch_prompts = [dataset[i]["text"] for i in range(start_idx, end_idx)]
		batch_audios = batch_generate(pipe, batch_prompts, generator)
		print(batch_audios.shape)
		# Save first waveform per prompt
		for local_i, audio_group in enumerate(batch_audios):
			print(audio_group.shape)
			if len(audio_group) == 0:
				failures += 1
				continue
			audio_np = torch.tensor(audio_group)  # convert ndarray -> tensor for uniform handling
			out_path = OUTPUT_DIR / f"{start_idx + local_i}.wav"
			save_audio(audio_np, sample_rate, out_path)
		if (start_idx // BATCH_SIZE) % max(1, PRINT_EVERY) == 0:
			done = end_idx
			elapsed = time.time() - start
			print(f"[PROGRESS] {done}/{total} ({100*done/total:.1f}%) | fail={failures} | elapsed={elapsed/60:.2f}m")

	elapsed = time.time() - start
	print(f"[DONE] Generated {total - failures}/{total} clips. Failures={failures}. Time={elapsed/60:.2f} min")


if __name__ == "__main__":
	main()

