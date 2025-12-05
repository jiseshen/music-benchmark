"""Batch music caption/label baseline with Qwen2-Audio.

Outputs a single JSON file `captions.json` containing a list of pairs:
[
    [ground_truth_text_0, raw_model_output_0],
    [ground_truth_text_1, raw_model_output_1],
    ...
]

Assumptions:
- Audio files are named like `audio_{i}.wav` matching dataset index i in
    HF dataset b583x2/MMIO_tasks_benchmark split test_eval.
- We intentionally keep the model raw output string (even if it looks like JSON) without parsing.

Usage example:
    python baseline/qwen2audio/batch_label_infer.py \
            --limit 20 --output tmp_output --max-new-tokens 256
"""

import os, json, argparse, re
from pathlib import Path
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm.auto import tqdm
import librosa
from datasets import load_dataset


def build_prompt() -> str:
    return (
        "Please listen to the following music and provide 0-2 concise labels for genre, mood, and instrument."
        "You may leave blank any category if unsure."
        "Return ONLY JSON in the format: {\"genre\": [...], \"mood\": [...], \"instrument\": [...]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Batch label inference (simplified)")
    parser.add_argument('--limit', type=int, default=None, help='Number of WAV files to process')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--audio-dir', type=str, default='/home/jianzhis/MMIO/test_gt/', help='Directory containing numbered *.wav files')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--captions-filename', type=str, default='captions.json', help='Name of the consolidated output file')
    args = parser.parse_args()

    debug = args.debug or (os.environ.get('DEBUG', '0') in ('1', 'true', 'True'))

    audio_dir = Path(args.audio_dir)
    wav_files = sorted(audio_dir.glob('*.wav'))
    if not wav_files:
        print(f"[ERROR] No wav files found in {audio_dir}")
        return

    target_len = min(args.limit, len(wav_files)) if args.limit else len(wav_files)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using {target_len}/{len(wav_files)} files from {audio_dir}")

    # Model + processor
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2-Audio-7B-Instruct')
    model = Qwen2AudioForConditionalGeneration.from_pretrained('Qwen/Qwen2-Audio-7B-Instruct', device_map='auto')

    target_sr = getattr(getattr(processor, 'feature_extractor', None), 'sampling_rate', 16000)
    if debug:
        print(f"[DEBUG] Target sampling rate: {target_sr}")

    # Load HF dataset for ground truth text
    DATASET_ID = "b583x2/MMIO_tasks_benchmark"
    DATASET_SPLIT = "test_eval"
    hf_token = os.getenv("HF_TOKEN")
    try:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, token=hf_token, download_mode="force_redownload")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset {DATASET_ID}:{DATASET_SPLIT}: {e}")
        return
    print(dataset[3]["text"])
    results = []  # list of [gt, raw_output]
    processed = 0

    pattern = re.compile(r'audio_(\d+)$')
    for wav_path in tqdm(wav_files[:target_len], desc='Infer'):
        sample_id = wav_path.stem  # expects audio_{i}
        m = pattern.match(sample_id)
        if not m:
            if debug:
                print(f"[WARN] Filename {wav_path.name} does not match audio_{{i}} pattern; skipping")
            continue
        idx = int(m.group(1))
        if idx >= len(dataset):
            if debug:
                print(f"[WARN] Index {idx} out of dataset range {len(dataset)}; skipping")
            continue
        gt_text = dataset[idx].get('text', '') or ''
        try:
            audio, sr = librosa.load(wav_path, sr=target_sr)
        except Exception as e:
            print(f"[WARN] Failed to load {wav_path}: {e}")
            continue
        if debug:
            print(f"[DEBUG] {wav_path.name}: sr={sr} samples={len(audio)}")
        conversation = [
            {"role": "system", "content": "You are a music expert assistant."},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": build_prompt()},
            ]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio], sampling_rate=target_sr, return_tensors='pt', padding=True)
        for k, v in inputs.items():
            if hasattr(v, 'to'):
                inputs[k] = v.to(model.device)
        gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        gen_ids = gen_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        results.append([gt_text, response])
        processed += 1
        if debug and processed % 10 == 0:
            print(f"[DEBUG] Processed {processed} items...")

    captions_path = output_dir / args.captions_filename
    with open(captions_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[SUMMARY] processed={processed} written={captions_path} dataset_size={len(dataset)}")


if __name__ == '__main__':
    main()