import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertConfig, AutoTokenizer
from accelerate import Accelerator
# Direct relative imports from the local clamp3/code package
from .clamp3.code.config import *
from .clamp3.code.utils import CLaMP3Model

# For feature extraction
import sys
import importlib.util

def extract_features_for_wavs(wav_paths, output_dir, model_path, mean_features=True):
	"""
	For a list of wav_paths, extract features using extract_mert.py (via subprocess),
	and save them to output_dir. Only process files that are not already cached.
	"""
	import sys
	import tempfile
	import subprocess
	from pathlib import Path
	os.makedirs(output_dir, exist_ok=True)

	with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
		for w in wav_paths:
			f.write(w + "\n")
		file_list_path = f.name

	# Call extract_mert.py as a subprocess
	extract_mert_path = os.path.join(os.path.dirname(__file__), "clamp3", "preprocessing", "audio", "extract_mert.py")
	cmd = [
		sys.executable, extract_mert_path,
		"--file_list", file_list_path,
		"--output_dir", output_dir,
		"--mert_model_path", model_path,
		"--mean_features"
	]
	print(f"[CLaMP3] Extracting features for {len(wav_paths)} wavs via subprocess (overwrite mode)...")
	subprocess.run(cmd, check=True)
	os.remove(file_list_path)
	return [os.path.join(output_dir, os.path.splitext(os.path.basename(wav))[0] + '.npy') for wav in wav_paths]

def _load_clamp3_model(device=None):
	audio_config = BertConfig(vocab_size=1,
							hidden_size=AUDIO_HIDDEN_SIZE,
							num_hidden_layers=AUDIO_NUM_LAYERS,
							num_attention_heads=AUDIO_HIDDEN_SIZE//64,
							intermediate_size=AUDIO_HIDDEN_SIZE*4,
							max_position_embeddings=MAX_AUDIO_LENGTH)
	symbolic_config = BertConfig(vocab_size=1,
								hidden_size=M3_HIDDEN_SIZE,
								num_hidden_layers=PATCH_NUM_LAYERS,
								num_attention_heads=M3_HIDDEN_SIZE//64,
								intermediate_size=M3_HIDDEN_SIZE*4,
								max_position_embeddings=PATCH_LENGTH)
	model = CLaMP3Model(audio_config=audio_config,
						symbolic_config=symbolic_config,
						text_model_name=TEXT_MODEL_NAME,
						hidden_size=CLAMP3_HIDDEN_SIZE,
						load_m3=CLAMP3_LOAD_M3)
	if device is None:
		accelerator = Accelerator()
		device = accelerator.device
	model = model.to(device)
	# Load weights
	checkpoint_path = CLAMP3_WEIGHTS_PATH
	if not os.path.exists(checkpoint_path):
		# Download weights if not present
		import requests
		checkpoint_url = "https://huggingface.co/sander-wood/clamp3/resolve/main/" + os.path.basename(CLAMP3_WEIGHTS_PATH)
		response = requests.get(checkpoint_url, stream=True)
		response.raise_for_status()
		with open(checkpoint_path, "wb") as f:
			for chunk in response.iter_content(chunk_size=8192):
				if chunk:
					f.write(chunk)
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
	model.load_state_dict(checkpoint['model'])
	model.eval()
	return model, device

def _extract_text_feature(text, model, tokenizer, device):
	item = tokenizer.sep_token.join([t for t in text.split('\n') if t.strip()])
	input_data = tokenizer(item, return_tensors="pt")['input_ids'].squeeze(0)
	input_masks = torch.ones(input_data.size(0), dtype=torch.long)
	pad_len = MAX_TEXT_LENGTH - input_data.size(0)
	if pad_len > 0:
		input_data = torch.cat([input_data, torch.ones(pad_len, dtype=torch.long)*tokenizer.pad_token_id], 0)
		input_masks = torch.cat([input_masks, torch.zeros(pad_len, dtype=torch.long)], 0)
	input_data = input_data[:MAX_TEXT_LENGTH]
	input_masks = input_masks[:MAX_TEXT_LENGTH]
	with torch.no_grad():
		feat = model.get_text_features(input_data.unsqueeze(0).to(device), input_masks.unsqueeze(0).to(device), get_global=True)
	return feat.squeeze(0).cpu().numpy()

def _extract_audio_feature(wav_path, model, device):
	if not str(wav_path).endswith('.npy'):
		raise ValueError(f"Only .npy feature files are supported for CLaMP3 scoring. Got: {wav_path}")
	arr = np.load(wav_path)
	if arr.ndim == 1 and arr.shape[0] == 768:
		arr = arr[None, :]  # (1, 768)
	elif arr.ndim == 2 and arr.shape[1] == 768:
		pass
	else:
		raise ValueError(f"Unexpected feature shape: {arr.shape}, expected (N,768) or (768,)")
	arr_tensor = torch.tensor(arr, dtype=torch.float32).to(device)
	audio_len = arr_tensor.shape[0]
	audio_masks = torch.ones(audio_len, dtype=torch.float32).to(device)
	if audio_len < MAX_AUDIO_LENGTH:
		pad_len = MAX_AUDIO_LENGTH - audio_len
		arr_tensor = torch.cat([arr_tensor, torch.zeros(pad_len, arr_tensor.shape[1], device=device)], dim=0)
		audio_masks = torch.cat([audio_masks, torch.zeros(pad_len, device=device)], dim=0)
	else:
		arr_tensor = arr_tensor[:MAX_AUDIO_LENGTH]
		audio_masks = audio_masks[:MAX_AUDIO_LENGTH]
	with torch.no_grad():
		pooled = model.get_audio_features(audio_inputs=arr_tensor.unsqueeze(0), audio_masks=audio_masks.unsqueeze(0), get_global=True)
	pooled = pooled.squeeze(0).cpu().numpy()  # (768,)
	return pooled

def clamp3_score(texts, gen_files):
	"""
	Compute CLaMP3 score between a list of texts and a list of generated audio .npy files.
	Returns the mean cosine similarity between text and audio features.
	"""
	assert len(texts) == len(gen_files) and len(texts) > 0
	accelerator = Accelerator()
	device = accelerator.device
	model, device = _load_clamp3_model(device)
	from transformers import AutoTokenizer
	tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
	if all(str(f).endswith('.wav') for f in gen_files):
		model_path = os.environ.get('CLAMP3_AUDIO_MODEL', 'm-a-p/MERT-v1-95M')
		cache_dir = os.environ.get('CLAMP3_CACHE_DIR', './cache/clamp3')
		gen_files = extract_features_for_wavs(gen_files, cache_dir, model_path, mean_features=True)
	text_embs = []
	audio_embs = []
	for text in tqdm(texts, desc='CLaMP3 text features'):
		text_embs.append(_extract_text_feature(text, model, tokenizer, device))
	for npy in gen_files:
		audio_embs.append(_extract_audio_feature(npy, model, device))

	sims = []
	total_query = 0
	for t, a in zip(text_embs, audio_embs):
		t_tensor = torch.tensor(t, dtype=torch.float32)
		if isinstance(a, np.ndarray) and a.ndim == 2:
			for seg in a:
				seg_tensor = torch.tensor(seg, dtype=torch.float32)
				sim = torch.nn.functional.cosine_similarity(seg_tensor, t_tensor, dim=0).item()
				sims.append(sim)
				total_query += 1
		else:  # (768,)
			a_tensor = torch.tensor(a, dtype=torch.float32)
			sim = torch.nn.functional.cosine_similarity(a_tensor, t_tensor, dim=0).item()
			sims.append(sim)
			total_query += 1
	avg_sim = float(np.mean(sims)) if sims else 0.0
	print(f"Total query features: {total_query}")
	print(f"Total reference features: {len(audio_embs)}")
	print(f"Avg. pairwise similarity: {round(avg_sim, 4)}")
	return avg_sim
