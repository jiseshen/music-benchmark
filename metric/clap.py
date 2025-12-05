import os, requests, torch, numpy as np, librosa, pyloudnorm as pyln
from tqdm import tqdm
import laion_clap
from laion_clap.clap_module.factory import load_state_dict

def int16_to_float32(x): return (x / 32767.0).astype(np.float32)
def float32_to_int16(x):
    x = np.clip(x, -1., 1.); return (x * 32767.).astype(np.int16)

def _load_clap(clap_model: str, cache_dir: str):
    if clap_model == 'music_speech_audioset_epoch_15_esc_89.98.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt'
        ckpt = f'{cache_dir}/music_speech_audioset_epoch_15_esc_89.98.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device='cuda')
    elif clap_model == 'music_audioset_epoch_15_esc_90.14.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
        ckpt = f'{cache_dir}/music_audioset_epoch_15_esc_90.14.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device='cuda')
    elif clap_model == 'music_speech_epoch_15_esc_89.25.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt'
        ckpt = f'{cache_dir}/music_speech_epoch_15_esc_89.25.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device='cuda')
    elif clap_model == '630k-audioset-fusion-best.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'
        ckpt = f'{cache_dir}/630k-audioset-fusion-best.pt'
        model = laion_clap.CLAP_Module(enable_fusion=True, device='cuda')
    else:
        raise ValueError('clap_model not implemented')
    if not os.path.exists(ckpt):
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(ckpt, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc='Downloading CLAP') as bar:
            for chunk in resp.iter_content(8192): f.write(chunk); bar.update(len(chunk))
    pkg = load_state_dict(ckpt)
    pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg)
    model.eval()
    return model

def clap_score(
    texts: list[str],
    gen_files: list[str],
    clap_model: str = "630k-audioset-fusion-best.pt",
    cache_dir: str = "cache/clap",
) -> float:
    assert len(texts) == len(gen_files) and len(texts) > 0
    model = _load_clap(clap_model, cache_dir)

    # text emb
    text_emb = []
    bs = 64
    for s in range(0, len(texts), bs):
        batch = texts[s:s+bs]
        with torch.no_grad():
            try:
                emb = model.get_text_embedding(batch, use_tensor=True)
            except TypeError:
                emb = model.get_text_embedding(batch)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            emb = emb.detach().cpu()
        text_emb.append(emb)
    text_emb = torch.vstack(text_emb)

    # audio emb + cosine
    sims = []
    for i, wav in enumerate(tqdm(gen_files, desc='CLAP eval')):
        a, _ = librosa.load(wav, sr=48000, mono=True)
        a = pyln.normalize.peak(a, -1.0)
        a_np = int16_to_float32(float32_to_int16(a.astype(np.float32)))
        with torch.no_grad():
            try:
                aemb = model.get_audio_embedding_from_data([a_np])
            except TypeError:
                aemb = model.get_audio_embedding_from_data(x=[a_np])
        if isinstance(aemb, np.ndarray):
            aemb = torch.from_numpy(aemb)
        aemb = aemb.detach().cpu()
        if aemb.ndim == 2 and aemb.size(0) == 1:
            aemb = aemb.squeeze(0)
        sim = torch.nn.functional.cosine_similarity(aemb, text_emb[i].unsqueeze(0), dim=1, eps=1e-8)[0]
        sims.append(float(sim))
    return float(np.mean(sims)) if sims else 0.0
