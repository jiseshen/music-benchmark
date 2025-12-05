import torch.nn.functional as F
import torch, pyloudnorm as pyln, librosa, numpy as np, contextlib, os
from tqdm import tqdm
from functools import partial
from hear21passt.base import get_basic_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SAMPLING_RATE = 32000

class _patch_passt_stft:
    def __init__(self): self.old_stft = torch.stft
    def __enter__(self): torch.stft = partial(torch.stft, return_complex=False)
    def __exit__(self, *exc): torch.stft = self.old_stft

def _return_probabilities(model, audio_path, window_size=10, overlap=5, collect='mean'):
    a, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    a = pyln.normalize.peak(a, -1.0)
    step = int((window_size - overlap) * SAMPLING_RATE)
    probs = []
    for i in range(0, max(step, len(a) - step), step):
        win = a[i:i + int(window_size * SAMPLING_RATE)]
        if len(win) < int(window_size * SAMPLING_RATE):
            if len(win) > int(window_size * SAMPLING_RATE * 0.15):
                tmp = np.zeros(int(window_size * SAMPLING_RATE)); tmp[:len(win)] = win; win = tmp
        x = torch.from_numpy(win.astype(np.float32)).unsqueeze(0).cuda()
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            with torch.no_grad(), _patch_passt_stft():
                logits = model(x); probs.append(torch.squeeze(logits))
    probs = torch.stack(probs)
    if collect == 'mean': probs = probs.mean(dim=0)
    elif collect == 'max': probs, _ = probs.max(dim=0)
    return F.softmax(probs, dim=0).squeeze().cpu()

def passt_kld(
    gen_files, ref_files, collect='mean'
) -> float:
    assert len(gen_files) == len(ref_files) and len(gen_files) > 0
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        model = get_basic_model(mode="logits"); model.eval(); model = model.cuda()
    kls = []
    for g, r in tqdm(list(zip(gen_files, ref_files)), desc='PaSST KLD'):
        ref_p  = _return_probabilities(model, r, collect=collect)
        eval_p = _return_probabilities(model, g, collect=collect)
        kls.append(float(F.kl_div((ref_p + 1e-6).log(), eval_p, reduction='sum', log_target=False)))
    return float(np.mean(kls)) if kls else 0.0
