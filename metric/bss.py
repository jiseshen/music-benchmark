from pathlib import Path
import numpy as np, librosa, torch
from tqdm import tqdm

try:  # optional SciPy for optimal permutation
    from scipy.optimize import linear_sum_assignment as hungarian
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as TM_SI_SDR
    _HAS_TM = True
except Exception:
    _HAS_TM = False


def _coerce_audio_array(data, mono: bool) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if mono:
            axis = 0 if arr.shape[0] <= arr.shape[1] else 1
            return arr.mean(axis=axis)
        return arr
    raise ValueError(f"Unsupported audio array ndim={arr.ndim}")


def _load_audio_like(src, *, target_sr: int | None, mono: bool, label: str):
    """Accept path-like or (array, sr) input and return (waveform, sr)."""
    try:
        if isinstance(src, (str, Path)):
            wav, sr = librosa.load(str(src), sr=target_sr, mono=mono)
            return wav, sr
        if isinstance(src, (tuple, list)) and len(src) == 2:
            wav_raw, sr = src
        elif isinstance(src, dict) and "sr" in src and ("audio" in src or "wav" in src):
            wav_raw = src.get("audio", src.get("wav"))
            sr = src["sr"]
        else:
            raise TypeError(f"Unsupported audio input type: {type(src)}")
        wav = _coerce_audio_array(wav_raw, mono=mono)
        sr = int(sr)
        if target_sr is not None and sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return wav, sr
    except Exception as exc:
        print(f"[BSS] Failed to load audio ({label}): {exc}")
        raise


def _si_sdr_pair(est, ref, eps=1e-8):
    """Scale-invariant SDR between two 1-D arrays."""
    ref = torch.tensor(ref).float()
    est = torch.tensor(est).float()
    ref = ref - ref.mean()
    est = est - est.mean()
    s_target = (torch.dot(est, ref) / (torch.dot(ref, ref) + eps)) * ref
    e_noise  = est - s_target
    ratio = (torch.sum(s_target**2) + eps) / (torch.sum(e_noise**2) + eps)
    return 10.0 * torch.log10(ratio).item()

def _si_sdr_input_pair(gen_input, ref_input, mono=True):
    rr, sr_ref = _load_audio_like(ref_input, target_sr=None, mono=mono, label="ref_single")
    ge, sr_est = _load_audio_like(gen_input, target_sr=sr_ref, mono=mono, label="est_single")
    if sr_est != sr_ref:
        rr, _ = _load_audio_like((rr, sr_ref), target_sr=sr_est, mono=mono, label="ref_resample")
    n = min(len(ge), len(rr))
    ge = ge[:n]
    rr = rr[:n]
    if _HAS_TM:
        metric = TM_SI_SDR()
        return metric(torch.tensor(ge)[None, :], torch.tensor(rr)[None, :]).item()
    return _si_sdr_pair(ge, rr)


# ---------------- Multi-source SI-BSS metrics (SI-SDR / SI-SIR / SI-SAR) ---------------- #
def _bss_decompose(est: np.ndarray, refs: np.ndarray, j: int, eps: float = 1e-8) -> tuple:
    """Decompose estimate for source j into target, interference, artifacts (scale-invariant).

    Args:
        est (np.ndarray): Estimated source signal (T,).
        refs (np.ndarray): Reference sources array (S, T).
        j (int): Index of the reference source for this estimate.
        eps (float): Numerical stability.
    Returns:
        (s_target, e_interf, e_artif) each (T,)
    """
    ref_j = refs[j]
    # Center signals (scale-invariant mean removal)
    est_z = est - est.mean()
    ref_j_z = ref_j - ref_j.mean()
    # Target projection
    alpha_j = np.dot(est_z, ref_j_z) / (np.dot(ref_j_z, ref_j_z) + eps)
    s_target = alpha_j * ref_j_z
    # Interference: projections onto other reference sources
    e_interf = np.zeros_like(est_z)
    for k in range(refs.shape[0]):
        if k == j:
            continue
        ref_k = refs[k] - refs[k].mean()
        alpha_k = np.dot(est_z, ref_k) / (np.dot(ref_k, ref_k) + eps)
        e_interf += alpha_k * ref_k
    # Artifacts
    e_artif = est_z - s_target - e_interf
    return s_target, e_interf, e_artif


def _si_metrics_from_components(s_target: np.ndarray, e_interf: np.ndarray, e_artif: np.ndarray, eps: float = 1e-8) -> tuple:
    """Return (si_sdr, si_sir, si_sar) for given components."""
    target_power = np.sum(s_target ** 2)
    interf_power = np.sum(e_interf ** 2)
    artif_power = np.sum(e_artif ** 2)
    resid_power = interf_power + artif_power
    si_sdr = 10 * np.log10((target_power + eps) / (resid_power + eps))
    si_sir = 10 * np.log10((target_power + eps) / (interf_power + eps)) if interf_power > 0 else np.inf
    si_sar = 10 * np.log10((target_power + interf_power + eps) / (artif_power + eps)) if artif_power > 0 else np.inf
    return si_sdr, si_sir, si_sar


def si_bss_track(est_files: list, ref_files: list, mono: bool = True, permute: bool = True, eps: float = 1e-8) -> dict:
    """(Legacy) multi-source SI-SDR/SI-SIR/SI-SAR. Prefer using bss_track."""
    assert len(est_files) == len(ref_files) and len(ref_files) > 0, "Mismatch in number of sources"
    S = len(ref_files)
    # Load audio
    refs = []
    ests = []
    sr_ref = None
    for fp in ref_files:
        x, sr = _load_audio_like(fp, target_sr=sr_ref, mono=mono, label="ref_multi")
        if sr_ref is None:
            sr_ref = sr
        refs.append(x)
    for fp in est_files:
        y, sr = _load_audio_like(fp, target_sr=sr_ref, mono=mono, label="est_multi")
        ests.append(y)
    # Pad/trim each pair to min length among all signals
    min_len = min(min(len(x) for x in refs), min(len(y) for y in ests))
    refs = np.stack([x[:min_len] for x in refs], axis=0)  # (S, T)
    ests = np.stack([y[:min_len] for y in ests], axis=0)  # (S, T)

    # Permutation alignment (maximize SI-SDR)
    if permute and S > 1:
        # Build matrix of si_sdr projections
        cost = np.zeros((S, S))
        for i in range(S):
            for j in range(S):
                s_target, e_interf, e_artif = _bss_decompose(ests[i], refs, j, eps=eps)
                si_sdr, *_ = _si_metrics_from_components(s_target, e_interf, e_artif, eps=eps)
                cost[i, j] = -si_sdr  # minimize cost = -si_sdr
        if _HAS_SCIPY:
            row_idx, col_idx = hungarian(cost)
        else:
            # Greedy fallback
            chosen = set()
            row_idx = []
            col_idx = []
            for i in range(S):
                j_best = None
                v_best = 1e9
                for j in range(S):
                    if j in chosen:
                        continue
                    if cost[i, j] < v_best:
                        v_best = cost[i, j]
                        j_best = j
                row_idx.append(i)
                col_idx.append(j_best)
                chosen.add(j_best)
            row_idx = np.array(row_idx)
            col_idx = np.array(col_idx)
        # Reorder ests to match refs order
        ests = ests[row_idx]
        order = col_idx.tolist()
        # After permutation, reorder indices so that est[k] aligned to ref[k]
        inv = np.zeros(S, dtype=int)
        for est_pos, ref_pos in enumerate(order):
            inv[ref_pos] = est_pos
        ests = ests[inv]
        order = inv.tolist()
    else:
        order = list(range(S))

    si_sdr_list = []
    si_sir_list = []
    si_sar_list = []
    for j in range(S):
        s_target, e_interf, e_artif = _bss_decompose(ests[j], refs, j, eps=eps)
        si_sdr, si_sir, si_sar = _si_metrics_from_components(s_target, e_interf, e_artif, eps=eps)
        si_sdr_list.append(float(si_sdr))
        si_sir_list.append(float(si_sir))
        si_sar_list.append(float(si_sar))

    return {
        'order': order,  # mapping index in ref -> index in est after permutation
        'si_sdr': si_sdr_list,
        'si_sir': si_sir_list,
        'si_sar': si_sar_list,
        'avg_si_sdr': float(np.mean(si_sdr_list)),
        'avg_si_sir': float(np.mean([v for v in si_sir_list if np.isfinite(v)])) if si_sir_list else 0.0,
        'avg_si_sar': float(np.mean([v for v in si_sar_list if np.isfinite(v)])) if si_sar_list else 0.0,
    }


def si_bss_batch(gen_groups: list, ref_groups: list, mono: bool = True, permute: bool = True) -> dict:
    """(Legacy) batch multi-source metrics. Prefer using bss_batch."""
    assert len(gen_groups) == len(ref_groups) and len(gen_groups) > 0
    details = []
    all_sdr = []
    all_sir = []
    all_sar = []
    for est_list, ref_list in tqdm(list(zip(gen_groups, ref_groups)), desc='SI-BSS'):
        res = si_bss_track(est_list, ref_list, mono=mono, permute=permute)
        details.append(res)
        all_sdr.extend(res['si_sdr'])
        all_sir.extend([v for v in res['si_sir'] if np.isfinite(v)])
        all_sar.extend([v for v in res['si_sar'] if np.isfinite(v)])
    return {
        'details': details,
        'global_avg_si_sdr': float(np.mean(all_sdr)) if all_sdr else 0.0,
        'global_avg_si_sir': float(np.mean(all_sir)) if all_sir else 0.0,
        'global_avg_si_sar': float(np.mean(all_sar)) if all_sar else 0.0,
    }


__all__ = ['bss_track', 'bss_batch', 'si_bss_track', 'si_bss_batch']


# ---------------- Unified interface (single-source editing + multi-source separation) ---------------- #
def bss_track(est_files, ref_files, mono: bool = True, permute: bool = True, eps: float = 1e-8):
    """Unified interface for SI metrics on one item (single or multi-source).

    单条数据统一接口：
    - 单源 (len==1): 计算 scale-invariant SI-SNR (同 SI-SDR)，返回 dict: { 'mode': 'single', 'si_snr': value }
      这里沿用术语 si_snr == si_sdr (scale-invariant) 方便和训练脚本衔接。
    - 多源 (len>1): 使用已有 si_bss_track 计算，并返回 { 'mode': 'multi', 'order', 'per_source': [ {'si_snr': .., 'si_sar': ..} ...], 'avg_si_snr', 'avg_si_sar' }
      注意：这里不返回 si_sir，精简接口聚焦：整体质量 (SI-SNR) + 伪影/artifact 抑制 (SI-SAR)。

    Args:
        est_files (list[str]): Estimated wav paths.
        ref_files (list[str]): Reference wav paths.
        mono (bool): Downmix to mono.
        permute (bool): For multi-source, whether to search best permutation by SI-SDR.
        eps (float): Numerical stability.
    Returns:
        dict: See schema above.

    Examples / 用法:
        # 单源编辑 (inpainting / enhancement)
        res = bss_track(['gen.wav'], ['ref.wav'])
        print(res['si_snr'])

        # 多源分离
        res = bss_track(['vocals_est.wav','accomp_est.wav'], ['vocals_ref.wav','accomp_ref.wav'])
        print(res['avg_si_snr'], res['avg_si_sar'])
    """
    assert len(est_files) == len(ref_files) and len(ref_files) > 0, "Number mismatch"
    if len(ref_files) == 1:
        v = _si_sdr_input_pair(est_files[0], ref_files[0], mono=mono)
        return {
            'mode': 'single',
            'si_snr': v,
        }
    # Multi-source path
    raw = si_bss_track(est_files, ref_files, mono=mono, permute=permute, eps=eps)
    per_source = []
    for i in range(len(raw['si_sdr'])):
        per_source.append({
            'si_snr': raw['si_sdr'][i],
            'si_sar': raw['si_sar'][i],
        })
    return {
        'mode': 'multi',
        'order': raw['order'],  # mapping ref index -> est index
        'per_source': per_source,
        'avg_si_snr': raw['avg_si_sdr'],
        'avg_si_sar': raw['avg_si_sar'],
    }


def bss_batch(gen_groups, ref_groups, mono: bool = True, permute: bool = True):
    """Unified batch interface.

    Accepts a list of groups; each group is list of estimated sources for one mixture/item, with matching reference group.
    Automatically distinguishes between:
        - Single-source: aggregates si_snr
        - Multi-source: aggregates si_snr and si_sar
    Returns:
        {
            'details': [ per-item dict same as bss_track return structure ],
            'per_sample_si_snr': [ float, ... ],  # per-sample SI-SNR values
            'global_avg_si_snr': float,
            'global_avg_si_sar': float or None (if all are single-source then None)
        }

    Args:
        gen_groups (list[list[str]]): Estimated wav paths grouped by item.
        ref_groups (list[list[str]]): Reference wav paths grouped by item.
        mono (bool): Downmix to mono.
        permute (bool): Multi-source permutation alignment.
    """
    assert len(gen_groups) == len(ref_groups) and len(gen_groups) > 0
    details = []
    all_snr = []
    all_sar = []
    per_sample_snr = []  # Track per-sample SI-SNR
    
    for est_list, ref_list in zip(gen_groups, ref_groups):
        item = bss_track(est_list, ref_list, mono=mono, permute=permute)
        details.append(item)
        if item['mode'] == 'single':
            snr_value = item['si_snr']
            all_snr.append(snr_value)
            per_sample_snr.append(snr_value)
        else:
            snr_value = item['avg_si_snr']
            all_snr.append(snr_value)
            per_sample_snr.append(snr_value)
            if np.isfinite(item['avg_si_sar']):
                all_sar.append(item['avg_si_sar'])
    
    return {
        'details': details,
        'per_sample_si_snr': per_sample_snr,  # Added per-sample results
        'global_avg_si_snr': float(np.mean(all_snr)) if all_snr else 0.0,
        'global_avg_si_sar': float(np.mean(all_sar)) if all_sar else None,
    }
