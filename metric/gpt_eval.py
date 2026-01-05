import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from openai import OpenAI


def _build_system_prompt() -> str:
    return """You are an assistant for evaluating music metadata by comparing reference vs generated annotations.

For each of the three categories (genre, mood, instruments), give:
1. A score (1-10) where 1=no relevance, 10=perfect match
2. List of missed labels (reference has but generated doesn't; treat related concepts like 'rock'≈'metal' as matches)
3. List of incorrect labels (generated has but reference doesn't; treat related concepts as acceptable)

Treat closely related musical concepts as matches (e.g., 'electronic'≈'edm', 'violin'≈'strings', 'upbeat'≈'energetic').

Respond ONLY in valid JSON:
{
  "genre": {"score": <1-10>, "missed": [...], "incorrect": [...]},
  "mood": {"score": <1-10>, "missed": [...], "incorrect": [...]},
  "instruments": {"score": <1-10>, "missed": [...], "incorrect": [...]}
}"""


def _parse_labels(text: str) -> Dict[str, List[str]]:
    """
    Parse labels from text (JSON or plain text format) into standardized categories.
    Always returns dict with keys: "genre", "mood", "instruments"
    
    Handles:
    - JSON: {"genre": [...], "mood": [...], "instruments": [...]}
    - Key-value: "Genre: rock, pop\nMood: energetic\nInstruments: guitar, drums"
    - Mixed formats
    """
    text = text.strip()
    parsed: Dict[str, List[str]] = {}
    
    # Try JSON first
    if text.startswith('{') or text.startswith('```'):
        # Strip markdown fences
        clean_text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
        if clean_text.startswith('{'):
            try:
                data = json.loads(clean_text)
                for key, value in data.items():
                    key_lower = str(key).lower()
                    labels = []
                    if isinstance(value, list):
                        labels = [str(v).strip() for v in value if v]
                    else:
                        labels = [str(value).strip()] if value else []
                    
                    # Map to standard categories
                    if key_lower in ['genre', 'genres']:
                        parsed.setdefault('genre', []).extend(labels)
                    elif key_lower in ['mood', 'moods', 'emotion', 'emotions']:
                        parsed.setdefault('mood', []).extend(labels)
                    elif key_lower in ['instrument', 'instruments', 'instrument']:
                        parsed.setdefault('instruments', []).extend(labels)
                
                if parsed:
                    return {k: list(set(parsed.get(k, []))) for k in ['genre', 'mood', 'instruments']}
            except json.JSONDecodeError:
                pass
    
    # Try key-value pairs (e.g., "Genre: rock, pop\nInstrument: guitar")
    lines = text.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key_lower = key.strip().lower()
            labels = [s.strip() for s in re.split(r'[,;/&]', value) if s.strip()]
            
            # Map to standard categories
            if key_lower in ['genre', 'genres']:
                parsed.setdefault('genre', []).extend(labels)
            elif key_lower in ['mood', 'moods', 'emotion', 'emotions']:
                parsed.setdefault('mood', []).extend(labels)
            elif key_lower in ['instrument', 'instruments']:
                parsed.setdefault('instruments', []).extend(labels)
    
    if parsed:
        return {k: list(set(parsed.get(k, []))) for k in ['genre', 'mood', 'instruments']}
    
    # Fallback: treat as comma/newline-separated labels (guess which category based on content)
    all_labels = [s.strip() for s in re.split(r'[,;\n]', text) if s.strip()]
    if all_labels:
        # Heuristic: check for common keywords to guess category
        lower_text = text.lower()
        guess = 'genre'
        if any(w in lower_text for w in ['mood', 'energy', 'feel', 'vibe']):
            guess = 'mood'
        elif any(w in lower_text for w in ['instrument', 'guitar', 'drum', 'piano']):
            guess = 'instruments'
        return {k: (all_labels if k == guess else []) for k in ['genre', 'mood', 'instruments']}
    
    # Empty result
    return {'genre': [], 'mood': [], 'instruments': []}


def _extract_json(s: str) -> Dict[str, Any]:
    """Try to parse JSON from the model's output robustly (strip code fences, pick inner {...})."""
    def try_load(x: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(x)
        except Exception:
            return None

    s = s.strip()
    # Remove ```json ... ``` fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", s).strip()
    # Try direct
    data = try_load(s)
    if data is not None:
        return data
    # Try to find the first {...} block
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        data = try_load(m.group(0))
        if data is not None:
            return data
    raise ValueError(f"Could not parse JSON from model output: {s[:200]}...")


def _call_gpt_pair(
    client: OpenAI,
    ref_text: str,
    gen_text: str,
    model: str = "gpt-5-mini",
    max_output_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Compare reference and generated texts via GPT.
    Returns scores for each category (genre, mood, instruments).
    """
    system_prompt = _build_system_prompt()
    
    # Parse inputs to standardized categories
    ref_labels = _parse_labels(ref_text)
    gen_labels = _parse_labels(gen_text)
    
    user_message = f"""Reference labels:
genre: {', '.join(ref_labels['genre']) or '(none)'}
mood: {', '.join(ref_labels['mood']) or '(none)'}
instruments: {', '.join(ref_labels['instruments']) or '(none)'}

Generated labels:
genre: {', '.join(gen_labels['genre']) or '(none)'}
mood: {', '.join(gen_labels['mood']) or '(none)'}
instruments: {', '.join(gen_labels['instruments']) or '(none)'}

Evaluate each category separately and return JSON."""

    # Call OpenAI
    resp = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    # Extract response text
    text: Optional[str] = None
    if hasattr(resp, 'content') and resp.content:
        text = resp.content[0].text if hasattr(resp.content[0], 'text') else str(resp.content[0])
    else:
        text = str(resp)

    # Parse JSON from response
    data = _extract_json(text)
    
    # Normalize structure: ensure all three categories exist
    normalized: Dict[str, Any] = {}
    for cat in ['genre', 'mood', 'instruments']:
        cat_data = data.get(cat, {})
        if not isinstance(cat_data, dict):
            cat_data = {}
        
        # Ensure fields exist
        missed = cat_data.get('missed', [])
        incorrect = cat_data.get('incorrect', [])
        score = cat_data.get('score', 5)
        
        # Normalize types
        if not isinstance(missed, list):
            missed = []
        if not isinstance(incorrect, list):
            incorrect = []
        
        # Clamp score to [1, 10]
        try:
            score = int(score)
            score = max(1, min(10, score))
        except (ValueError, TypeError):
            score = 5
        
        normalized[cat] = {
            'ref_labels': ref_labels[cat],
            'gen_labels': gen_labels[cat],
            'missed': missed,
            'incorrect': incorrect,
            'score': score,
        }
    
    return normalized


def gpt_eval(
    ref_texts: List[str],
    gen_texts: List[str],
    model: str = "gpt-5-mini",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare each (ref_text, gen_text) pair via GPT and aggregate metrics.
    
    Returns dict with:
    - 'per_category_metrics': dict of metrics for each category (recall, precision, f1, gpt_score)
    - 'gpt_score': average GPT score (1-10)
    - 'f1', 'precision', 'recall': global metrics
    - 'pairs': detailed per-pair results
    """
    assert len(ref_texts) == len(gen_texts), "ref_texts and gen_texts must have same length"
    if len(ref_texts) == 0:
        return {
            "pairs": [],
            "per_category_metrics": {},
            "gpt_score": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    # Init client
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please export it or pass api_key.")
    client = OpenAI(api_key=key)

    pair_results: List[Dict[str, Any]] = []
    failed_pairs = 0
    
    # Per-category aggregation
    category_stats: Dict[str, Dict[str, int | float]] = {
        'genre': {'ref_count': 0, 'gen_count': 0, 'missed_count': 0, 'incorrect_count': 0, 'score_sum': 0.0, 'pair_count': 0},
        'mood': {'ref_count': 0, 'gen_count': 0, 'missed_count': 0, 'incorrect_count': 0, 'score_sum': 0.0, 'pair_count': 0},
        'instruments': {'ref_count': 0, 'gen_count': 0, 'missed_count': 0, 'incorrect_count': 0, 'score_sum': 0.0, 'pair_count': 0},
    }
    
    gpt_score_sum = 0.0

    import time
    for ref, gen in tqdm(zip(ref_texts, gen_texts), total=len(ref_texts), desc="GPT Eval"):
        # Retry logic
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                res = _call_gpt_pair(client, ref, gen, model=model)
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        else:
            print(f"[WARN] GPT evaluation failed: {last_err}, skipping...")
            failed_pairs += 1
            continue
        
        pair_results.append(res)
        
        # Aggregate per category
        cat_scores = []
        for cat_name in ['genre', 'mood', 'instruments']:
            cat_data = res[cat_name]
            ref_labels = cat_data['ref_labels']
            gen_labels = cat_data['gen_labels']
            missed = cat_data['missed']
            incorrect = cat_data['incorrect']
            score = cat_data['score']
            
            stats = category_stats[cat_name]
            stats['ref_count'] += len(ref_labels)
            stats['gen_count'] += len(gen_labels)
            stats['missed_count'] += len(missed) if isinstance(missed, list) else 0
            stats['incorrect_count'] += len(incorrect) if isinstance(incorrect, list) else 0
            stats['score_sum'] += score
            stats['pair_count'] += 1
            cat_scores.append(score)
        
        # Average score for this pair
        gpt_score_sum += sum(cat_scores) / len(cat_scores) if cat_scores else 5.0

    # Compute per-category metrics
    per_category_metrics: Dict[str, Dict[str, float]] = {}
    total_ref, total_gen, total_missed, total_incorrect = 0, 0, 0, 0
    
    for cat_name in ['genre', 'mood', 'instruments']:
        stats = category_stats[cat_name]
        ref_c = int(stats['ref_count'])
        gen_c = int(stats['gen_count'])
        missed_c = int(stats['missed_count'])
        incorrect_c = int(stats['incorrect_count'])
        pair_c = int(stats['pair_count'])
        score_avg = float(stats['score_sum']) / pair_c if pair_c > 0 else 5.0
        
        # Metrics
        recall = (ref_c - missed_c) / ref_c if ref_c > 0 else 0.0
        precision = (gen_c - incorrect_c) / gen_c if gen_c > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_category_metrics[cat_name] = {
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "gpt_score": float(score_avg),
        }
        
        total_ref += ref_c
        total_gen += gen_c
        total_missed += missed_c
        total_incorrect += incorrect_c

    # Global metrics
    global_recall = (total_ref - total_missed) / total_ref if total_ref > 0 else 0.0
    global_precision = (total_gen - total_incorrect) / total_gen if total_gen > 0 else 0.0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
    avg_gpt_score = gpt_score_sum / len(pair_results) if pair_results else 5.0

    if failed_pairs > 0:
        print(f"[WARN] {failed_pairs} / {len(ref_texts)} pairs failed GPT evaluation.")
    
    return {
        "pairs": pair_results,
        "per_category_metrics": per_category_metrics,
        "gpt_score": float(avg_gpt_score),
        "score": float(avg_gpt_score),  # Alias for compatibility
        "f1": float(global_f1),
        "precision": float(global_precision),
        "recall": float(global_recall),
    }


if __name__ == "__main__":
    # Demo with mixed input formats
    demo_ref = [
        "Genre: rock, pop\nMood: energetic\nInstruments: guitar, drums",
        """{"genre": ["electronic"], "mood": ["upbeat"], "instruments": ["synthesizer"]}""",
    ]
    demo_gen = [
        "Genre: rock\nMood: high-energy\nInstruments: guitar, keyboard",
        """{"genre": ["edm"], "mood": ["energetic"], "instruments": ["synth"]}""",
    ]
    
    print("[TEST] Running GPT evaluation with mixed formats...")
    out = gpt_eval(demo_ref, demo_gen, model=os.getenv("GPT_MODEL", "gpt-5-mini"))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    
    print("\n[TEST] Per-category metrics:")
    for cat, metrics in out.get("per_category_metrics", {}).items():
        print(f"  {cat}: F1={metrics.get('f1', 0):.3f}, GPT-Score={metrics.get('gpt_score', 0):.1f}")


