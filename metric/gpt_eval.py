import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from openai import OpenAI


def _build_system_prompt() -> str:
    parts = [
        "You are an assistant for evaluating music annotations.",
        "You will be given a reference set of labels and a generated set of labels.",
        "1. Under 'ref_count', output the number of labels in the reference set.",
        "2. Under 'gen_count', output the number of labels in the generated set.",
        (
            "3. Under 'missed', list reference labels not present in the generated set. "
            "Treat closely related or sonically similar concepts (e.g., 'electronic' ≈ 'edm', "
            "'new age' ≈ 'ambient', 'rock' ≈ 'metal', 'violin' ≈ 'string') as matches, not misses."
        ),
        (
            "4. Under 'incorrect', list generated labels not present in the reference set. "
            "Again, treat closely related concepts as acceptable, not incorrect."
        ),
        (
            "5. Under 'score', give an integer from 1 to 10, where 1 = no relevance and 10 = perfect match. "
            "Use high but not perfect scores when related (but not exact) labels are generated."
        ),
        (
            "Respond strictly in valid JSON: "
            '{"ref_count": <int>, "gen_count": <int>, "missed": [...], "incorrect": [...], "score": <int 1–10>}'
            "Do not include any extra text."
        ),
    ]
    return "\n".join(parts)


def _parse_labels(text: str) -> List[str]:
    # Heuristic split: commas, semicolons, newlines; trim and drop empties
    items = re.split(r"[\n,;]+", text)
    labels = [s.strip() for s in items if s.strip()]
    return labels


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
    reasoning_effort: str = "minimal",
    max_output_tokens: int = 256,
) -> Dict[str, Any]:
    system_prompt = _build_system_prompt()
    ref_labels = _parse_labels(ref_text)
    gen_labels = _parse_labels(gen_text)

    # Build a single string input for compatibility across SDK versions
    user_payload = json.dumps(
        {
            "reference": {"raw_text": ref_text, "labels": ref_labels},
            "generated": {"raw_text": gen_text, "labels": gen_labels},
        },
        ensure_ascii=False,
    )
    input_text = f"System:\n{system_prompt}\n\nUser:\n{user_payload}"

    resp = client.responses.create(
        model=model,
        input=input_text,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
    )

    # Best-effort to get text content across SDK versions
    text: Optional[str] = None
    if hasattr(resp, "output_text") and resp.output_text:
        text = resp.output_text
    elif hasattr(resp, "output") and resp.output:
        try:
            text = resp.output[0].content[0].text  # type: ignore[attr-defined]
        except Exception:
            pass
    if not text:
        text = str(resp)

    data = _extract_json(text)
    # Sanity defaults
    data.setdefault("ref_count", len(ref_labels))
    data.setdefault("gen_count", len(gen_labels))
    data.setdefault("missed", [])
    data.setdefault("incorrect", [])
    data.setdefault("score", 0)
    # Normalize types
    if not isinstance(data.get("missed"), list):
        data["missed"] = []
    if not isinstance(data.get("incorrect"), list):
        data["incorrect"] = []
    # Clamp score to [1,10]
    try:
        s = int(data.get("score", 0))
        data["score"] = max(1, min(10, s))
    except Exception:
        raise ValueError(f"Invalid score value from model: {data.get('score')}")
    return data


def gpt_eval(
    ref_texts: List[str],
    gen_texts: List[str],
    model: str = "gpt-5-mini",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare each (ref_text, gen_text) pair via GPT and aggregate metrics.

    Returns dict with per-pair results and overall averages:
      {
        'pairs': [ {'ref_count': int, 'gen_count': int, 'missed': [...], 'incorrect': [...], 'score': int}, ... ],
        'score': float,           # average of pairwise 'score'
        'precision': float,       # (Σ(gen_count) - Σ(incorrect)) / Σ(gen_count)
        'recall': float,          # (Σ(ref_count) - Σ(missed)) / Σ(ref_count)
        'f1': float
      }
    """
    assert len(ref_texts) == len(gen_texts), "ref_texts and gen_texts must have same length"
    if len(ref_texts) == 0:
        return {"pairs": [], "score": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Init client
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please export it or pass api_key.")
    client = OpenAI(api_key=key)

    pair_results: List[Dict[str, Any]] = []
    failed_pairs = 0
    total_ref = 0
    total_gen = 0
    total_missed = 0
    total_incorrect = 0
    score_sum = 0.0

    import time
    for ref, gen in tqdm(zip(ref_texts, gen_texts), total=len(ref_texts), desc="GPT Eval"):
        # simple retry for transient API errors
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                res = _call_gpt_pair(client, ref, gen, model=model)
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        else:
            # all retries failed
            print(f"[WARN] GPT evaluation failed for a pair after 3 attempts: {last_err}, skipping...")
            failed_pairs += 1
            continue
        pair_results.append(res)
        # Aggregate
        rc = int(res.get("ref_count", 0))
        gc = int(res.get("gen_count", 0))
        missed = res.get("missed", [])
        incorrect = res.get("incorrect", [])
        sc = float(res.get("score", 0))
        total_ref += max(0, rc)
        total_gen += max(0, gc)
        total_missed += len(missed) if isinstance(missed, list) else 0
        total_incorrect += len(incorrect) if isinstance(incorrect, list) else 0
        score_sum += sc

    # Precision/Recall over ALL labels across all pairs
    precision = 0.0
    recall = 0.0
    if total_gen > 0:
        precision = max(0.0, min(1.0, (total_gen - total_incorrect) / total_gen))
    if total_ref > 0:
        recall = max(0.0, min(1.0, (total_ref - total_missed) / total_ref))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    avg_score = score_sum / len(pair_results)
    if failed_pairs > 0:
        print(f"[WARN] {failed_pairs} / {len(ref_texts)} pairs failed GPT evaluation and were skipped.")
    return {
        "pairs": pair_results,
        "score": float(avg_score),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "totals": {
            "ref_count": total_ref,
            "gen_count": total_gen,
            "missed": total_missed,
            "incorrect": total_incorrect,
        },
    }


if __name__ == "__main__":
    # Tiny smoke example
    demo_ref = [
        "electronic, ambient, pads, violin",
        "rock, metal, guitar solo",
    ]
    demo_gen = [
        "edm, new age, strings, pad",
        "rock, heavy metal, guitar lead",
    ]
    out = gpt_eval(demo_ref, demo_gen, model=os.getenv("GPT_MODEL", "gpt-5-mini"))
    print(json.dumps(out, ensure_ascii=False, indent=2))
