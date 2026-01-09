import json
import re
from typing import Any, Dict, Optional, Tuple
import time

ALLOWED = {0, 1, 2, 3, 4}
REQUIRED_KEYS = ["accuracy", "faithfulness", "relevance", "completeness"]

def validate_judge_output(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Output is not a JSON object."
    if set(obj.keys()) != set(REQUIRED_KEYS):
        return False, f"Keys must be exactly {REQUIRED_KEYS}."
    for k in REQUIRED_KEYS:
        v = obj.get(k)
        if not isinstance(v, int):
            return False, f"'{k}' must be an integer."
        if v not in ALLOWED:
            return False, f"'{k}' must be one of {sorted(ALLOWED)}."
    return True, "OK"

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to extract the first JSON object from arbitrary text.
    Use only as a last resort.
    """
    # naive but effective for many cases:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def guard_parse(text: str) -> Dict[str, int]:
    """
    Strict parse: must be valid JSON object with required keys and allowed int values.
    """
    obj = json.loads(text)
    ok, msg = validate_judge_output(obj)
    if not ok:
        raise ValueError(f"Invalid judge JSON: {msg}")
    return obj  # type: ignore

def judge_with_output_guard(
    call_llm_func,
    prompt: str,
    max_retries: int = 2,
    backoff_base_sec: float = 0.8,
) -> Dict[str, int]:
    """
    Calls LLM and strictly enforces JSON schema via retries.
    """
    last_text = ""
    correction = (
        "Önceki çıktın geçersizdi. "
        "SADECE geçerli bir JSON obje döndür. "
        "Anahtarlar TAM OLARAK: accuracy, faithfulness, relevance, completeness. "
        "Değerler integer ve sadece {0,1,2,3,4}. "
        "Başka hiçbir metin yazma."
    )

    current_prompt = prompt

    for attempt in range(max_retries + 1):
        last_text = call_llm_func(current_prompt)

        # 1) strict parse
        try:
            return guard_parse(last_text)
        except Exception:
            pass

        # 2) extract + validate (last resort)
        extracted = extract_json_object(last_text)
        if extracted is not None:
            ok, _ = validate_judge_output(extracted)
            if ok:
                return extracted  # type: ignore

        # 3) retry: append correction to prompt (simple, provider-agnostic)
        sleep_s = backoff_base_sec * (2 ** attempt)
        time.sleep(sleep_s)
        current_prompt = f"{prompt}\n\n{correction}"

    raise RuntimeError(f"LLM failed to return valid judge JSON. Last output:\n{last_text}")
