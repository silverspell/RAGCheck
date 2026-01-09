import argparse
import weakref
from prompt import build_judge_prompt
from guards import judge_with_output_guard
import pandas as pd
import os
from tqdm import tqdm
from openai import OpenAI
from analyze import analyze_rag_judge_results, plot_metric_correlation, drill_down_low_scores, score_to_action_recommendations

from dotenv import load_dotenv

ALLOWED = {0, 1, 2, 3, 4}
REQUIRED_KEYS = ["accuracy", "faithfulness", "relevance", "completeness"]

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <input>_scored.csv)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of rows for a quick run")
    parser.add_argument("--resume", action="store_true", help="Skip rows that already have scores in output file")
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output or os.path.splitext(in_path)[0] + "_scored.csv"

    df = pd.read_csv(in_path)

    # Required columns check
    required_cols = ["question", "answer", "context", "ground_truth"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    # If resume, load existing output and merge scores where present
    if args.resume and os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        # Try to align by index; simplest approach: same row order assumption
        for col in REQUIRED_KEYS:
            if col in existing.columns and col not in df.columns:
                df[col] = existing[col].iloc[: len(df)].values
            elif col in existing.columns and col in df.columns:
                # keep existing values where df has NaN
                df[col] = df[col].fillna(existing[col].iloc[: len(df)].values)

    # Ensure score columns exist
    for col in REQUIRED_KEYS:
        if col not in df.columns:
            df[col] = pd.NA

    # LLM call adapter
    def call_llm(prompt: str) -> str:
        return call_llm_openai(prompt=prompt, model=args.model)

    # Batch scoring
    for i in tqdm(range(len(df)), desc="Scoring rows"):
        # Skip already scored rows (if resume)
        if all(pd.notna(df.at[i, col]) for col in REQUIRED_KEYS):
            continue

        q = str(df.at[i, "question"])
        a = str(df.at[i, "answer"])
        c = str(df.at[i, "context"])
        gt = str(df.at[i, "ground_truth"])

        prompt = build_judge_prompt(q, a, c, gt)

        try:
            scores = judge_with_output_guard(call_llm, prompt, 2)
            for col in REQUIRED_KEYS:
                df.at[i, col] = int(scores[col])
        except Exception as e:
            # Fail-safe: keep row, mark as NA and continue
            # (You can also choose to raise if you prefer strictness.)
            print(f"[WARN] Row {i} failed: {e}")
            for col in REQUIRED_KEYS:
                df.at[i, col] = pd.NA

        # Save progressively to avoid losing progress on long runs
        if (i + 1) % 20 == 0:
            df.to_csv(out_path, index=False)

    # Final save
    df.to_csv(out_path, index=False)
    print(f"Done. Wrote: {out_path}")
    result = analyze_rag_judge_results(
        csv_path=out_path,
        low_score_threshold=2,
        show=False,
    )
    weakdf = result["weak_rows"]
    weakdf.head()

    corr = plot_metric_correlation("test_scored.csv")
    print(corr)

    low_rows = drill_down_low_scores(
        csv_path="test_scored.csv",
        threshold=2
    )
    low_rows.head()

    actions = score_to_action_recommendations("test_scored.csv")

    for a in actions["recommendations"]:
        print("\n❗", a["problem"])
        print("→", a["interpretation"])
        for act in a["actions"]:
            print("  -", act)



def call_llm_openai(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 200) -> str:
    """
    Minimal OpenAI call. Requires:
      pip install openai
      export OPENAI_API_KEY=...
    """
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a strict JSON generator. Output only valid JSON."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ],
        max_tokens=max_tokens,
    )

    return (resp.choices[0].message.content or "").strip()



if __name__ == "__main__":
    main()