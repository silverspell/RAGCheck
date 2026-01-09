import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METRICS = ["accuracy", "faithfulness", "relevance", "completeness"]

def analyze_rag_judge_results(
    csv_path: str,
    low_score_threshold: int = 2,
    show: bool = True,
):
    """
    Visual & numeric analysis for LLM-as-a-judge RAG scores.

    Args:
        csv_path: path to *_scored.csv
        low_score_threshold: rows with any metric <= this are flagged
        show: whether to show plots
    """

    df = pd.read_csv(csv_path)

    # --- basic sanity ---
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")

    print("\n===== BASIC STATS =====")
    stats = df[METRICS].describe()
    print(stats)

    # --- mean scores ---
    means = df[METRICS].mean().sort_values(ascending=False)

    if show:
        plt.figure()
        means.plot(kind="bar")
        plt.title("Average RAG Judge Scores")
        plt.ylabel("Score (0–4)")
        plt.ylim(0, 4)
        plt.grid(axis="y")
        plt.show()

    # --- distribution per metric ---
    for metric in METRICS:
        if show:
            plt.figure()
            df[metric].value_counts().sort_index().plot(kind="bar")
            plt.title(f"{metric.capitalize()} Score Distribution")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.xticks(rotation=0)
            plt.grid(axis="y")
            plt.show()

    # --- weak rows ---
    weak_mask = (df[METRICS] <= low_score_threshold).any(axis=1)
    weak_rows = df[weak_mask]

    print("\n===== WEAK SAMPLE SUMMARY =====")
    print(f"Rows with any metric <= {low_score_threshold}: {len(weak_rows)} / {len(df)}")

    # Per-metric weakness count
    weakness_counts = {
        m: (df[m] <= low_score_threshold).sum()
        for m in METRICS
    }
    print("\nWeakness count by metric:")
    for k, v in weakness_counts.items():
        print(f"- {k}: {v}")

    return {
        "stats": stats,
        "means": means,
        "weak_rows": weak_rows,
        "weakness_counts": weakness_counts,
    }


# -------------------------------------------------
# 1️⃣ Metric Correlation Heatmap
# -------------------------------------------------

def plot_metric_correlation(csv_path: str):
    df = pd.read_csv(csv_path)

    corr = df[METRICS].corr()

    plt.figure()
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )
    plt.title("RAG Judge Metric Correlation Heatmap")
    plt.show()

    return corr


# -------------------------------------------------
# 2️⃣ Row-level Drill-down Analysis
# -------------------------------------------------

def drill_down_low_scores(
    csv_path: str,
    threshold: int = 2
):
    """
    Returns rows with low scores + human-readable diagnosis.
    """
    df = pd.read_csv(csv_path)

    def diagnose(row):
        issues = []

        if row["faithfulness"] <= threshold:
            issues.append("Context dışı bilgi / hallucination riski")

        if row["accuracy"] <= threshold:
            issues.append("Ground truth ile uyumsuzluk")

        if row["relevance"] <= threshold:
            issues.append("Soruya odaklanmıyor / konu dışı")

        if row["completeness"] <= threshold:
            issues.append("Eksik cevap / partial answer")

        return " | ".join(issues)

    mask = (df[METRICS] <= threshold).any(axis=1)
    weak_df = df[mask].copy()

    weak_df["diagnosis"] = weak_df.apply(diagnose, axis=1)

    cols = [
        "question",
        "answer",
        "accuracy",
        "faithfulness",
        "relevance",
        "completeness",
        "diagnosis",
    ]

    return weak_df[cols].sort_values(METRICS)


# -------------------------------------------------
# 4️⃣ Score → Action Mapping (Auto Recommendation)
# -------------------------------------------------

def score_to_action_recommendations(csv_path: str):
    df = pd.read_csv(csv_path)

    actions = []

    means = df[METRICS].mean()

    if means["faithfulness"] < 3:
        actions.append({
            "problem": "Low Faithfulness",
            "interpretation": "Model context dışına taşıyor (hallucination)",
            "actions": [
                "Chunk size / overlap yeniden ayarla",
                "Top-k düşür veya dynamic top-k kullan",
                "Metadata filter zorunlu hale getir",
                "Context dışı cevapları yasaklayan prompt guard ekle"
            ]
        })

    if means["accuracy"] < 3:
        actions.append({
            "problem": "Low Accuracy",
            "interpretation": "Yanlış bilgi veya eksik ground truth kapsaması",
            "actions": [
                "Retrieval query rewriting ekle",
                "Ground truth kapsayan dokümanları genişlet",
                "Hybrid search (BM25 + embedding) kullan",
            ]
        })

    if means["relevance"] < 3:
        actions.append({
            "problem": "Low Relevance",
            "interpretation": "Cevaplar soruya tam odaklanmıyor",
            "actions": [
                "System prompt’u kısalt ve netleştir",
                "Answer format zorlaması ekle (bullet / step)",
                "‘Soruyu tekrar etme’ ve ‘yalnızca isteneni cevapla’ guard’ları"
            ]
        })

    if means["completeness"] < 3:
        actions.append({
            "problem": "Low Completeness",
            "interpretation": "Cevaplar eksik / yüzeysel",
            "actions": [
                "Cevap öncesi checklist yaklaşımı",
                "‘Önce düşün, sonra cevapla’ (hidden CoT) yapısı",
                "Multi-pass answer generation (draft → refine)"
            ]
        })

    return {
        "mean_scores": means.to_dict(),
        "recommendations": actions
    }