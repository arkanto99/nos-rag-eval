import json
import argparse
import os

def compute_avg_from_jsonl(path, keys):
    """Lee un .jsonl y calcula la media de las métricas indicadas."""
    values = {k: [] for k in keys}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            for k in keys:
                if k in data:
                    values[k].append(data[k])
    return {
        k: (sum(v) / len(v) if v else 0.0)
        for k, v in values.items()
    }

def aggregate_experiment(folder, output_file="global_avg.json"):
    result = {}

    # --- Traditional metrics ---
    trad_file = os.path.join(folder, "traditional_metric_results.jsonl")
    if os.path.exists(trad_file):
        result["traditional_metrics"] = compute_avg_from_jsonl(
            trad_file,
            [
                "avg_precision_paragraph",
                "avg_recall_paragraph",
                "avg_mrr_paragraph",
                "avg_precision_document",
                "avg_recall_document",
                "avg_mrr_document"
            ]
        )

    # --- Judge precision ---
    prec_file = os.path.join(folder, "judge_precision.jsonl")
    if os.path.exists(prec_file):
        result["judge_precision"] = compute_avg_from_jsonl(
            prec_file,
            ["average_context_precision"]
        )

    # --- Judge recall ---
    rec_file = os.path.join(folder, "judge_recall.jsonl")
    if os.path.exists(rec_file):
        result["judge_recall"] = compute_avg_from_jsonl(
            rec_file,
            ["average_context_recall"]
        )

    # Guardar
    out_path = os.path.join(folder, output_file)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✔ Guardado {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate metrics from experiment results")
    parser.add_argument("--folder", type=str, required=True, help="Path to experiment folder")
    parser.add_argument("--output", type=str, default="global_avg.json", help="Output filename")
    args = parser.parse_args()

    aggregate_experiment(args.folder, args.output)
