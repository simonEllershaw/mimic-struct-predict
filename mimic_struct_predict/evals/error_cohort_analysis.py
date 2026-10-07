import argparse
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl

from mimic_struct_predict.evals.error_rates import load_predictions
from mimic_struct_predict.evals.plot_style import (
    format_task_title,
    sort_tasks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

COHORTS = ["0%", "5-50%", "55-95%", "100%"]


def extract_history_length(text: str) -> int:
    """Extracts num_events from decoded_input."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sum(1 for line in lines if not line.lstrip("/").startswith("#"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate error cohorts and extract cohort analysis statistics.")
    parser.add_argument("--split", type=str, default="held_out", choices=["train", "tuning", "held_out"])
    parser.add_argument("--tasks", nargs="+", default=os.environ.get("ACES_TASKS", "").split())
    parser.add_argument(
        "--struct_dir",
        type=Path,
        default=Path(os.environ.get("STRUCTURED_INFERENCE_DIR", "")) / "pt" / "rollout",
    )
    parser.add_argument(
        "--nl_dir",
        type=Path,
        default=Path(os.environ.get("NATURAL_LANGUAGE_INFERENCE_DIR", "")) / "cpt" / "rollout",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("EVAL_PLOTS_DIR", "")) / "error_cohort_analysis",
    )
    args = parser.parse_args()
    args.tasks = sort_tasks(args.tasks)
    log.info(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        ("Structured", args.struct_dir),
        ("Natural Language", args.nl_dir),
    ]

    flat_rows = []

    for mod_name, mod_dir in sources:
        log.info(f"--- Processing {mod_name} ---")
        for task in args.tasks:
            roll_df = load_predictions(mod_dir, task, args.split)
            n_patients = len(roll_df)

            # Assign cohort based on % failed trajectories
            # 0% errors -> '0%', 5-50% errors -> '5-50%', 55-95% errors -> '55-95%', 100% errors -> '100%'
            cohorts = []
            for fa in roll_df["format_adherence"]:
                n_traj = len(fa)
                n_errors = n_traj - sum(fa)
                pct_errors = (n_errors / n_traj * 100.0) if n_traj > 0 else 0.0
                if n_errors == 0:
                    cohorts.append("0%")
                elif n_errors == n_traj:
                    cohorts.append("100%")
                elif pct_errors <= 50.0:
                    cohorts.append("5-50%")
                else:
                    cohorts.append("55-95%")

            # Extract features from decoded_input
            n_events = [extract_history_length(t) for t in roll_df["decoded_input"]]
            bool_vals = roll_df["boolean_value"].to_list()

            for cohort in COHORTS:
                idxs = [i for i, c in enumerate(cohorts) if c == cohort]
                cohort_n = len(idxs)

                prop = (cohort_n / n_patients * 100.0) if n_patients > 0 else 0.0

                c_events = [n_events[i] for i in idxs]
                if c_events:
                    med_events = float(np.median(c_events))
                    q25_events = float(np.percentile(c_events, 25))
                    q75_events = float(np.percentile(c_events, 75))
                    hist_events_str = f"{med_events:.2f} [{q25_events:.2f}, {q75_events:.2f}]"
                else:
                    hist_events_str = "-"

                c_labels = [bool_vals[i] for i in idxs]
                prev = (sum(c_labels) / cohort_n * 100.0) if cohort_n > 0 else None

                flat_rows.append(
                    {
                        "Modality": mod_name,
                        "Task": format_task_title(task),
                        "Error Rate": cohort,
                        "Patients, N (%)": f"{cohort_n:,} ({prop:.2f}%)",
                        "History Length, Median [IQR] (Num Events)": hist_events_str,
                        "Outcome Prevalence (%)": prev,
                    }
                )

    df = pl.DataFrame(flat_rows)

    # 1. Save CSV
    csv_path = args.output_dir / "error_cohort_analysis.csv"
    df.write_csv(csv_path)
    log.info(f"Saved CSV table to {csv_path}")

    # 2. Save LaTeX
    tex_path = args.output_dir / "error_cohort_analysis.tex"
    tex_path.write_text(df.to_pandas().to_latex(index=False, escape=True, float_format="%.2f"))
    log.info(f"Saved LaTeX table to {tex_path}")

    # 3. Save Markdown
    md_df = df.to_pandas()
    md_df["Task"] = md_df["Task"].mask(md_df.duplicated(subset=["Modality", "Task"]), "")
    md_df["Modality"] = md_df["Modality"].mask(md_df.duplicated(subset=["Modality"]), "")
    md_path = args.output_dir / "error_cohort_analysis.md"
    md_path.write_text(md_df.to_markdown(index=False, floatfmt=".2f"))
    log.info(f"Saved Markdown table to {md_path}")


if __name__ == "__main__":
    main()
