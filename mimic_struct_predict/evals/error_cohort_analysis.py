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


def assign_cohort(format_adherence: list[int]) -> str:
    """Assigns error cohort based on format adherence across trajectories."""
    n_traj = len(format_adherence)
    n_errors = n_traj - sum(format_adherence)
    if n_errors == 0:
        return "0%"
    if n_errors == n_traj:
        return "100%"
    return "5-50%" if (n_errors / n_traj <= 0.5) else "55-95%"


def export_table(df: pl.DataFrame, name: str, title: str, output_dir: Path) -> None:
    """Exports table to CSV, LaTeX, and Markdown formats."""
    # 1. Save CSV
    csv_path = output_dir / f"{name}.csv"
    df.write_csv(csv_path)

    # 2. Mask repeated modality for display formats
    pdf = df.to_pandas()
    pdf["Modality"] = pdf["Modality"].mask(pdf.duplicated(subset=["Modality"]), "")

    # 3. Save LaTeX
    tex_path = output_dir / f"{name}.tex"
    tex_path.write_text(pdf.to_latex(index=False, escape=True))

    # 4. Save Markdown
    md_path = output_dir / f"{name}.md"
    md_path.write_text(pdf.to_markdown(index=False))

    log.info(f"Saved {name} (.csv, .tex, .md) to {output_dir}")


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

    history_rows = []
    prevalence_rows = []

    for mod_name, mod_dir in sources:
        log.info(f"--- Processing {mod_name} ---")
        for task in args.tasks:
            roll_df = load_predictions(mod_dir, task, args.split)
            n_patients = len(roll_df)

            cohorts = [assign_cohort(fa) for fa in roll_df["format_adherence"]]
            n_events = [extract_history_length(t) for t in roll_df["decoded_input"]]
            bool_vals = roll_df["boolean_value"].to_list()

            hist_row = {"Modality": mod_name, "Task": format_task_title(task)}
            prev_row = {"Modality": mod_name, "Task": format_task_title(task)}

            for cohort in COHORTS:
                idxs = [i for i, c in enumerate(cohorts) if c == cohort]
                cohort_n = len(idxs)
                prop = (cohort_n / n_patients * 100.0) if n_patients > 0 else 0.0

                c_events = [n_events[i] for i in idxs]
                if c_events:
                    med = float(np.median(c_events))
                    q25 = float(np.percentile(c_events, 25))
                    q75 = float(np.percentile(c_events, 75))
                    hist_row[cohort] = f"{med:.2f} [{q25:.2f}, {q75:.2f}] ({prop:.2f}%)"
                else:
                    hist_row[cohort] = "-"

                c_labels = [bool_vals[i] for i in idxs]
                if cohort_n > 0:
                    prev = sum(c_labels) / cohort_n * 100.0
                    prev_row[cohort] = f"{prev:.2f}% ({prop:.2f}%)"
                else:
                    prev_row[cohort] = "-"

            history_rows.append(hist_row)
            prevalence_rows.append(prev_row)

    hist_df = pl.DataFrame(history_rows)
    prev_df = pl.DataFrame(prevalence_rows)

    export_table(
        hist_df,
        "history_length",
        "History Length, Median [IQR] (Cohort Percentage)",
        args.output_dir,
    )
    export_table(
        prev_df,
        "outcome_prevalence",
        "Outcome Prevalence % (Cohort Percentage)",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
