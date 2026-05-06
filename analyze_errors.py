"""
Per-image error analysis for student vs teacher at a given downscale.
Run on any eval output folder to get a structured bias/improvement report.

Usage:
    python analyze_errors.py --eval_dir student_eval_outputs/e50_lr3e-5_clw0.5_ds2
"""
import csv
import statistics
import argparse
import os


def load(path):
    with open(path) as f:
        return {r["image"]: r for r in csv.DictReader(f)}


def analyze(eval_dir):
    # Find the downscale from the folder structure
    teacher_1x_path = os.path.join(eval_dir, "teacher_1x", "all_results.csv")
    student_dirs = [d for d in os.listdir(eval_dir) if d.startswith("student_")]
    teacher_dirs = [d for d in os.listdir(eval_dir) if d.startswith("teacher_") and d != "teacher_1x"]

    if not student_dirs or not teacher_dirs:
        print(f"Could not find student/teacher result dirs in {eval_dir}")
        return

    student_dir = os.path.join(eval_dir, sorted(student_dirs)[0])
    teacher_lr_dir = os.path.join(eval_dir, sorted(teacher_dirs)[0])

    t_lr = load(os.path.join(teacher_lr_dir, "all_results.csv"))
    s = load(os.path.join(student_dir, "all_results.csv"))

    rows = []
    for img in t_lr:
        if img not in s:
            continue
        t_err = float(t_lr[img]["abs_error"])
        s_err = float(s[img]["abs_error"])
        gt = float(t_lr[img]["gt_count"])
        t_pred = float(t_lr[img]["pred_count"])
        s_pred = float(s[img]["pred_count"])
        rows.append({
            "image": img, "gt": gt,
            "t_pred": t_pred, "s_pred": s_pred,
            "t_err": t_err, "s_err": s_err,
            "improvement": t_err - s_err,
        })

    improved = [r for r in rows if r["improvement"] > 0]
    worsened = [r for r in rows if r["improvement"] < 0]

    print(f"\n{'='*60}")
    print(f"Eval dir: {eval_dir}")
    print(f"{'='*60}")

    print(f"\n--- Prediction bias ---")
    print(f"Mean GT:                {statistics.mean(r['gt'] for r in rows):.1f}")
    print(f"Mean teacher@LR pred:   {statistics.mean(r['t_pred'] for r in rows):.1f}")
    print(f"Mean student pred:      {statistics.mean(r['s_pred'] for r in rows):.1f}")

    print(f"\n--- Win/loss ---")
    print(f"Student improves: {len(improved)}/{len(rows)} images ({100*len(improved)/len(rows):.1f}%)")
    print(f"Student worsens:  {len(worsened)}/{len(rows)} images ({100*len(worsened)/len(rows):.1f}%)")
    print(f"Avg GT where student improves: {statistics.mean(r['gt'] for r in improved):.1f}")
    print(f"Avg GT where student worsens:  {statistics.mean(r['gt'] for r in worsened):.1f}")

    print(f"\n--- By crowd density ---")
    bins = [(0, 50, "sparse"), (50, 200, "medium"), (200, 500, "dense"), (500, 99999, "very dense")]
    for lo, hi, label in bins:
        subset = [r for r in rows if lo <= r["gt"] < hi]
        if not subset:
            continue
        wins = sum(1 for r in subset if r["improvement"] > 0)
        avg_imp = statistics.mean(r["improvement"] for r in subset)
        print(f"  {label:>12} (GT {lo:>4}–{hi if hi < 99999 else '∞':>4}): "
              f"{wins:>3}/{len(subset):<3} wins, avg improvement={avg_imp:>+.1f}")

    print(f"\n--- Top 10 where student wins most ---")
    print(f"  {'image':<12} {'GT':>8} {'T err':>10} {'S err':>10} {'Gain':>10}")
    for r in sorted(improved, key=lambda x: x["improvement"], reverse=True)[:10]:
        print(f"  {r['image']:<12} {r['gt']:>8.0f} {r['t_err']:>10.1f} {r['s_err']:>10.1f} {r['improvement']:>+10.1f}")

    print(f"\n--- Top 10 where student is worst ---")
    print(f"  {'image':<12} {'GT':>8} {'T err':>10} {'S err':>10} {'Loss':>10}")
    for r in sorted(worsened, key=lambda x: x["improvement"])[:10]:
        print(f"  {r['image']:<12} {r['gt']:>8.0f} {r['t_err']:>10.1f} {r['s_err']:>10.1f} {r['improvement']:>+10.1f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Path to eval output folder (e.g. student_eval_outputs/e50_lr1e-5_ds2)")
    args = parser.parse_args()
    analyze(args.eval_dir)
