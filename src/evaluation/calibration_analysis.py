"""
Calibration Analysis for Fake News Detection Models

Evaluates how well a model's predicted probabilities match actual outcomes.
A well-calibrated model predicting 80% probability should be correct ~80% of the time.

Metrics:
- Reliability diagrams (calibration curves)
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score

Uses saved predictions (predictions.pkl) — no retraining needed.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from config import cfg
from src.utils.common import MODEL_DIR_MAP
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Publication-quality plot settings ────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

MODEL_COLORS = {
    'Logistic Regression': '#1f77b4',
    'SVM': '#ff7f0e',
    'BiLSTM': '#2ca02c',
    'PhoBERT': '#d62728',
}


# ── Core calibration metrics ────────────────────────────────────────────────

def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve (reliability diagram data).

    Args:
        y_true:   Binary ground-truth labels (0/1).
        y_prob:   Predicted probability of positive class.
        n_bins:   Number of bins.
        strategy: 'uniform' (equal-width) or 'quantile' (equal-count).

    Returns:
        bin_centers:       Midpoint of each bin on x-axis (mean predicted prob).
        bin_true_fracs:    Fraction of positives in each bin (actual accuracy).
        bin_counts:        Number of samples per bin.
    """
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges = np.unique(bin_edges)  # deduplicate
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_centers = []
    bin_true_fracs = []
    bin_counts = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        count = mask.sum()
        if count == 0:
            continue

        bin_centers.append(y_prob[mask].mean())
        bin_true_fracs.append(y_true[mask].mean())
        bin_counts.append(count)

    return np.array(bin_centers), np.array(bin_true_fracs), np.array(bin_counts)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Lower is better.  0 = perfectly calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        count = mask.sum()
        if count == 0:
            continue

        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (count / n) * abs(acc - conf)

    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Maximum Calibration Error (MCE).

    MCE = max_b |acc(B_b) - conf(B_b)|
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        count = mask.sum()
        if count == 0:
            continue

        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        mce = max(mce, abs(acc - conf))

    return mce


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score.  Lower is better.  0 = perfect."""
    return float(np.mean((y_prob - y_true) ** 2))


# ── Load predictions helper ─────────────────────────────────────────────────

def load_all_predictions() -> Dict[str, Dict[str, np.ndarray]]:
    """Load saved predictions for all available models.

    Returns:
        {model_name: {'y_true': ..., 'y_pred': ..., 'y_prob': ...}}
    """
    all_preds = {}
    for model_name, dir_name in MODEL_DIR_MAP.items():
        pred_path = os.path.join(cfg.PATHS.experiments_dir, dir_name, 'predictions.pkl')
        if os.path.exists(pred_path):
            preds = joblib.load(pred_path)
            all_preds[model_name] = {
                'y_true': np.array(preds['y_true']),
                'y_pred': np.array(preds['y_pred']),
                'y_prob': np.array(preds['y_prob']),
            }
            log.info("Loaded predictions for %s (%d samples)", model_name, len(preds['y_true']))
        else:
            log.warning("No predictions found for %s at %s", model_name, pred_path)
    return all_preds


# ── Per-model calibration analysis ──────────────────────────────────────────

def analyze_calibration(
    all_preds: Dict[str, Dict[str, np.ndarray]],
    n_bins: int = 10,
) -> Dict[str, dict]:
    """Compute calibration metrics for all models.

    Returns:
        {model_name: {ece, mce, brier, curve: {centers, fracs, counts}}}
    """
    results = {}
    for name, preds in all_preds.items():
        y_true = preds['y_true']
        y_prob = preds['y_prob']

        centers, fracs, counts = compute_calibration_curve(y_true, y_prob, n_bins)
        ece = expected_calibration_error(y_true, y_prob, n_bins)
        mce = maximum_calibration_error(y_true, y_prob, n_bins)
        bs = brier_score(y_true, y_prob)

        results[name] = {
            'ece': ece,
            'mce': mce,
            'brier': bs,
            'curve': {
                'centers': centers.tolist(),
                'fracs': fracs.tolist(),
                'counts': counts.tolist(),
            },
        }
        log.info(
            "%s — ECE=%.4f  MCE=%.4f  Brier=%.4f",
            name, ece, mce, bs,
        )
    return results


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_calibration_curves(
    all_preds: Dict[str, Dict[str, np.ndarray]],
    cal_results: Dict[str, dict],
    save_path: str,
    n_bins: int = 10,
):
    """Create reliability diagrams for all models in a single figure.

    Layout: 2×2 grid of individual calibration curves + a combined overlay.
    """
    model_names = [n for n in MODEL_DIR_MAP.keys() if n in all_preds]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        res = cal_results[name]
        centers = np.array(res['curve']['centers'])
        fracs = np.array(res['curve']['fracs'])
        counts = np.array(res['curve']['counts'])
        color = MODEL_COLORS.get(name, '#333333')

        # Perfect calibration reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Hiệu chuẩn hoàn hảo')

        # Calibration curve
        ax.plot(
            centers, fracs, 's-',
            color=color, linewidth=2, markersize=6,
            label=f'{name}\n(ECE={res["ece"]:.4f})',
        )

        # Histogram of predictions
        ax2 = ax.twinx()
        ax2.bar(
            centers, counts, width=1.0 / n_bins * 0.7,
            alpha=0.15, color=color, label='Số mẫu',
        )
        ax2.set_ylabel('Số mẫu', fontsize=9, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
        ax2.set_ylim(0, max(counts) * 3 if len(counts) > 0 else 1)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Xác suất dự đoán trung bình')
        ax.set_ylabel('Tỷ lệ mẫu dương thực tế')
        ax.set_title(name, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Biểu đồ hiệu chuẩn (Reliability Diagram) của bốn mô hình',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    log.info("Saved calibration curves → %s", save_path)


def plot_calibration_overlay(
    all_preds: Dict[str, Dict[str, np.ndarray]],
    cal_results: Dict[str, dict],
    save_path: str,
):
    """Single-axis overlay of all models' calibration curves for comparison."""
    model_names = [n for n in MODEL_DIR_MAP.keys() if n in all_preds]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Hiệu chuẩn hoàn hảo')

    for name in model_names:
        res = cal_results[name]
        centers = np.array(res['curve']['centers'])
        fracs = np.array(res['curve']['fracs'])
        color = MODEL_COLORS.get(name, '#333333')

        ax.plot(
            centers, fracs, 's-',
            color=color, linewidth=2, markersize=5,
            label=f"{name} (ECE={res['ece']:.4f})",
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Xác suất dự đoán trung bình')
    ax.set_ylabel('Tỷ lệ mẫu dương thực tế')
    ax.set_title('So sánh hiệu chuẩn xác suất giữa các mô hình', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    log.info("Saved calibration overlay → %s", save_path)


# ── Save results ────────────────────────────────────────────────────────────

def save_calibration_results(
    cal_results: Dict[str, dict],
    tables_dir: str,
) -> Tuple[str, str]:
    """Save calibration metrics as JSON and CSV.

    Returns:
        (json_path, csv_path)
    """
    os.makedirs(tables_dir, exist_ok=True)

    # JSON (full detail)
    json_path = os.path.join(tables_dir, 'calibration_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(cal_results, f, indent=2)
    log.info("Saved calibration JSON → %s", json_path)

    # CSV summary
    rows = []
    for name, res in cal_results.items():
        rows.append({
            'Model': name,
            'ECE': round(res['ece'], 4),
            'MCE': round(res['mce'], 4),
            'Brier Score': round(res['brier'], 4),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(tables_dir, 'calibration_summary.csv')
    df.to_csv(csv_path, index=False)
    log.info("Saved calibration CSV → %s", csv_path)

    return json_path, csv_path


# ── LaTeX table generation ──────────────────────────────────────────────────

def generate_calibration_latex_table(
    cal_results: Dict[str, dict],
    save_path: str,
):
    """Generate a LaTeX table of calibration metrics for the paper."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Chỉ số hiệu chuẩn xác suất của bốn mô hình trên tập kiểm tra}",
        r"\label{tab:calibration}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Mô hình} & \textbf{ECE} $\downarrow$ & \textbf{MCE} $\downarrow$ & \textbf{Brier Score} $\downarrow$ \\",
        r"\midrule",
    ]

    # Find the best (lowest) for bolding
    best_ece = min(r['ece'] for r in cal_results.values())
    best_mce = min(r['mce'] for r in cal_results.values())
    best_brier = min(r['brier'] for r in cal_results.values())

    model_order = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    for name in model_order:
        if name not in cal_results:
            continue
        r = cal_results[name]

        ece_str = f"{r['ece']:.4f}".replace('.', '{,}')
        mce_str = f"{r['mce']:.4f}".replace('.', '{,}')
        brier_str = f"{r['brier']:.4f}".replace('.', '{,}')

        if abs(r['ece'] - best_ece) < 1e-6:
            ece_str = r"\textbf{" + ece_str + "}"
        if abs(r['mce'] - best_mce) < 1e-6:
            mce_str = r"\textbf{" + mce_str + "}"
        if abs(r['brier'] - best_brier) < 1e-6:
            brier_str = r"\textbf{" + brier_str + "}"

        lines.append(f"{name} & {ece_str} & {mce_str} & {brier_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log.info("Saved calibration LaTeX table → %s", save_path)


# ── Main entry point ────────────────────────────────────────────────────────

def main():
    """Run the complete calibration analysis pipeline."""
    log.info("=" * 60)
    log.info("  CALIBRATION ANALYSIS")
    log.info("=" * 60)

    # 1. Load predictions
    all_preds = load_all_predictions()
    if not all_preds:
        log.error("No predictions found. Run training first.")
        return

    # 2. Compute calibration metrics
    cal_results = analyze_calibration(all_preds, n_bins=10)

    # 3. Save results
    save_calibration_results(cal_results, cfg.PATHS.tables_dir)

    # 4. Generate plots
    figures_dir = cfg.PATHS.paper_figures_dir
    os.makedirs(figures_dir, exist_ok=True)

    plot_calibration_curves(
        all_preds, cal_results,
        save_path=os.path.join(figures_dir, 'fig_calibration_curves.png'),
    )
    plot_calibration_overlay(
        all_preds, cal_results,
        save_path=os.path.join(figures_dir, 'fig_calibration_overlay.png'),
    )

    # Also save to results/figures for reference
    results_fig_dir = cfg.PATHS.figures_dir
    os.makedirs(results_fig_dir, exist_ok=True)
    plot_calibration_curves(
        all_preds, cal_results,
        save_path=os.path.join(results_fig_dir, 'calibration_curves.png'),
    )

    # 5. Generate LaTeX table
    generate_calibration_latex_table(
        cal_results,
        save_path=os.path.join(cfg.PATHS.paper_tables_dir, 'table_calibration.tex'),
    )

    # 6. Print summary
    log.info("")
    log.info("=" * 60)
    log.info("  CALIBRATION SUMMARY")
    log.info("=" * 60)
    log.info("%-25s  %8s  %8s  %10s", "Model", "ECE", "MCE", "Brier")
    log.info("-" * 55)
    for name in ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']:
        if name in cal_results:
            r = cal_results[name]
            log.info("%-25s  %8.4f  %8.4f  %10.4f", name, r['ece'], r['mce'], r['brier'])
    log.info("=" * 60)

    return cal_results


if __name__ == "__main__":
    main()
