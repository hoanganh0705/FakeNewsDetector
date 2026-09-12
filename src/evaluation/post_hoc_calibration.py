"""
Post-hoc calibration methods (Phase 2 of the implementation plan).

Three re-calibrators are provided:

* ``platt_scaling``        — fits a 1-D logistic regression on the model's
  logits.  Same parameterisation as the original Platt (1999) trick,
  simplified to the binary case (single slope + intercept).
* ``temperature_scaling``  — fits a single scalar ``T ≥ 0`` that divides
  the logits before the sigmoid (Guo et al., 2017).  Optimised with
  L-BFGS on the validation NLL.
* ``isotonic_regression``  — non-parametric monotone recalibrator
  (Zadrozny & Elkan, 2002).  More flexible but prone to overfit on small
  validation sets.

All three expose the same contract:

    def f(logits: np.ndarray) -> np.ndarray:   # returns calibrated probabilities

The logits are *single-axis* — the score for the positive class only
(matching the format saved in ``raw_logits.pkl``).

A second helper, ``evaluate_recalibration``, re-uses
``calibration_analysis.compute_metrics`` so the same metrics (ECE, MCE,
Brier, accuracy, F1) are reported for original vs. recalibrated
probabilities.

Public API (Step 2.1 of ``IMPLEMENTATION_PLAN.md`` §6)
------------------------------------------------------
* ``platt_scaling(logits, y) -> CalibratedProb``
* ``temperature_scaling(logits, y, T_init=1.0) -> CalibratedProb``
* ``isotonic_regression(logits, y) -> CalibratedProb``
* ``evaluate_recalibration(y_true, y_prob_orig, y_prob_recal) -> dict``
* ``run_post_hoc_calibration(experiments_dir, ...)`` — orchestrator
  that runs all 4 models × 3 methods × val→test and writes the
  results table / figure required by the paper.

A ``CalibratedProb`` is a small dataclass holding ``(predict_proba_fn,
params_dict)`` so the orchestrator can persist a human-readable summary
of the calibration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize

from config import cfg
from src.evaluation.calibration_analysis import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from src.utils.common import MODEL_DIR_MAP
from src.utils.logger import get_logger
from src.evaluation.metrics import compute_metrics

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# CalibratedProb container
# ──────────────────────────────────────────────────────────────────────


@dataclass
class CalibratedProb:
    """Output of any recalibration method.

    Attributes:
        method: Human-readable name (e.g. ``"platt"``).
        predict_proba: Callable mapping raw logit → calibrated P(y=1|x).
        params: Dict of fitted hyperparameters — persisted to JSON so
            the calibration can be re-applied to new data without
            refitting.
    """

    method: str
    predict_proba: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    params: Dict[str, float] = field(default_factory=dict)

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return self.predict_proba(np.asarray(logits, dtype=np.float64))


# ──────────────────────────────────────────────────────────────────────
# 1. Platt scaling
# ──────────────────────────────────────────────────────────────────────


def platt_scaling(
    logits: np.ndarray,
    y: np.ndarray,
) -> CalibratedProb:
    """Fit a 1-D Platt sigmoid on the raw logits.

    The calibrated probability is

        p(y=1 | x) = sigmoid(a · logit(x) + b)

    with ``a, b`` chosen to minimise validation binary cross-entropy.
    We delegate to ``sklearn.linear_model.LogisticRegression`` on a
    single-feature design matrix — equivalent to the original Platt
    (1999) parameterisation but with a proper L-BFGS / L2 solver.

    Args:
        logits: 1-D array of raw model scores for the positive class.
        y:      1-D array of binary labels (0/1).

    Returns:
        ``CalibratedProb`` whose ``predict_proba`` returns calibrated
        probabilities in [0, 1].
    """
    from sklearn.linear_model import LogisticRegression

    z = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.int64)
    if z.shape[0] != y.shape[0]:
        raise ValueError(
            f"logits ({z.shape[0]}) and y ({y.shape[0]}) must have the same length"
        )
    if z.shape[0] < 2:
        raise ValueError("platt_scaling needs at least 2 samples")

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(z, y)

    a = float(model.coef_.ravel()[0])
    b = float(model.intercept_.ravel()[0])

    def _predict(raw_logits: np.ndarray) -> np.ndarray:
        zz = np.asarray(raw_logits, dtype=np.float64).reshape(-1)
        return _sigmoid(a * zz + b)

    return CalibratedProb(
        method="platt",
        predict_proba=_predict,
        params={"a": a, "b": b},
    )


# ──────────────────────────────────────────────────────────────────────
# 2. Temperature scaling
# ──────────────────────────────────────────────────────────────────────


def temperature_scaling(
    logits: np.ndarray,
    y: np.ndarray,
    T_init: float = 1.0,
) -> CalibratedProb:
    """Fit a single-temperature scalar (Guo et al., 2017).

    The calibrated probability is ``sigmoid(logit(x) / T)``.  ``T`` is
    found by minimising the binary cross-entropy on the validation
    set with L-BFGS.  We clip ``T`` to ``[1e-3, 1e3]`` so the
    optimiser cannot collapse the sigmoid into a step function.

    Args:
        logits: 1-D array of raw model scores for the positive class.
        y:      1-D array of binary labels (0/1).
        T_init: Initial guess for ``T`` (default 1.0 = no change).

    Returns:
        ``CalibratedProb`` with ``predict_proba(logit) = sigmoid(logit/T)``.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64)

    # Objective: binary cross-entropy with sigmoid(z / T).
    def _nll(T: np.ndarray) -> float:
        T_val = float(T[0])
        T_val = float(np.clip(T_val, 1e-3, 1e3))
        p = _sigmoid(z / T_val)
        # Standard log-loss with epsilon clipping for stability.
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize(
        _nll,
        x0=np.array([T_init], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(1e-3, 1e3)],
    )
    T_opt = float(np.clip(res.x[0], 1e-3, 1e3))

    def _predict(raw_logits: np.ndarray) -> np.ndarray:
        zz = np.asarray(raw_logits, dtype=np.float64).reshape(-1)
        return _sigmoid(zz / T_opt)

    return CalibratedProb(
        method="temperature",
        predict_proba=_predict,
        params={"T": T_opt, "nll_opt": float(res.fun)},
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Isotonic regression
# ──────────────────────────────────────────────────────────────────────


def isotonic_regression(
    logits: np.ndarray,
    y: np.ndarray,
) -> CalibratedProb:
    """Non-parametric monotone recalibration (Zadrozny & Elkan, 2002).

    We feed ``sigmoid(logit)`` as the input feature so the recalibrator
    learns a monotone mapping from ``p`` to ``p'``.  This is the
    standard recipe when raw logits come from a probabilistic classifier
    (Niculescu-Mizil & Caruana, 2005).

    Args:
        logits: 1-D array of raw model scores for the positive class.
        y:      1-D array of binary labels (0/1).

    Returns:
        ``CalibratedProb`` whose ``predict_proba`` applies the fitted
        isotonic mapping.
    """
    from sklearn.isotonic import IsotonicRegression

    p = _sigmoid(np.asarray(logits, dtype=np.float64).reshape(-1))
    y = np.asarray(y, dtype=np.float64)

    if p.shape[0] != y.shape[0]:
        raise ValueError(
            f"logits ({p.shape[0]}) and y ({y.shape[0]}) must have the same length"
        )

    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p, y)

    # Persist the threshold / output pairs as JSON-friendly lists.
    x_thresholds = ir.X_thresholds_.tolist()
    y_thresholds = ir.y_thresholds_.tolist()

    def _predict(raw_logits: np.ndarray) -> np.ndarray:
        pp = _sigmoid(np.asarray(raw_logits, dtype=np.float64).reshape(-1))
        return np.clip(ir.predict(pp), 0.0, 1.0)

    return CalibratedProb(
        method="isotonic",
        predict_proba=_predict,
        params={
            "n_thresholds": int(len(x_thresholds)),
            "x_thresholds_first": float(x_thresholds[0]) if x_thresholds else 0.0,
            "x_thresholds_last":  float(x_thresholds[-1]) if x_thresholds else 0.0,
            "y_thresholds_first": float(y_thresholds[0]) if y_thresholds else 0.0,
            "y_thresholds_last":  float(y_thresholds[-1]) if y_thresholds else 0.0,
        },
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Evaluation helper
# ──────────────────────────────────────────────────────────────────────


def evaluate_recalibration(
    y_true: np.ndarray,
    y_prob_orig: np.ndarray,
    y_prob_recal: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compare original vs. recalibrated probabilities.

    Returns a dict with two sub-dicts, ``"original"`` and
    ``"recalibrated"``, each containing

    * ``accuracy``           — at threshold 0.5
    * ``f1_macro``           — macro-averaged
    * ``roc_auc``            — AUC of the positive class
    * ``ece``                — expected calibration error (10 bins)
    * ``mce``                — maximum calibration error
    * ``brier``              — Brier score
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob_orig = np.asarray(y_prob_orig, dtype=np.float64).reshape(-1)
    y_prob_recal = np.asarray(y_prob_recal, dtype=np.float64).reshape(-1)

    out: Dict[str, Dict[str, float]] = {}
    for label, p in [("original", y_prob_orig), ("recalibrated", y_prob_recal)]:
        y_pred = (p >= 0.5).astype(np.int64)
        try:
            metrics = compute_metrics(y_true, y_pred, p)
        except Exception:
            metrics = {"accuracy": float("nan"), "f1_macro": float("nan"), "roc_auc": float("nan")}
        metrics.update({
            "ece": expected_calibration_error(y_true, p, n_bins),
            "mce": maximum_calibration_error(y_true, p, n_bins),
            "brier": brier_score(y_true, p),
        })
        out[label] = {k: float(v) for k, v in metrics.items()
                      if isinstance(v, (int, float, np.floating))}
    return out


# ──────────────────────────────────────────────────────────────────────
# 5. Orchestrator — runs 4 models × 3 methods × val→test
# ──────────────────────────────────────────────────────────────────────


def run_post_hoc_calibration(
    experiments_dir: Optional[str] = None,
    figures_dir: Optional[str] = None,
    tables_dir: Optional[str] = None,
    n_bins: int = 10,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Run all four models × three recalibration methods.

    For each model we:

    1. Load ``raw_logits.pkl`` (val + test) and ``predictions.pkl``.
    2. Fit Platt, Temperature and Isotonic on the *validation* split.
    3. Apply each recalibrator to the *test* logits.
    4. Save recalibrated probabilities to ``raw_logits_recal.pkl``
       (per-model) and write a 16-row CSV / LaTeX summary.

    Args:
        experiments_dir: Where to find ``raw_logits.pkl`` /
            ``predictions.pkl`` (default: ``cfg.PATHS.experiments_dir``).
        figures_dir:     Where to save reliability diagrams (default:
            ``cfg.PATHS.paper_figures_dir``).
        tables_dir:      Where to save the post-hoc calibration table
            (default: ``cfg.PATHS.paper_tables_dir``).
        n_bins:          Bin count for ECE / MCE.

    Returns:
        Nested dict::

            {model_name: {
                "original":    {"accuracy", "f1_macro", "roc_auc", "ece", "mce", "brier"},
                "platt":       {...},
                "temperature": {...},
                "isotonic":    {...},
            }}

        Numbers are *test-set* metrics; val-set numbers are stored
        separately inside the per-model ``raw_logits_recal.pkl``.
    """
    experiments_dir = experiments_dir or cfg.PATHS.experiments_dir
    figures_dir = figures_dir or cfg.PATHS.paper_figures_dir
    tables_dir = tables_dir or cfg.PATHS.paper_tables_dir
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("  POST-HOC CALIBRATION (Phase 2)")
    log.info("=" * 70)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    fitted: Dict[str, Dict[str, CalibratedProb]] = {}

    for model_name, dir_name in MODEL_DIR_MAP.items():
        model_dir = Path(experiments_dir) / dir_name
        logits_path = model_dir / "raw_logits.pkl"
        preds_path = model_dir / "predictions.pkl"

        if not logits_path.exists() or not preds_path.exists():
            log.warning(
                "[%s] missing %s or %s — skipping",
                model_name, logits_path, preds_path,
            )
            continue

        logit_bundle = joblib.load(logits_path)
        pred_bundle = joblib.load(preds_path)

        val_logits = np.asarray(logit_bundle["val"]["raw_logit"], dtype=np.float64)
        val_y = np.asarray(logit_bundle["val"]["y_true"], dtype=np.int64)
        test_logits = np.asarray(logit_bundle["test"]["raw_logit"], dtype=np.float64)
        test_y = np.asarray(logit_bundle["test"]["y_true"], dtype=np.int64)
        test_prob_orig = np.asarray(pred_bundle["y_prob"], dtype=np.float64).reshape(-1)

        # ── Fit the three recalibrators on the *validation* logits ──
        try:
            platt = platt_scaling(val_logits, val_y)
        except Exception as exc:
            log.warning("[%s] Platt failed: %s", model_name, exc)
            platt = None
        try:
            ts = temperature_scaling(val_logits, val_y)
        except Exception as exc:
            log.warning("[%s] Temperature failed: %s", model_name, exc)
            ts = None
        try:
            iso = isotonic_regression(val_logits, val_y)
        except Exception as exc:
            log.warning("[%s] Isotonic failed: %s", model_name, exc)
            iso = None

        # ── Apply on test set ──
        test_prob_platt = platt(test_logits) if platt else None
        test_prob_temp  = ts(test_logits) if ts else None
        test_prob_iso   = iso(test_logits) if iso else None

        # ── Build the result dict for this model ──
        model_results: Dict[str, Dict[str, float]] = {}
        # Original probabilities first.
        model_results["original"] = evaluate_recalibration(test_y, test_prob_orig, test_prob_orig)["original"]
        if test_prob_platt is not None:
            model_results["platt"] = evaluate_recalibration(
                test_y, test_prob_orig, test_prob_platt,
            )["recalibrated"]
        if test_prob_temp is not None:
            model_results["temperature"] = evaluate_recalibration(
                test_y, test_prob_orig, test_prob_temp,
            )["recalibrated"]
        if test_prob_iso is not None:
            model_results["isotonic"] = evaluate_recalibration(
                test_y, test_prob_orig, test_prob_iso,
            )["recalibrated"]

        results[model_name] = model_results
        fitted[model_name] = {
            "platt": platt,
            "temperature": ts,
            "isotonic": iso,
        }

        # ── Persist recalibrated logits + params ──
        recal_bundle = {
            "model_name": model_name,
            "methods": {
                "platt":       asdict_safe(platt),
                "temperature": asdict_safe(ts),
                "isotonic":    asdict_safe(iso),
            },
            "test_prob_recalibrated": {
                "platt":       _to_list(test_prob_platt) if test_prob_platt is not None else None,
                "temperature": _to_list(test_prob_temp)  if test_prob_temp  is not None else None,
                "isotonic":    _to_list(test_prob_iso)   if test_prob_iso   is not None else None,
            },
            "y_true_test": _to_list(test_y),
            "y_prob_orig": _to_list(test_prob_orig),
        }
        out_pkl = model_dir / "raw_logits_recal.pkl"
        joblib.dump(recal_bundle, out_pkl)
        log.info("[%s] recalibrated probabilities → %s", model_name, out_pkl)

        log.info(
            "[%s] orig ECE=%.4f | platt=%.4f | temp=%.4f | iso=%.4f",
            model_name,
            model_results["original"]["ece"],
            model_results.get("platt", {}).get("ece", float("nan")),
            model_results.get("temperature", {}).get("ece", float("nan")),
            model_results.get("isotonic", {}).get("ece", float("nan")),
        )

    # ── Render reliability diagrams & tables ────────────────────────────
    _render_reliability_grid(results, figures_dir, n_bins)
    _render_post_hoc_latex(results, tables_dir)
    _write_post_hoc_csv(results, tables_dir)

    return results


# ──────────────────────────────────────────────────────────────────────
# Visualization — 2×4 grid of reliability diagrams (Step 2.3)
# ──────────────────────────────────────────────────────────────────────


def _render_reliability_grid(
    results: Dict[str, Dict[str, Dict[str, float]]],
    figures_dir: str,
    n_bins: int,
) -> str:
    """Render ``fig_reliability_diagrams_before_after.png`` (2 × 4 grid).

    Top row: original reliability for each of the 4 models.
    Bottom row: best-recalibration reliability for each model.

    The reliability curves are recomputed from the *raw* probabilities
    persisted in each model's ``raw_logits_recal.pkl`` bundle so the
    visualisation reflects the actual data, not a synthetic sketch.
    """
    import matplotlib.pyplot as plt

    from src.evaluation.calibration_analysis import compute_calibration_curve

    model_order = [m for m in ("Logistic Regression", "SVM", "BiLSTM", "PhoBERT")
                    if m in results]
    n_models = len(model_order)
    if n_models == 0:
        log.warning("No models available for reliability diagram.")
        return ""

    fig, axes = plt.subplots(2, n_models, figsize=(4.2 * n_models, 8.4), squeeze=False)

    # We need the test probabilities to draw the curves.  Reload from the
    # per-model ``raw_logits_recal.pkl`` bundle.
    raw_prob_cache: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name in model_order:
        dir_name = MODEL_DIR_MAP[model_name]
        recal_path = Path(cfg.PATHS.experiments_dir) / dir_name / "raw_logits_recal.pkl"
        if recal_path.exists():
            raw_prob_cache[model_name] = joblib.load(recal_path)

    for col, model_name in enumerate(model_order):
        orig = results[model_name]["original"]
        # Find best (lowest ECE) recalibration method.
        best_method = "original"
        best_ece = orig["ece"]
        for m in ("platt", "temperature", "isotonic"):
            if m in results[model_name] and results[model_name][m]["ece"] < best_ece:
                best_method = m
                best_ece = results[model_name][m]["ece"]

        # ── top row: original ──
        bundle = raw_prob_cache.get(model_name)
        if bundle is not None:
            y_true = np.asarray(bundle["y_true_test"], dtype=np.int64)
            p_orig = np.asarray(bundle["y_prob_orig"], dtype=np.float64)
        else:
            y_true = np.array([0])
            p_orig = np.array([0.0])
        _plot_reliability_row(
            axes[0, col],
            title=f"{model_name}\n(Original, ECE={orig['ece']:.3f})",
            y_true=y_true, y_prob=p_orig, n_bins=n_bins,
        )

        # ── bottom row: best recalibrator ──
        if best_method == "original" or bundle is None or not bundle["test_prob_recalibrated"].get(best_method):
            axes[1, col].text(
                0.5, 0.5, "No recalibration\navailable",
                ha="center", va="center", transform=axes[1, col].transAxes,
            )
            axes[1, col].set_axis_off()
        else:
            p_recal = np.asarray(bundle["test_prob_recalibrated"][best_method], dtype=np.float64)
            best = results[model_name][best_method]
            _plot_reliability_row(
                axes[1, col],
                title=f"{model_name}\n({best_method}, ECE={best['ece']:.3f})",
                y_true=y_true, y_prob=p_recal, n_bins=n_bins,
            )

    fig.suptitle(
        "Reliability diagrams — before (top) vs after best post-hoc calibration (bottom)",
        fontsize=14, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "fig_reliability_diagrams_before_after.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved reliability grid → %s", save_path)
    return save_path


def _plot_reliability_row(
    ax,
    title: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> None:
    """Draw one reliability panel — perfect diagonal + the model's curve.

    Uses ``calibration_analysis.compute_calibration_curve`` so the
    binning is consistent with the rest of the project (uniform-width,
    ``n_bins`` buckets, last bin inclusive).
    """
    from src.evaluation.calibration_analysis import compute_calibration_curve

    centres, fracs, counts = compute_calibration_curve(y_true, y_prob, n_bins)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.0, label="Perfect")
    if centres.size > 0:
        ax.plot(centres, fracs, "s-", color="#d62728", linewidth=1.8,
                markersize=5, label="Model")
        # Bar histogram of bucket counts on a secondary axis to mirror
        # the standard reliability-diagram look.
        ax2 = ax.twinx()
        ax2.bar(centres, counts, width=1.0 / n_bins * 0.7, alpha=0.15,
                color="#1f77b4", label="Count")
        ax2.set_ylim(0, max(counts) * 3 if counts.size > 0 else 1)
        ax2.set_ylabel("Count", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


# ──────────────────────────────────────────────────────────────────────
# LaTeX + CSV exports (Step 2.3)
# ──────────────────────────────────────────────────────────────────────


def _render_post_hoc_latex(
    results: Dict[str, Dict[str, Dict[str, float]]],
    tables_dir: str,
) -> str:
    """Write ``table_post_hoc_calibration.tex`` to ``tables_dir``.

    Layout:

        | Model | ECE (orig) | ECE (Platt) | ECE (Temp) | ECE (Isotonic) | Brier (best) |
    """
    model_order = [m for m in ("Logistic Regression", "SVM", "BiLSTM", "PhoBERT")
                    if m in results]
    if not model_order:
        return ""

    methods = ("platt", "temperature", "isotonic")
    method_labels = {"platt": "Platt", "temperature": "Temp", "isotonic": "Isotonic"}

    rows: List[str] = []
    for name in model_order:
        orig_ece = results[name]["original"]["ece"]
        cells = [name, _fmt(orig_ece)]
        for m in methods:
            cells.append(_fmt(results[name].get(m, {}).get("ece", float("nan"))))
        # Best (lowest) Brier across the 3 recalibrators (or original).
        candidates = {m: results[name][m]["brier"] for m in methods if m in results[name]}
        if candidates:
            best_brier = min(candidates.values())
        else:
            best_brier = results[name]["original"]["brier"]
        cells.append(_fmt(best_brier))
        rows.append(" & ".join(cells) + r" \\")

    latex = (
        "% Auto-generated by src/evaluation/post_hoc_calibration.py\n"
        r"\begin{table}[H]" "\n"
        r"  \centering" "\n"
        r"  \caption{So sánh hiệu chuẩn sau khi áp dụng ba phương pháp post-hoc calibration "
        r"(Platt, Temperature, Isotonic) trên tập kiểm tra. ECE giảm ⇒ hiệu chuẩn tốt hơn.}"
        "\n"
        r"  \label{tab:post-hoc-calibration}" "\n"
        r"  \begin{tabular}{lcccc}" "\n"
        r"    \toprule" "\n"
        r"    \textbf{Mô hình} & \textbf{ECE (gốc)} $\downarrow$ & "
        r"\textbf{ECE (Platt)} $\downarrow$ & \textbf{ECE (Temp)} $\downarrow$ & "
        r"\textbf{ECE (Isotonic)} $\downarrow$ & \textbf{Brier (tốt nhất)} $\downarrow$ \\" "\n"
        r"    \midrule" "\n"
        + "\n".join(rows) + "\n"
        + r"    \bottomrule" + "\n"
        + r"  \end{tabular}" + "\n"
        + r"\end{table}" + "\n"
    )

    save_path = os.path.join(tables_dir, "table_post_hoc_calibration.tex")
    os.makedirs(tables_dir, exist_ok=True)
    with open(save_path, "w") as fh:
        fh.write(latex)
    log.info("Saved post-hoc calibration LaTeX table → %s", save_path)
    return save_path


def _write_post_hoc_csv(
    results: Dict[str, Dict[str, Dict[str, float]]],
    tables_dir: str,
) -> str:
    """Write ``post_hoc_calibration.csv`` for downstream tooling."""
    import csv

    save_path = os.path.join(tables_dir, "post_hoc_calibration.csv")
    os.makedirs(tables_dir, exist_ok=True)
    fieldnames = [
        "model", "method",
        "accuracy", "f1_macro", "roc_auc",
        "ece", "mce", "brier",
    ]
    with open(save_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, methods in results.items():
            for method, metrics in methods.items():
                writer.writerow({
                    "model": model_name,
                    "method": method,
                    "accuracy": metrics.get("accuracy"),
                    "f1_macro": metrics.get("f1_macro"),
                    "roc_auc": metrics.get("roc_auc"),
                    "ece": metrics.get("ece"),
                    "mce": metrics.get("mce"),
                    "brier": metrics.get("brier"),
                })
    log.info("Saved post-hoc calibration CSV → %s", save_path)
    return save_path


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _to_list(arr: Optional[np.ndarray]) -> Optional[list]:
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float64).tolist()


def _fmt(x: float) -> str:
    """Format a float as ``0{,}1234`` (Vietnamese LaTeX decimal style)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(v):
        return "--"
    return f"{v:.4f}".replace(".", "{,}")


def asdict_safe(cal: Optional[CalibratedProb]) -> Optional[Dict]:
    """Serialise a ``CalibratedProb`` to a JSON-friendly dict (no callables)."""
    if cal is None:
        return None
    return {"method": cal.method, "params": cal.params}


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point — equivalent to ``fakenews recalibrate``."""
    log.info("=" * 60)
    log.info("  POST-HOC CALIBRATION (Phase 2)")
    log.info("=" * 60)
    run_post_hoc_calibration()
    log.info("Done.")


__all__ = [
    "CalibratedProb",
    "platt_scaling",
    "temperature_scaling",
    "isotonic_regression",
    "evaluate_recalibration",
    "run_post_hoc_calibration",
    "main",
]