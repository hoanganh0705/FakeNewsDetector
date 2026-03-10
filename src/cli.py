"""
CLI entry point for the FakeNewsDetector pipeline.

Usage (after ``pip install -e .``):

    fakenews preprocess          # word segmentation
    fakenews split               # train/val/test split
    fakenews features            # extract TF-IDF, embedding & PhoBERT features
    fakenews train [MODEL ...]   # train models (lr, svm, bilstm, phobert, all)
    fakenews evaluate            # full evaluation + figures

    fakenews run                 # run the entire pipeline end-to-end
"""

import argparse
import sys
import time
from datetime import datetime

from src.utils.logger import get_logger

log = get_logger(__name__)

VALID_MODELS = ("lr", "svm", "bilstm", "phobert", "all")


# ── helpers ────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    log.info("=" * 60)
    log.info(title)
    log.info("=" * 60)


def _step_preprocess() -> None:
    _banner("STEP 1 / 5 — Word Segmentation")
    from src.preprocessing.word_segmentation import main as seg_main
    seg_main()


def _step_split() -> None:
    _banner("STEP 2 / 5 — Train / Val / Test Split")
    from src.preprocessing.split_data import main as split_main
    split_main()


def _step_features() -> None:
    _banner("STEP 3 / 5 — Feature Extraction")
    from src.features.extract_all_features import main as feat_main
    feat_main()


def _step_train(models: list[str] | None = None) -> None:
    """Train one or more models.  *models* defaults to ``["all"]``."""
    models = models or ["all"]
    _banner("STEP 4 / 5 — Training (%s)" % ", ".join(models))

    if "all" in models:
        from src.training.train_all import main as train_all
        train_all()
        return

    _dispatch = {
        "lr": ("Logistic Regression", "src.training.train_lr"),
        "svm": ("SVM", "src.training.train_svm"),
        "bilstm": ("BiLSTM", "src.training.train_bilstm"),
        "phobert": ("PhoBERT", "src.training.train_phobert"),
    }
    for m in models:
        label, module_path = _dispatch[m]
        log.info("Training %s …", label)
        import importlib
        mod = importlib.import_module(module_path)
        mod.main()


def _step_evaluate() -> None:
    _banner("STEP 5 / 5 — Evaluation")
    from src.evaluation.evaluate_all import main as eval_main
    eval_main()


def _step_calibration() -> None:
    _banner("Calibration Analysis")
    from src.evaluation.calibration_analysis import main as cal_main
    cal_main()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fakenews",
        description="FakeNewsDetector — Vietnamese fake news detection pipeline.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("preprocess", help="Run Vietnamese word segmentation")
    sub.add_parser("split", help="Split data into train / val / test")
    sub.add_parser("features", help="Extract all features (TF-IDF, embeddings, PhoBERT)")

    train_p = sub.add_parser("train", help="Train model(s)")
    train_p.add_argument(
        "models",
        nargs="*",
        default=["all"],
        choices=VALID_MODELS,
        help="Models to train (default: all)",
    )

    sub.add_parser("evaluate", help="Run full evaluation suite")
    sub.add_parser("calibration", help="Run calibration analysis (ECE, MCE, Brier)")
    sub.add_parser("run", help="Run the entire pipeline end-to-end")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    start = time.time()
    log.info(
        "fakenews %s — started at %s",
        args.command,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    if args.command == "preprocess":
        _step_preprocess()
    elif args.command == "split":
        _step_split()
    elif args.command == "features":
        _step_features()
    elif args.command == "train":
        _step_train(args.models)
    elif args.command == "evaluate":
        _step_evaluate()
    elif args.command == "calibration":
        _step_calibration()
    elif args.command == "run":
        _step_preprocess()
        _step_split()
        _step_features()
        _step_train(["all"])
        _step_evaluate()

    elapsed = time.time() - start
    log.info("Done in %.1f s.", elapsed)


if __name__ == "__main__":
    main()
