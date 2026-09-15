import argparse
import sys
import time
from datetime import datetime

from src.utils.logger import get_logger

log = get_logger(__name__)

VALID_MODELS = ("lr", "svm", "bilstm", "phobert", "all")



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


def _step_explain(args) -> None:
    _banner("Token-Level Explainability")
    from src.analysis.explainability_runner import run_explainability

    run_explainability(
        n_examples=args.n_examples,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
    )


def _step_recalibrate(args) -> None:
    _banner("Post-hoc Calibration")
    from src.evaluation.post_hoc_calibration import run_post_hoc_calibration

    run_post_hoc_calibration(
        experiments_dir=args.experiments_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        n_bins=args.n_bins,
    )


def _step_hard_cases(args) -> None:
    _banner("Hard Cases Deep Analysis")
    from src.evaluation.hard_cases import analyze_hard_examples

    analyze_hard_examples(
        per_id_confidence_path=args.per_id_confidence,
        test_csv_path=args.test_csv,
        attributions_dir=args.attributions_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
    )


def _step_distill(args) -> None:
    _banner("Knowledge Distillation")
    if args.mode in ("train", "all"):
        from src.training.train_student import main as train_main
        train_main(
            alpha=args.alpha,
            temperature=args.temperature,
            epochs=args.epochs,
            patience=args.patience,
            teacher_model=args.teacher_model,
        )
    if args.mode in ("eval", "all"):
        from src.training.distillation_evaluation import run_distillation_evaluation
        run_distillation_evaluation(
            tables_dir=args.tables_dir,
            figures_dir=args.figures_dir,
            n_runs=args.n_runs,
        )


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
        default=None,
        help="Models to train: lr, svm, bilstm, phobert, all (default: all)",
    )

    sub.add_parser("evaluate", help="Run full evaluation suite")
    sub.add_parser("calibration", help="Run calibration analysis (ECE, MCE, Brier)")

    explain_p = sub.add_parser(
        "explain",
        help="Token-level attribution (SHAP / IG / attention rollout) for all 4 models",
    )
    explain_p.add_argument(
        "--n-examples", type=int, default=20,
        help="Number of informative test samples to attribute (default: 20)",
    )
    explain_p.add_argument(
        "--output-dir", default=None,
        help="Where to save per-example attribution pickles "
             "(default: <results_dir>/attributions)",
    )
    explain_p.add_argument(
        "--figures-dir", default=None,
        help="Where to save aggregate figures (default: paper/figures)",
    )
    explain_p.add_argument(
        "--tables-dir", default=None,
        help="Where to save aggregate LaTeX tables (default: paper/tables)",
    )

    recal_p = sub.add_parser(
        "recalibrate",
        help="Post-hoc calibration (Platt / Temperature / Isotonic) for all 4 models",
    )
    recal_p.add_argument(
        "--experiments-dir", default=None,
        help="Where to find raw_logits.pkl / predictions.pkl "
             "(default: <experiments_dir>)",
    )
    recal_p.add_argument(
        "--figures-dir", default=None,
        help="Where to save the reliability-diagram grid "
             "(default: paper/figures)",
    )
    recal_p.add_argument(
        "--tables-dir", default=None,
        help="Where to save the post-hoc LaTeX table "
             "(default: paper/tables)",
    )
    recal_p.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of bins for ECE / MCE / reliability diagrams (default: 10)",
    )

    hard_p = sub.add_parser(
        "hard-cases",
        help="Deep analysis of hard cases (samples wrong by all 4 models)",
    )
    hard_p.add_argument(
        "--per-id-confidence", default=None,
        help="Path to per_id_confidence.csv "
             "(default: <tables_dir>/per_id_confidence.csv)",
    )
    hard_p.add_argument(
        "--test-csv", default=None,
        help="Path to test.csv (default: <splits_dir>/test.csv)",
    )
    hard_p.add_argument(
        "--attributions-dir", default=None,
        help="Directory of attribution pickles "
             "(default: <results_dir>/attributions)",
    )
    hard_p.add_argument(
        "--figures-dir", default=None,
        help="Where to save the distribution figure (default: paper/figures)",
    )
    hard_p.add_argument(
        "--tables-dir", default=None,
        help="Where to save the annotated LaTeX table (default: paper/tables)",
    )

    distill_p = sub.add_parser(
        "distill",
        help="Knowledge Distillation: train a small Student BiLSTM "
             "with KD loss against a teacher (default: PhoBERT) and "
             "produce the F1-vs-size trade-off figure.",
    )
    distill_p.add_argument(
        "--mode", default="all", choices=["train", "eval", "all"],
        help="What to run: train student, evaluate only, or both (default: all)",
    )
    distill_p.add_argument(
        "--alpha", type=float, default=0.7,
        help="Weight on the soft (KD) target (default: 0.7)",
    )
    distill_p.add_argument(
        "--temperature", type=float, default=4.0,
        help="KD temperature (default: 4.0)",
    )
    distill_p.add_argument(
        "--epochs", type=int, default=30,
        help="Max epochs (default: 30)",
    )
    distill_p.add_argument(
        "--patience", type=int, default=5,
        help="Early-stopping patience (default: 5)",
    )
    distill_p.add_argument(
        "--teacher-model", default="bert",
        help="Directory name of teacher inside experiments/ "
             "(bert=PhoBERT, bilstm=BiLSTM teacher; default: bert)",
    )
    distill_p.add_argument(
        "--figures-dir", default=None,
        help="Where to save the trade-off figure (default: paper/figures)",
    )
    distill_p.add_argument(
        "--tables-dir", default=None,
        help="Where to save the comparison LaTeX table (default: paper/tables)",
    )
    distill_p.add_argument(
        "--n-runs", type=int, default=30,
        help="Number of inference-timing runs per model (default: 30)",
    )

    sub.add_parser("run", help="Run the entire pipeline end-to-end")

    return parser


def _validate_models(parser: argparse.ArgumentParser, models: list[str]) -> list[str]:
    invalid = [m for m in models if m not in VALID_MODELS]
    if invalid:
        parser.error(
            "argument models: invalid choice(s): %s (choose from %s)"
            % (", ".join(repr(m) for m in invalid),
               ", ".join(repr(c) for c in VALID_MODELS))
        )
    return models


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        models = args.models or ["all"]
        _validate_models(parser, models)
        args.models = models

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
    elif args.command == "explain":
        _step_explain(args)
    elif args.command == "recalibrate":
        _step_recalibrate(args)
    elif args.command == "hard-cases":
        _step_hard_cases(args)
    elif args.command == "distill":
        _step_distill(args)
    elif args.command == "run":
        _step_preprocess()
        _step_split()
        _step_features()
        _step_train(["all"])
        _step_evaluate()
        _step_calibration()
    elapsed = time.time() - start
    log.info("Done in %.1f s.", elapsed)


if __name__ == "__main__":
    main()