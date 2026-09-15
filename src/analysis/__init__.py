"""Analysis module for statistical tests and paper generation."""

from src.analysis.statistical_tests import (
    mcnemar_test,
    holm_bonferroni_correction,
    bootstrap_confidence_interval,
    cohens_d,
    load_predictions,
    run_statistical_analysis,
)
from src.analysis.generate_paper_tables import (
    table1_dataset_statistics,
    table2_model_comparison,
    table3_per_class_metrics,
    table4_hyperparameters,
    table5_training_time,
)
from src.analysis.generate_attribution_figures import (
    load_all_attributions,
    find_best_record,
    create_phobert_attribution_figure,
    create_method_agreement_figure,
    create_cross_model_agreement_figure,
)

__all__ = [
    # statistical_tests
    "mcnemar_test",
    "holm_bonferroni_correction",
    "bootstrap_confidence_interval",
    "cohens_d",
    "load_predictions",
    "run_statistical_analysis",
    # generate_paper_tables
    "table1_dataset_statistics",
    "table2_model_comparison",
    "table3_per_class_metrics",
    "table4_hyperparameters",
    "table5_training_time",
    # generate_attribution_figures
    "load_all_attributions",
    "find_best_record",
    "create_phobert_attribution_figure",
    "create_method_agreement_figure",
    "create_cross_model_agreement_figure",
]
