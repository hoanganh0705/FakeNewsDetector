# Evaluation module
from .metrics import (
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    save_metrics,
)
from .calibration_analysis import (
    analyze_calibration,
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    plot_calibration_curves,
)
from .post_hoc_calibration import (  # noqa: F401
    platt_scaling,
    temperature_scaling,
    isotonic_regression,
    evaluate_recalibration,
    run_post_hoc_calibration,
)
from .hard_cases import (  # noqa: F401
    HardExampleAnnotation,
    analyze_hard_examples,
)

__all__ = [
    'compute_metrics',
    'print_metrics',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'save_metrics',
    'analyze_calibration',
    'expected_calibration_error',
    'maximum_calibration_error',
    'brier_score',
    'plot_calibration_curves',
    # Phase 2
    'platt_scaling',
    'temperature_scaling',
    'isotonic_regression',
    'evaluate_recalibration',
    'run_post_hoc_calibration',
    # Phase 4
    'HardExampleAnnotation',
    'analyze_hard_examples',
]
