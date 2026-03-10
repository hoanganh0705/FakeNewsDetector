"""
Smoke tests for modules that previously had zero test coverage (§7.3).

Each test imports the target module and calls its ``main()`` with mocked I/O
so that no real data, trained models, or filesystem writes are needed.
"""

from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_METRICS = {
    'Logistic Regression': {
        'test': {
            'accuracy': 0.95, 'precision_macro': 0.94, 'recall_macro': 0.93,
            'f1_macro': 0.94, 'roc_auc': 0.98,
            'confusion_matrix': [[90, 5], [3, 102]],
            'precision_per_class': [0.94, 0.96],
            'recall_per_class': [0.95, 0.95],
            'f1_per_class': [0.94, 0.96],
            'classification_report': '',
        },
        'training_time': 1.2,
    },
}


def _fake_test_df() -> pd.DataFrame:
    """Minimal DataFrame that mimics ``load_test_data_with_predictions``."""
    n = 6
    df = pd.DataFrame({
        'id': range(n),
        'text': ['foo bar'] * n,
        'label': [0, 1, 0, 1, 0, 1],
    })
    for model in ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']:
        df[f'{model}_pred'] = df['label']
        df[f'{model}_prob'] = 0.9
        df[f'{model}_correct'] = 1
    return df


# ---------------------------------------------------------------------------
# §7.3-a  src/analysis/explainability.py
# ---------------------------------------------------------------------------

class TestExplainabilitySmoke:
    @patch('src.analysis.explainability.json.dump')
    @patch('builtins.open', mock_open())
    @patch('src.analysis.explainability.os.makedirs')
    @patch('src.analysis.explainability.create_error_taxonomy_plot')
    @patch('src.analysis.explainability.create_feature_importance_plot')
    @patch('src.analysis.explainability.analyze_error_categories', return_value={})
    @patch('src.analysis.explainability.analyze_lr_feature_importance', return_value={})
    def test_main_returns_dict(self, *mocks):
        from src.analysis.explainability import main
        result = main()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# §7.3-b  src/analysis/generate_paper_figures.py
# ---------------------------------------------------------------------------

class TestGeneratePaperFiguresSmoke:
    @patch('src.analysis.generate_paper_figures.os.listdir', return_value=[])
    @patch('src.analysis.generate_paper_figures.os.makedirs')
    @patch('src.analysis.generate_paper_figures.figure6_model_paradigm_comparison')
    @patch('src.analysis.generate_paper_figures.figure5_per_class_performance')
    @patch('src.analysis.generate_paper_figures.figure4_precision_recall_curves')
    @patch('src.analysis.generate_paper_figures.figure3_roc_curves')
    @patch('src.analysis.generate_paper_figures.figure2_confusion_matrices')
    @patch('src.analysis.generate_paper_figures.figure1_model_comparison_bar')
    @patch('src.analysis.generate_paper_figures.figure0_label_distribution')
    @patch('src.analysis.generate_paper_figures.load_all_data',
           return_value=(_FAKE_METRICS, {}))
    def test_main_runs(self, *mocks):
        from src.analysis.generate_paper_figures import main
        main()  # returns None


# ---------------------------------------------------------------------------
# §7.3-c  src/analysis/generate_paper_tables.py
# ---------------------------------------------------------------------------

class TestGeneratePaperTablesSmoke:
    @patch('src.analysis.generate_paper_tables.os.makedirs')
    @patch('src.analysis.generate_paper_tables.table5_training_time')
    @patch('src.analysis.generate_paper_tables.table4_hyperparameters')
    @patch('src.analysis.generate_paper_tables.table3_per_class_metrics')
    @patch('src.analysis.generate_paper_tables.table2_model_comparison')
    @patch('src.analysis.generate_paper_tables.table1_dataset_statistics')
    @patch('src.analysis.generate_paper_tables._fetch_all_metrics',
           return_value=_FAKE_METRICS)
    def test_main_runs(self, *mocks):
        from src.analysis.generate_paper_tables import main
        main()


# ---------------------------------------------------------------------------
# §7.3-d  src/evaluation/error_analysis.py
# ---------------------------------------------------------------------------

class TestErrorAnalysisSmoke:
    @patch('src.evaluation.error_analysis.track_per_id_confidence',
           return_value=pd.DataFrame())
    @patch('src.evaluation.error_analysis.os.path.exists', return_value=False)
    @patch('src.evaluation.error_analysis.os.makedirs')
    @patch('src.evaluation.error_analysis.plot_error_analysis')
    @patch('src.evaluation.error_analysis.analyze_error_patterns', return_value=None)
    @patch('src.evaluation.error_analysis.find_hard_examples',
           return_value=pd.DataFrame(columns=['id', 'text', 'label']))
    @patch('src.evaluation.error_analysis.load_test_data_with_predictions')
    def test_main_returns_dataframe(self, mock_load, *mocks):
        mock_load.return_value = _fake_test_df()
        from src.evaluation.error_analysis import main
        result = main()
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# §7.3-e  src/evaluation/evaluate_all.py
# ---------------------------------------------------------------------------

class TestEvaluateAllSmoke:
    @patch('src.evaluation.evaluate_all.json.dump')
    @patch('builtins.open', mock_open())
    @patch('src.evaluation.evaluate_all.plot_training_history')
    @patch('src.evaluation.evaluate_all.plot_roc_curves_comparison')
    @patch('src.evaluation.evaluate_all.plot_confusion_matrices_grid')
    @patch('src.evaluation.evaluate_all.plot_model_comparison')
    @patch('src.evaluation.evaluate_all.generate_latex_table')
    @patch('src.evaluation.evaluate_all.save_predictions_for_analysis')
    @patch('src.evaluation.evaluate_all.os.makedirs')
    @patch('src.evaluation.evaluate_all.load_all_metrics',
           return_value=_FAKE_METRICS)
    def test_main_returns_tuple(self, *mocks):
        from src.evaluation.evaluate_all import main
        result = main()
        assert result is not None
        comparison_df, metrics = result
        assert isinstance(comparison_df, pd.DataFrame)
        assert isinstance(metrics, dict)

    @patch('src.evaluation.evaluate_all.os.makedirs')
    @patch('src.evaluation.evaluate_all.load_all_metrics', return_value={})
    def test_main_returns_none_when_no_metrics(self, *mocks):
        from src.evaluation.evaluate_all import main
        result = main()
        assert result is None


# ---------------------------------------------------------------------------
# §7.3-f  src/evaluation/cross_validation.py
# ---------------------------------------------------------------------------

class TestCrossValidationSmoke:
    _CV_RESULT = {
        metric: {
            'mean': 0.95, 'std': 0.01, 'min': 0.93, 'max': 0.97,
            'ci_lower': 0.93, 'ci_upper': 0.97, 'all_scores': [0.95],
        }
        for metric in ['accuracy', 'f1_macro', 'precision_macro',
                        'recall_macro', 'roc_auc']
    }

    @patch('src.evaluation.cross_validation.json.dump')
    @patch('builtins.open', mock_open())
    @patch('src.evaluation.cross_validation.os.makedirs')
    @patch('src.evaluation.cross_validation.run_cross_validation')
    @patch('src.evaluation.cross_validation.joblib.load')
    @patch('src.evaluation.cross_validation.os.path.exists', return_value=True)
    def test_main_returns_dict(self, mock_exists, mock_load, mock_run_cv,
                                *mocks):
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.eye(4))
        mock_load.return_value = {
            'X_train': X, 'X_val': X,
            'y_train': np.array([0, 1, 0, 1]),
            'y_val': np.array([0, 1, 0, 1]),
        }
        mock_run_cv.return_value = self._CV_RESULT

        from src.evaluation.cross_validation import main
        result = main()
        assert isinstance(result, dict)

    @patch('src.evaluation.cross_validation.os.path.exists', return_value=False)
    def test_main_returns_none_when_no_features(self, *mocks):
        from src.evaluation.cross_validation import main
        result = main()
        assert result is None


# ---------------------------------------------------------------------------
# §7.3-g  src/evaluation/ablation_study.py
# ---------------------------------------------------------------------------

class TestAblationStudySmoke:
    _ROW_VOCAB = [{'vocab_size': 10000, 'accuracy': 0.95, 'f1_macro': 0.95, 'time_s': 1.0}]
    _ROW_NGRAM = [{'ngram_range': '(1,2)', 'label': 'Uni+Bi', 'accuracy': 0.95, 'f1_macro': 0.95, 'time_s': 1.0}]
    _ROW_SEG = [
        {'config': 'Có tách từ', 'accuracy': 0.95, 'f1_macro': 0.95},
        {'config': 'Không tách từ', 'accuracy': 0.90, 'f1_macro': 0.90},
    ]
    _ROW_REG = [{'C': 1, 'accuracy': 0.95, 'f1_macro': 0.95}]
    _ROW_TF = [
        {'config': 'sublinear', 'sublinear_tf': True, 'accuracy': 0.95, 'f1_macro': 0.95},
        {'config': 'standard', 'sublinear_tf': False, 'accuracy': 0.93, 'f1_macro': 0.93},
    ]

    @patch('src.evaluation.ablation_study.json.dump')
    @patch('builtins.open', mock_open())
    @patch('src.evaluation.ablation_study.os.makedirs')
    @patch('src.evaluation.ablation_study.ablation_sublinear_tf')
    @patch('src.evaluation.ablation_study.ablation_lr_regularization')
    @patch('src.evaluation.ablation_study.ablation_word_segmentation')
    @patch('src.evaluation.ablation_study.ablation_ngram_range')
    @patch('src.evaluation.ablation_study.ablation_tfidf_vocab_size')
    @patch('src.evaluation.ablation_study.load_text_data')
    def test_main_returns_dict(self, mock_load, mock_vocab, mock_ngram,
                                mock_seg, mock_reg, mock_tf, *mocks):
        mock_load.return_value = (
            pd.Series(['a', 'b']), pd.Series(['c', 'd']),
            np.array([0, 1]), np.array([0, 1]),
        )
        mock_vocab.return_value = self._ROW_VOCAB
        mock_ngram.return_value = self._ROW_NGRAM
        mock_seg.return_value = self._ROW_SEG
        mock_reg.return_value = self._ROW_REG
        mock_tf.return_value = self._ROW_TF

        from src.evaluation.ablation_study import main
        result = main()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# §7.3-h  src/training/train_all.py
# ---------------------------------------------------------------------------

class TestTrainAllSmoke:
    @patch('src.training.train_all.importlib.import_module')
    def test_main_returns_dict(self, mock_import):
        fake_mod = MagicMock()
        fake_mod.main.return_value = (
            MagicMock(),  # trainer
            {'accuracy': 0.95, 'f1_macro': 0.94},
        )
        mock_import.return_value = fake_mod

        from src.training.train_all import main
        results = main()
        assert isinstance(results, dict)
        assert len(results) == 4
        for info in results.values():
            assert info['status'] == 'success'

    @patch('src.training.train_all.importlib.import_module')
    def test_main_handles_failure(self, mock_import):
        mock_import.side_effect = RuntimeError('boom')

        from src.training.train_all import main
        results = main()
        assert isinstance(results, dict)
        for info in results.values():
            assert info['status'] == 'failed'
