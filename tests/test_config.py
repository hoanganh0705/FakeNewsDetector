"""Tests for the centralised config object — structure & types only."""

import pytest

from config import cfg, Config, Paths


class TestConfig:
    """Smoke-tests for the centralised config object."""

    def test_random_state_is_int(self):
        assert isinstance(cfg.RANDOM_STATE, int)

    def test_split_ratios_sum_to_one(self):
        total = cfg.DATA.train_ratio + cfg.DATA.val_ratio + cfg.DATA.test_ratio
        assert total == pytest.approx(1.0)

    def test_paths_are_strings(self):
        assert isinstance(cfg.PATHS.raw_data, str)
        assert isinstance(cfg.PATHS.splits_dir, str)

    def test_bilstm_embedding_dim_is_positive_int(self):
        assert isinstance(cfg.BILSTM.embedding_dim, int)
        assert cfg.BILSTM.embedding_dim > 0

    def test_phobert_model_name_is_nonempty_string(self):
        assert isinstance(cfg.PHOBERT.model_name, str)
        assert len(cfg.PHOBERT.model_name) > 0

    def test_label_smoothing_fields_exist(self):
        assert hasattr(cfg.PHOBERT, "label_smoothing")
        assert hasattr(cfg.BILSTM, "label_smoothing")

    def test_fasttext_path_field_exists(self):
        assert hasattr(cfg.BILSTM, "fasttext_path")

    def test_all_path_fields_are_strings(self):
        """Every field in Paths should be a non-empty string."""
        import dataclasses

        for f in dataclasses.fields(cfg.PATHS):
            value = getattr(cfg.PATHS, f.name)
            assert isinstance(value, str), f"PATHS.{f.name} is not a string"
            assert len(value) > 0, f"PATHS.{f.name} is empty"

    def test_learning_rates_are_positive_floats(self):
        """Model learning rates must be positive floats."""
        for section in (cfg.BILSTM, cfg.PHOBERT):
            assert isinstance(section.learning_rate, float)
            assert section.learning_rate > 0

    def test_fresh_config_equal_to_singleton(self):
        """A freshly constructed Config should match the module-level singleton."""
        fresh = Config()
        assert fresh.RANDOM_STATE == cfg.RANDOM_STATE
        assert fresh.DATA.train_ratio == cfg.DATA.train_ratio
