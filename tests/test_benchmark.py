"""Unit tests for BenchmarkRunner: add_result, get_best_per_metric, CSV output."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.benchmark import BenchmarkResult, BenchmarkRunner
from src.evaluation.metrics import EmbeddingMetrics


@pytest.fixture
def sample_metrics_a():
    """Create sample EmbeddingMetrics for model A (stronger model)."""
    return EmbeddingMetrics(
        knn_accuracy={1: 0.92, 3: 0.90, 5: 0.88, 10: 0.85},
        recall_at_k={1: 0.85, 5: 0.95, 10: 0.98},
        precision_at_k={1: 0.85, 5: 0.80, 10: 0.75},
        silhouette_score=0.65,
        nmi=0.78,
        mean_average_precision=0.82,
        inter_class_distance=15.0,
        intra_class_distance=2.0,
        separability_index=7.5,
    )


@pytest.fixture
def sample_metrics_b():
    """Create sample EmbeddingMetrics for model B (weaker model)."""
    return EmbeddingMetrics(
        knn_accuracy={1: 0.80, 3: 0.78, 5: 0.75, 10: 0.72},
        recall_at_k={1: 0.70, 5: 0.85, 10: 0.92},
        precision_at_k={1: 0.70, 5: 0.65, 10: 0.60},
        silhouette_score=0.45,
        nmi=0.60,
        mean_average_precision=0.68,
        inter_class_distance=10.0,
        intra_class_distance=3.5,
        separability_index=2.86,
    )


@pytest.fixture
def sample_metrics_c():
    """Create sample EmbeddingMetrics for model C (mixed performance)."""
    return EmbeddingMetrics(
        knn_accuracy={1: 0.85, 3: 0.84, 5: 0.82, 10: 0.80},
        recall_at_k={1: 0.80, 5: 0.92, 10: 0.96},
        precision_at_k={1: 0.80, 5: 0.75, 10: 0.70},
        silhouette_score=0.70,  # Best silhouette
        nmi=0.72,
        mean_average_precision=0.75,
        inter_class_distance=12.0,
        intra_class_distance=1.5,  # Best (lowest) intra-class
        separability_index=8.0,  # Best separability
    )


@pytest.fixture
def runner_with_results(sample_metrics_a, sample_metrics_b, sample_metrics_c):
    """Create a BenchmarkRunner with 3 model results added."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(results_dir=Path(tmpdir))
        runner.add_result("resnet50", "finetune", sample_metrics_a)
        runner.add_result("efficientnet_b0", "triplet", sample_metrics_b)
        runner.add_result("vit_b16", "supcon", sample_metrics_c)
        yield runner


class TestBenchmarkRunnerInit:
    """Tests for BenchmarkRunner initialization."""

    def test_creates_results_dir(self):
        """Should create the results directory on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "new_dir" / "benchmark"
            runner = BenchmarkRunner(results_dir=target)
            assert target.exists()

    def test_empty_results_on_init(self):
        """Should start with no results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            assert len(runner.results) == 0


class TestAddResult:
    """Tests for adding model results."""

    def test_add_single_result(self, sample_metrics_a):
        """Should add a single result correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            runner.add_result("resnet50", "finetune", sample_metrics_a)

            assert len(runner.results) == 1
            assert runner.results[0].model_name == "resnet50"
            assert runner.results[0].training_mode == "finetune"
            assert runner.results[0].metrics is sample_metrics_a

    def test_add_multiple_results(self, sample_metrics_a, sample_metrics_b):
        """Should accumulate multiple results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            runner.add_result("resnet50", "finetune", sample_metrics_a)
            runner.add_result("efficientnet_b0", "triplet", sample_metrics_b)

            assert len(runner.results) == 2

    def test_result_is_benchmark_result_dataclass(self, sample_metrics_a):
        """Results should be BenchmarkResult instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            runner.add_result("resnet50", "finetune", sample_metrics_a)

            result = runner.results[0]
            assert isinstance(result, BenchmarkResult)


class TestGetBestPerMetric:
    """Tests for identifying the best model per metric."""

    def test_empty_results_returns_empty(self):
        """Should return empty dict when no results exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            assert runner.get_best_per_metric() == {}

    def test_single_model_is_best_everywhere(self, sample_metrics_a):
        """With one model, it should be best for all metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            runner.add_result("resnet50", "finetune", sample_metrics_a)

            best = runner.get_best_per_metric()
            for metric_name, result in best.items():
                assert result.model_name == "resnet50"

    def test_best_higher_is_better(self, runner_with_results):
        """For higher-is-better metrics, picks the model with highest value."""
        best = runner_with_results.get_best_per_metric()

        # Model A has best knn_accuracy_k1 (0.92)
        assert best["knn_accuracy_k1"].model_name == "resnet50"
        assert best["knn_accuracy_k1"].training_mode == "finetune"

        # Model A has best MAP (0.82)
        assert best["mean_average_precision"].model_name == "resnet50"

    def test_best_lower_is_better(self, runner_with_results):
        """For intra_class_distance, lower is better."""
        best = runner_with_results.get_best_per_metric()

        # Model C has lowest intra_class_distance (1.5)
        assert best["intra_class_distance"].model_name == "vit_b16"
        assert best["intra_class_distance"].training_mode == "supcon"

    def test_best_silhouette(self, runner_with_results):
        """Model C has best silhouette (0.70)."""
        best = runner_with_results.get_best_per_metric()
        assert best["silhouette_score"].model_name == "vit_b16"

    def test_best_separability(self, runner_with_results):
        """Model C has best separability index (8.0)."""
        best = runner_with_results.get_best_per_metric()
        assert best["separability_index"].model_name == "vit_b16"

    def test_all_metrics_have_a_best(self, runner_with_results):
        """Every metric name should have a best model assigned."""
        best = runner_with_results.get_best_per_metric()
        all_metric_names = runner_with_results._get_all_metric_names()
        for name in all_metric_names:
            assert name in best


class TestGenerateCSV:
    """Tests for CSV generation."""

    def test_csv_created(self, runner_with_results):
        """Should create a CSV file at the specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "benchmark.csv"
            runner_with_results.generate_csv(csv_path)
            assert csv_path.exists()

    def test_csv_has_correct_rows(self, runner_with_results):
        """CSV should have one header row plus one row per model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "benchmark.csv"
            runner_with_results.generate_csv(csv_path)

            with open(csv_path, "r") as f:
                import csv as csv_mod
                reader = list(csv_mod.DictReader(f))

            assert len(reader) == 3  # 3 models

    def test_csv_has_model_columns(self, runner_with_results):
        """CSV should have model_name and training_mode columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "benchmark.csv"
            runner_with_results.generate_csv(csv_path)

            with open(csv_path, "r") as f:
                import csv as csv_mod
                reader = csv_mod.DictReader(f)
                fieldnames = reader.fieldnames

            assert "model_name" in fieldnames
            assert "training_mode" in fieldnames

    def test_csv_has_metric_columns(self, runner_with_results):
        """CSV should have columns for each metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "benchmark.csv"
            runner_with_results.generate_csv(csv_path)

            with open(csv_path, "r") as f:
                import csv as csv_mod
                reader = csv_mod.DictReader(f)
                fieldnames = reader.fieldnames

            assert "knn_accuracy_k1" in fieldnames
            assert "mean_average_precision" in fieldnames
            assert "silhouette_score" in fieldnames

    def test_csv_values_match(self, sample_metrics_a):
        """CSV values should match the original metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            runner.add_result("resnet50", "finetune", sample_metrics_a)

            csv_path = Path(tmpdir) / "benchmark.csv"
            runner.generate_csv(csv_path)

            with open(csv_path, "r") as f:
                import csv as csv_mod
                rows = list(csv_mod.DictReader(f))

            row = rows[0]
            assert row["model_name"] == "resnet50"
            assert row["training_mode"] == "finetune"
            assert abs(float(row["knn_accuracy_k1"]) - 0.92) < 1e-5
            assert abs(float(row["mean_average_precision"]) - 0.82) < 1e-5

    def test_csv_empty_results_no_error(self):
        """Should handle empty results gracefully (no crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            csv_path = Path(tmpdir) / "benchmark.csv"
            runner.generate_csv(csv_path)
            # File might not be created or be empty - just shouldn't crash


class TestGenerateLatex:
    """Tests for LaTeX table generation."""

    def test_latex_file_created(self, runner_with_results):
        """Should create a .tex file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "benchmark.tex"
            runner_with_results.generate_latex(tex_path)
            assert tex_path.exists()

    def test_latex_contains_textbf(self, runner_with_results):
        """Best values should be highlighted with \\textbf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "benchmark.tex"
            runner_with_results.generate_latex(tex_path)

            content = tex_path.read_text()
            assert r"\textbf{" in content

    def test_latex_contains_table_structure(self, runner_with_results):
        """Should contain LaTeX table structure elements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "benchmark.tex"
            runner_with_results.generate_latex(tex_path)

            content = tex_path.read_text()
            assert r"\begin{table}" in content
            assert r"\end{table}" in content
            assert r"\begin{tabular}" in content
            assert r"\toprule" in content
            assert r"\bottomrule" in content


class TestGenerateBarChart:
    """Tests for bar chart generation."""

    def test_bar_chart_created(self, runner_with_results):
        """Should create a PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "benchmark.png"
            runner_with_results.generate_bar_chart(png_path)
            assert png_path.exists()

    def test_bar_chart_file_size(self, runner_with_results):
        """PNG file should have non-trivial size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "benchmark.png"
            runner_with_results.generate_bar_chart(png_path)
            assert png_path.stat().st_size > 1000  # At least 1KB


class TestSignificanceTest:
    """Tests for paired bootstrap significance test."""

    def test_identical_models_high_p(self, sample_metrics_a):
        """Comparing a model with itself should give high p-value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            runner.add_result("resnet50", "finetune", sample_metrics_a)
            runner.add_result("resnet50", "finetune_copy", sample_metrics_a)

            p = runner.run_significance_test(
                "resnet50/finetune",
                "resnet50/finetune_copy",
                "knn_accuracy_k1",
                n_bootstrap=500,
            )
            # Identical scores should not be significant
            assert p > 0.05

    def test_different_models_returns_p_value(
        self, sample_metrics_a, sample_metrics_b
    ):
        """Should return a valid p-value in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            runner.add_result("resnet50", "finetune", sample_metrics_a)
            runner.add_result("efficientnet_b0", "triplet", sample_metrics_b)

            p = runner.run_significance_test(
                "resnet50/finetune",
                "efficientnet_b0/triplet",
                "knn_accuracy_k1",
                n_bootstrap=500,
            )
            assert 0.0 <= p <= 1.0

    def test_nonexistent_model_raises(self, sample_metrics_a):
        """Should raise ValueError for unknown model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            runner.add_result("resnet50", "finetune", sample_metrics_a)

            with pytest.raises(ValueError, match="not found"):
                runner.run_significance_test(
                    "resnet50/finetune",
                    "nonexistent/model",
                    "knn_accuracy_k1",
                )

    def test_nonexistent_metric_raises(
        self, sample_metrics_a, sample_metrics_b
    ):
        """Should raise ValueError for unknown metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=Path(tmpdir))
            runner.add_result("resnet50", "finetune", sample_metrics_a)
            runner.add_result("efficientnet_b0", "triplet", sample_metrics_b)

            with pytest.raises(ValueError, match="not found"):
                runner.run_significance_test(
                    "resnet50/finetune",
                    "efficientnet_b0/triplet",
                    "nonexistent_metric",
                )
