"""Tests for pricing tools."""

import pytest

from app.tools.pricing import approval_required, calculate_price_range


class TestCalculatePriceRange:
    def test_basic_calculation(self):
        result = calculate_price_range(
            avg_views=100_000, qty=1, target_cpm_brl=40.0
        )
        # base = 100_000 * 1 * 40 / 1000 = 4000
        assert result["target"] == 4000.0
        assert result["floor"] == 2800.0  # 70%
        assert result["ceiling"] == 5200.0  # 130%

    def test_with_qty(self):
        result = calculate_price_range(
            avg_views=100_000, qty=3, target_cpm_brl=40.0
        )
        # base = 100_000 * 3 * 40 / 1000 = 12000
        assert result["target"] == 12000.0
        assert result["floor"] == 8400.0
        assert result["ceiling"] == 15600.0

    def test_with_benchmarks_higher_cpm(self):
        benchmarks = {"avg_cpm": 50.0, "count": 5}
        result = calculate_price_range(
            avg_views=100_000, qty=1, target_cpm_brl=40.0, benchmarks=benchmarks
        )
        # effective_cpm = max(40, 50) = 50, base = 100_000 * 1 * 50 / 1000 = 5000
        assert result["target"] == 5000.0
        assert result["floor"] == 3500.0
        assert result["ceiling"] == 6500.0

    def test_with_benchmarks_lower_cpm(self):
        benchmarks = {"avg_cpm": 30.0, "count": 5}
        result = calculate_price_range(
            avg_views=100_000, qty=1, target_cpm_brl=40.0, benchmarks=benchmarks
        )
        # effective_cpm = max(40, 30) = 40
        assert result["target"] == 4000.0

    def test_with_no_benchmark_cpm(self):
        benchmarks = {"avg_cpm": None, "count": 0}
        result = calculate_price_range(
            avg_views=100_000, qty=1, target_cpm_brl=40.0, benchmarks=benchmarks
        )
        assert result["target"] == 4000.0

    def test_small_views(self):
        result = calculate_price_range(
            avg_views=25_000, qty=1, target_cpm_brl=40.0
        )
        # base = 25_000 * 1 * 40 / 1000 = 1000
        assert result["target"] == 1000.0
        assert result["floor"] == 700.0
        assert result["ceiling"] == 1300.0


class TestApprovalRequired:
    def test_no_benchmarks(self):
        assert approval_required(5000, {"floor": 3000, "ceiling": 7000}, None) is True

    def test_empty_benchmarks(self):
        assert (
            approval_required(5000, {"floor": 3000, "ceiling": 7000}, {"count": 0})
            is True
        )

    def test_within_range(self):
        benchmarks = {"count": 5, "avg_cpm": 40.0}
        assert (
            approval_required(5000, {"floor": 3000, "ceiling": 7000}, benchmarks)
            is False
        )

    def test_below_floor(self):
        benchmarks = {"count": 5, "avg_cpm": 40.0}
        assert (
            approval_required(2000, {"floor": 3000, "ceiling": 7000}, benchmarks)
            is True
        )

    def test_above_ceiling(self):
        benchmarks = {"count": 5, "avg_cpm": 40.0}
        assert (
            approval_required(8000, {"floor": 3000, "ceiling": 7000}, benchmarks)
            is True
        )

    def test_at_floor_boundary(self):
        benchmarks = {"count": 5, "avg_cpm": 40.0}
        assert (
            approval_required(3000, {"floor": 3000, "ceiling": 7000}, benchmarks)
            is False
        )
