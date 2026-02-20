"""Pricing tools: calculate price ranges and check approval requirements."""


def calculate_price_range(
    avg_views: int,
    qty: int,
    target_cpm_brl: float,
    benchmarks: dict | None = None,
) -> dict:
    """Calculate floor/target/ceiling price range.

    Uses effective_cpm = max(target_cpm, benchmark_avg_cpm) when benchmarks available.
    Floor = 70% of target, Ceiling = 130% of target.
    """
    effective_cpm = target_cpm_brl

    if benchmarks and benchmarks.get("avg_cpm"):
        effective_cpm = max(target_cpm_brl, benchmarks["avg_cpm"])

    base_price = (avg_views * qty * effective_cpm) / 1000

    return {
        "floor": round(base_price * 0.70, 2),
        "target": round(base_price, 2),
        "ceiling": round(base_price * 1.30, 2),
    }


def approval_required(
    proposed_brl: float,
    price_range: dict,
    benchmarks: dict | None = None,
) -> bool:
    """Check if the proposed price requires human approval.

    Returns True if:
    - proposed price is below floor or above ceiling
    - no benchmarks available (count == 0)
    """
    if not benchmarks or benchmarks.get("count", 0) == 0:
        return True

    if proposed_brl < price_range.get("floor", 0):
        return True

    if proposed_brl > price_range.get("ceiling", float("inf")):
        return True

    return False
