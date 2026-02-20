"""Retrieval tool: query historical deals from SQLite."""

from statistics import median

from sqlalchemy import func

from app.db.models import Deal
from app.db.session import SessionLocal


def retrieve_benchmarks(
    platform: str,
    deliverable_type: str,
    avg_views: int,
    niche: str | None = None,
    k: int = 5,
    session=None,
) -> dict:
    """Query deals filtered by platform + deliverable_type, sorted by view proximity.

    Returns stats (count, avg_cpm, median_price, min_price, max_price) + top-k samples.
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()

    try:
        query = session.query(Deal).filter(
            Deal.platform == platform.lower(),
            Deal.deliverable_type == deliverable_type.lower(),
        )

        if niche:
            niche_query = query.filter(Deal.niche == niche.lower())
            if niche_query.count() > 0:
                query = niche_query

        deals = query.all()

        if not deals:
            return {
                "count": 0,
                "avg_cpm": None,
                "median_price": None,
                "min_price": None,
                "max_price": None,
                "samples": [],
            }

        deals_sorted = sorted(deals, key=lambda d: abs(d.avg_views - avg_views))
        top_k = deals_sorted[:k]

        prices = [d.final_price_brl for d in deals]
        cpms = [d.cpm_brl for d in deals]

        return {
            "count": len(deals),
            "avg_cpm": round(sum(cpms) / len(cpms), 2),
            "median_price": round(median(prices), 2),
            "min_price": min(prices),
            "max_price": max(prices),
            "samples": [
                {
                    "influencer": d.influencer_name,
                    "avg_views": d.avg_views,
                    "price_brl": d.final_price_brl,
                    "cpm_brl": d.cpm_brl,
                    "niche": d.niche,
                }
                for d in top_k
            ],
        }
    finally:
        if own_session:
            session.close()
