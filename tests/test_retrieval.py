"""Tests for retrieval tool."""

import pytest

from app.db.models import Base, Deal
from app.db.session import get_engine
from app.tools.retrieval import retrieve_benchmarks
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


@pytest.fixture
def db_session():
    """Create in-memory SQLite DB with test data."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    deals = [
        Deal(
            influencer_name="Test1", platform="instagram", niche="fitness",
            deliverable_type="reel", qty=1, avg_views=100_000,
            final_price_brl=4000, cpm_brl=40.0,
        ),
        Deal(
            influencer_name="Test2", platform="instagram", niche="fitness",
            deliverable_type="reel", qty=1, avg_views=80_000,
            final_price_brl=3200, cpm_brl=40.0,
        ),
        Deal(
            influencer_name="Test3", platform="tiktok", niche="moda",
            deliverable_type="video", qty=1, avg_views=150_000,
            final_price_brl=6000, cpm_brl=40.0,
        ),
    ]
    session.add_all(deals)
    session.commit()
    yield session
    session.close()


class TestRetrieveBenchmarks:
    def test_matching_deals(self, db_session):
        result = retrieve_benchmarks(
            platform="instagram",
            deliverable_type="reel",
            avg_views=90_000,
            session=db_session,
        )
        assert result["count"] == 2
        assert result["avg_cpm"] == 40.0
        assert len(result["samples"]) == 2

    def test_no_matching_deals(self, db_session):
        result = retrieve_benchmarks(
            platform="youtube",
            deliverable_type="reel",
            avg_views=90_000,
            session=db_session,
        )
        assert result["count"] == 0
        assert result["avg_cpm"] is None
        assert result["samples"] == []

    def test_sorted_by_view_proximity(self, db_session):
        result = retrieve_benchmarks(
            platform="instagram",
            deliverable_type="reel",
            avg_views=85_000,
            session=db_session,
        )
        # 80_000 is closer to 85_000 than 100_000
        assert result["samples"][0]["avg_views"] == 80_000
        assert result["samples"][1]["avg_views"] == 100_000
