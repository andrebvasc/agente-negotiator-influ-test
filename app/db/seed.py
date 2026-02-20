"""Seed database with 20 realistic deals."""

from datetime import datetime, timedelta, timezone

from app.db.models import Deal
from app.db.session import SessionLocal, init_db

SEED_DEALS = [
    ("Ana Fitness", "instagram", "fitness", "reel", 1, 80_000, 3200.00),
    ("Bruno Shape", "instagram", "fitness", "post", 1, 120_000, 4800.00),
    ("Carla Yoga", "tiktok", "fitness", "video", 1, 50_000, 1500.00),
    ("Duda Fashion", "instagram", "moda", "reel", 2, 150_000, 9000.00),
    ("Edu Style", "tiktok", "moda", "video", 1, 90_000, 3600.00),
    ("Fabi Glam", "instagram", "beleza", "story", 3, 60_000, 2700.00),
    ("Gabi Makeup", "instagram", "beleza", "reel", 1, 200_000, 8000.00),
    ("Hugo Tech", "youtube", "tech", "video", 1, 100_000, 6000.00),
    ("Igor Reviews", "youtube", "tech", "video", 1, 250_000, 15000.00),
    ("Julia Comedy", "tiktok", "humor", "video", 2, 300_000, 12000.00),
    ("Kaio Gamer", "youtube", "games", "video", 1, 180_000, 9000.00),
    ("Lara Chef", "instagram", "culinaria", "reel", 1, 70_000, 2800.00),
    ("Manu Viaja", "instagram", "viagem", "post", 2, 95_000, 5700.00),
    ("Neto Finance", "youtube", "financas", "video", 1, 130_000, 7800.00),
    ("Oli Fitness", "tiktok", "fitness", "video", 1, 45_000, 1350.00),
    ("Paula Moda", "instagram", "moda", "story", 4, 110_000, 6600.00),
    ("Quinn Beauty", "tiktok", "beleza", "video", 1, 75_000, 2250.00),
    ("Rafa Humor", "instagram", "humor", "reel", 1, 160_000, 6400.00),
    ("Sofia Cook", "tiktok", "culinaria", "video", 1, 55_000, 1650.00),
    ("Tiago Travel", "youtube", "viagem", "video", 1, 220_000, 13200.00),
]


def seed() -> int:
    """Insert seed deals, return count inserted."""
    init_db()
    session = SessionLocal()
    try:
        existing = session.query(Deal).count()
        if existing > 0:
            return 0

        now = datetime.now(timezone.utc)
        for i, (name, plat, niche, dtype, qty, views, price) in enumerate(SEED_DEALS):
            cpm = (price / (views * qty)) * 1000
            deal = Deal(
                influencer_name=name,
                platform=plat,
                niche=niche,
                deliverable_type=dtype,
                qty=qty,
                avg_views=views,
                final_price_brl=price,
                cpm_brl=round(cpm, 2),
                closed_at=now - timedelta(days=30 - i),
            )
            session.add(deal)

        session.commit()
        return len(SEED_DEALS)
    finally:
        session.close()
