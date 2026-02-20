"""Engine + session factory."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/negotiator.db")


def _ensure_data_dir() -> None:
    """Create data/ directory if it doesn't exist (for SQLite)."""
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def get_engine():
    _ensure_data_dir()
    return create_engine(DATABASE_URL, echo=False)


def init_db(engine=None):
    """Create all tables."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


engine = get_engine()
SessionLocal = sessionmaker(bind=engine)
