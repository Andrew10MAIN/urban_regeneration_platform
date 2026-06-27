import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Reads from env (Docker) or falls back to local dev defaults
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://urban_user:urban_password@localhost:5433/urban_db"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)


def get_engine():
    return engine


def get_connection():
    """Context manager — use with `with get_connection() as conn:`"""
    return engine.connect()


def query(sql: str, **params):
    """Quick helper for raw SQL queries, returns list of dicts."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        return [dict(row._mapping) for row in result]
