"""
Database configuration and session management.
Uses SQLModel with SQLite for simplicity.
"""

import os
from typing import Generator
from sqlmodel import SQLModel, create_engine, Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL from environment or default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./perfume_finder.db")

# Create engine
# check_same_thread=False is needed for SQLite with FastAPI
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)


def create_db_and_tables():
    """
    Create all database tables.
    Should be called on application startup or in seed script.
    """
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get a database session.
    
    Usage:
        @app.get("/users/{user_id}")
        def get_user(user_id: int, session: Session = Depends(get_session)):
            return session.get(User, user_id)
    """
    with Session(engine) as session:
        yield session

