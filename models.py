"""
Database models for the Perfume Finder application.
Uses SQLModel for seamless integration between Pydantic and SQLAlchemy.
"""

from typing import Optional, List
from datetime import datetime
from sqlmodel import Field, SQLModel, Relationship, JSON, Column
from sqlalchemy import JSON as SQLA_JSON


class NoteVocab(SQLModel, table=True):
    """Vocabulary of all perfume notes (e.g., 'vanilla', 'bergamot')."""
    
    __tablename__ = "note_vocab"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    

class Perfume(SQLModel, table=True):
    """
    Core perfume entity with metadata, pricing, and computed features.
    
    Attributes:
        accords: Dict of accord weights, e.g., {"citrus": 0.8, "floral": 0.2}
        season_vector: Dict of season suitability, e.g., {"spring": 0.7, "summer": 0.9}
        notes_top/middle/base: List of note names in each layer
        feature_vector: Precomputed ML feature vector (JSON array)
        cluster_id: KMeans cluster assignment
        umap_x, umap_y: 2D coordinates for visualization
    """
    
    __tablename__ = "perfumes"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    brand: str = Field(index=True)
    year: Optional[int] = None
    gender: Optional[str] = None  # "male", "female", "unisex"
    price: Optional[float] = None  # Average/reference price in USD
    longevity: Optional[int] = Field(default=3, ge=1, le=5)  # 1=very weak, 5=eternal
    sillage: Optional[int] = Field(default=3, ge=1, le=5)  # 1=intimate, 5=enormous
    
    # JSON fields for flexible data
    accords: dict = Field(default_factory=dict, sa_column=Column(SQLA_JSON))
    season_vector: dict = Field(default_factory=dict, sa_column=Column(SQLA_JSON))
    notes_top: List[str] = Field(default_factory=list, sa_column=Column(SQLA_JSON))
    notes_middle: List[str] = Field(default_factory=list, sa_column=Column(SQLA_JSON))
    notes_base: List[str] = Field(default_factory=list, sa_column=Column(SQLA_JSON))
    
    # ML computed fields
    feature_vector: Optional[List[float]] = Field(default=None, sa_column=Column(SQLA_JSON))
    cluster_id: Optional[int] = None
    umap_x: Optional[float] = None
    umap_y: Optional[float] = None
    
    # Metadata
    description: Optional[str] = None
    image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PerfumeNote(SQLModel, table=True):
    """
    Many-to-many relationship between perfumes and notes.
    Tracks which notes appear in which perfumes and at what intensity.
    """
    
    __tablename__ = "perfume_note"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    perfume_id: int = Field(foreign_key="perfumes.id", index=True)
    note_id: int = Field(foreign_key="note_vocab.id", index=True)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)  # Importance of this note
    level: str = Field(default="middle")  # "top", "middle", "base"


class User(SQLModel, table=True):
    """User account (mock/simple implementation)."""
    
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserInventory(SQLModel, table=True):
    """
    User's perfume collection with optional ratings.
    Used to build personalized recommendations.
    """
    
    __tablename__ = "user_inventory"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    perfume_id: int = Field(foreign_key="perfumes.id", index=True)
    rating: Optional[int] = Field(default=None, ge=1, le=5)  # 1-5 star rating
    added_at: datetime = Field(default_factory=datetime.utcnow)


class UserPreferences(SQLModel, table=True):
    """
    User taste profile from onboarding questionnaire.
    
    Attributes:
        accord_weights: Liked accords, e.g., {"citrus": 0.9, "woody": 0.6}
        disliked_notes: List of note names to avoid
        season_vector: Preferred seasons, e.g., {"summer": 1.0, "winter": 0.3}
    """
    
    __tablename__ = "user_preferences"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", unique=True, index=True)
    
    # Taste vectors
    accord_weights: dict = Field(default_factory=dict, sa_column=Column(SQLA_JSON))
    disliked_notes: List[str] = Field(default_factory=list, sa_column=Column(SQLA_JSON))
    season_vector: dict = Field(default_factory=dict, sa_column=Column(SQLA_JSON))
    
    # Budget constraints
    budget_min: Optional[float] = Field(default=0)
    budget_max: Optional[float] = Field(default=500)
    
    # Performance preferences
    longevity_pref: Optional[int] = Field(default=3, ge=1, le=5)
    sillage_pref: Optional[int] = Field(default=3, ge=1, le=5)
    
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PriceOffer(SQLModel, table=True):
    """Price comparison from various online retailers."""
    
    __tablename__ = "price_offers"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    perfume_id: int = Field(foreign_key="perfumes.id", index=True)
    shop_name: str
    url: str
    price: float
    currency: str = Field(default="USD")
    last_seen_at: datetime = Field(default_factory=datetime.utcnow)

