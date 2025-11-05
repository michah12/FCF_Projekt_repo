"""
Pydantic schemas for API request/response validation.
Separate from database models for clean API contracts.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime


# ========== Perfume Schemas ==========

class PerfumeBase(BaseModel):
    """Base perfume schema with common fields."""
    name: str
    brand: str
    year: Optional[int] = None
    gender: Optional[str] = None
    price: Optional[float] = None
    longevity: Optional[int] = Field(default=3, ge=1, le=5)
    sillage: Optional[int] = Field(default=3, ge=1, le=5)
    accords: Dict[str, float] = Field(default_factory=dict)
    season_vector: Dict[str, float] = Field(default_factory=dict)
    notes_top: List[str] = Field(default_factory=list)
    notes_middle: List[str] = Field(default_factory=list)
    notes_base: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    image_url: Optional[str] = None


class PerfumeResponse(PerfumeBase):
    """Full perfume details for API responses."""
    id: int
    cluster_id: Optional[int] = None
    umap_x: Optional[float] = None
    umap_y: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class PerfumeSearchParams(BaseModel):
    """Search/filter parameters for perfume queries."""
    query: Optional[str] = None
    notes: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    seasons: Optional[List[str]] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    gender: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


# ========== User Schemas ==========

class UserCreate(BaseModel):
    """Request to create a new user."""
    username: str


class UserResponse(BaseModel):
    """User profile response."""
    id: int
    username: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ========== Inventory Schemas ==========

class InventoryAddRequest(BaseModel):
    """Request to add a perfume to user's inventory."""
    perfume_id: int
    rating: Optional[int] = Field(default=None, ge=1, le=5)


class InventoryItemResponse(BaseModel):
    """Inventory item with perfume details."""
    id: int
    perfume_id: int
    rating: Optional[int] = None
    added_at: datetime
    perfume: PerfumeResponse
    
    class Config:
        from_attributes = True


# ========== Preferences Schemas ==========

class PreferencesCreate(BaseModel):
    """User preferences from onboarding questionnaire."""
    user_id: int
    accord_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="e.g., {'citrus': 0.9, 'floral': 0.7}"
    )
    disliked_notes: List[str] = Field(default_factory=list)
    season_vector: Dict[str, float] = Field(
        default_factory=dict,
        description="e.g., {'spring': 0.8, 'summer': 1.0}"
    )
    budget_min: float = Field(default=0, ge=0)
    budget_max: float = Field(default=500, ge=0)
    longevity_pref: int = Field(default=3, ge=1, le=5)
    sillage_pref: int = Field(default=3, ge=1, le=5)


class PreferencesResponse(PreferencesCreate):
    """User preferences with metadata."""
    id: int
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ========== ML Schemas ==========

class SimilarPerfume(BaseModel):
    """A similar perfume with explanation."""
    perfume: PerfumeResponse
    similarity: float = Field(ge=0.0, le=1.0)
    explanation: str


class RecommendationRequest(BaseModel):
    """Request for personalized recommendations."""
    user_id: int
    top_n: int = Field(default=10, ge=1, le=50)
    exclude_owned: bool = True
    filter_by_budget: bool = True
    filter_by_season: bool = False


class RecommendationResponse(BaseModel):
    """Recommendation result with explanations."""
    recommendations: List[SimilarPerfume]
    based_on_inventory_count: int
    based_on_preferences: bool


class ClusterInfo(BaseModel):
    """Summary of a perfume cluster."""
    cluster_id: int
    size: int
    top_accords: Dict[str, float]
    top_notes: List[str]
    avg_price: Optional[float]
    representative_perfumes: List[str]  # Names


class ClusterResponse(BaseModel):
    """All cluster information."""
    clusters: List[ClusterInfo]
    total_perfumes: int


class MapPoint(BaseModel):
    """2D point for similarity map visualization."""
    perfume_id: int
    name: str
    brand: str
    cluster_id: Optional[int]
    x: float
    y: float
    price: Optional[float]
    accords: Dict[str, float]


class MapResponse(BaseModel):
    """Full similarity map data."""
    points: List[MapPoint]
    clusters: List[ClusterInfo]


# ========== Price Schemas ==========

class PriceOfferResponse(BaseModel):
    """Price offer from a retailer."""
    id: int
    shop_name: str
    url: str
    price: float
    currency: str
    last_seen_at: datetime
    
    class Config:
        from_attributes = True

