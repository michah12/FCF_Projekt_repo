"""
FastAPI application for Perfume Finder.

Provides REST API for:
- Perfume search and retrieval
- User inventory management
- Preference storage
- ML recommendations and clustering
- Price comparison
"""

import os
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from dotenv import load_dotenv
import numpy as np

from db import get_session, create_db_and_tables
from models import (
    Perfume, User, UserInventory, UserPreferences,
    PriceOffer, NoteVocab, PerfumeNote
)
from schemas import (
    PerfumeResponse, PerfumeSearchParams,
    UserCreate, UserResponse,
    InventoryAddRequest, InventoryItemResponse,
    PreferencesCreate, PreferencesResponse,
    RecommendationRequest, RecommendationResponse, SimilarPerfume,
    ClusterResponse, ClusterInfo, MapResponse, MapPoint,
    PriceOfferResponse
)
from ml.features import FeatureEngineer, cosine_similarity
from ml.recommend import PerfumeRecommender, find_similar_perfumes
from ml.cluster import analyze_clusters

# Load environment
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Perfume Finder API",
    description="ML-powered perfume recommendation and discovery platform",
    version="1.0.0"
)

# CORS configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def on_startup():
    """Create database tables if they don't exist."""
    create_db_and_tables()
    print("✓ Database initialized")


# ========== Health Check ==========

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "perfume-finder"}


# ========== Perfume Endpoints ==========

@app.get("/perfumes", response_model=List[PerfumeResponse])
def search_perfumes(
    query: Optional[str] = None,
    notes: Optional[str] = Query(None, description="Comma-separated notes"),
    brands: Optional[str] = Query(None, description="Comma-separated brands"),
    seasons: Optional[str] = Query(None, description="Comma-separated seasons"),
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    gender: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session)
):
    """
    Search perfumes with filters.
    
    Supports filtering by name, brand, notes, season, price range, and gender.
    """
    
    statement = select(Perfume)
    
    # Text search in name or brand
    if query:
        statement = statement.where(
            (Perfume.name.contains(query)) | (Perfume.brand.contains(query))
        )
    
    # Brand filter
    if brands:
        brand_list = [b.strip() for b in brands.split(",")]
        statement = statement.where(Perfume.brand.in_(brand_list))
    
    # Gender filter
    if gender:
        statement = statement.where(Perfume.gender == gender)
    
    # Price range
    if price_min is not None:
        statement = statement.where(Perfume.price >= price_min)
    if price_max is not None:
        statement = statement.where(Perfume.price <= price_max)
    
    # Execute query with pagination
    statement = statement.offset(offset).limit(limit)
    perfumes = session.exec(statement).all()
    
    # Post-filter by notes and seasons (since they're JSON fields)
    if notes:
        note_list = {n.strip().lower() for n in notes.split(",")}
        perfumes = [
            p for p in perfumes
            if any(
                n.lower() in note_list
                for n in (p.notes_top + p.notes_middle + p.notes_base)
            )
        ]
    
    if seasons:
        season_list = [s.strip().lower() for s in seasons.split(",")]
        perfumes = [
            p for p in perfumes
            if any(p.season_vector.get(s, 0) > 0.5 for s in season_list)
        ]
    
    return perfumes


@app.get("/perfumes/{perfume_id}", response_model=PerfumeResponse)
def get_perfume(perfume_id: int, session: Session = Depends(get_session)):
    """Get detailed information about a specific perfume."""
    
    perfume = session.get(Perfume, perfume_id)
    if not perfume:
        raise HTTPException(status_code=404, detail="Perfume not found")
    
    return perfume


# ========== Price Endpoints ==========

@app.get("/prices/{perfume_id}", response_model=List[PriceOfferResponse])
def get_price_offers(perfume_id: int, session: Session = Depends(get_session)):
    """Get price offers for a perfume from various retailers."""
    
    statement = select(PriceOffer).where(PriceOffer.perfume_id == perfume_id)
    offers = session.exec(statement).all()
    
    return offers


# ========== User Endpoints ==========

@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate, session: Session = Depends(get_session)):
    """Create a new user (mock authentication)."""
    
    # Check if username already exists
    existing = session.exec(
        select(User).where(User.username == user.username)
    ).first()
    
    if existing:
        return existing  # Return existing user
    
    new_user = User(username=user.username)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    return new_user


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, session: Session = Depends(get_session)):
    """Get user profile."""
    
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


# ========== Inventory Endpoints ==========

@app.get("/users/{user_id}/inventory", response_model=List[InventoryItemResponse])
def get_inventory(user_id: int, session: Session = Depends(get_session)):
    """Get user's perfume inventory with full perfume details."""
    
    statement = select(UserInventory).where(UserInventory.user_id == user_id)
    inventory_items = session.exec(statement).all()
    
    # Enrich with perfume details
    results = []
    for item in inventory_items:
        perfume = session.get(Perfume, item.perfume_id)
        if perfume:
            results.append(InventoryItemResponse(
                id=item.id,
                perfume_id=item.perfume_id,
                rating=item.rating,
                added_at=item.added_at,
                perfume=PerfumeResponse.from_orm(perfume)
            ))
    
    return results


@app.post("/users/{user_id}/inventory")
def add_to_inventory(
    user_id: int,
    request: InventoryAddRequest,
    session: Session = Depends(get_session)
):
    """Add a perfume to user's inventory."""
    
    # Check if perfume exists
    perfume = session.get(Perfume, request.perfume_id)
    if not perfume:
        raise HTTPException(status_code=404, detail="Perfume not found")
    
    # Check if already in inventory
    existing = session.exec(
        select(UserInventory).where(
            UserInventory.user_id == user_id,
            UserInventory.perfume_id == request.perfume_id
        )
    ).first()
    
    if existing:
        # Update rating if provided
        if request.rating is not None:
            existing.rating = request.rating
            session.add(existing)
            session.commit()
        return {"message": "Inventory item updated", "id": existing.id}
    
    # Add new inventory item
    new_item = UserInventory(
        user_id=user_id,
        perfume_id=request.perfume_id,
        rating=request.rating
    )
    session.add(new_item)
    session.commit()
    session.refresh(new_item)
    
    return {"message": "Perfume added to inventory", "id": new_item.id}


# ========== Preferences Endpoints ==========

@app.post("/preferences", response_model=PreferencesResponse)
def save_preferences(
    prefs: PreferencesCreate,
    session: Session = Depends(get_session)
):
    """Save or update user preferences from onboarding questionnaire."""
    
    # Check if preferences already exist
    existing = session.exec(
        select(UserPreferences).where(UserPreferences.user_id == prefs.user_id)
    ).first()
    
    if existing:
        # Update existing
        existing.accord_weights = prefs.accord_weights
        existing.disliked_notes = prefs.disliked_notes
        existing.season_vector = prefs.season_vector
        existing.budget_min = prefs.budget_min
        existing.budget_max = prefs.budget_max
        existing.longevity_pref = prefs.longevity_pref
        existing.sillage_pref = prefs.sillage_pref
        
        session.add(existing)
        session.commit()
        session.refresh(existing)
        
        return existing
    
    # Create new
    new_prefs = UserPreferences(**prefs.dict())
    session.add(new_prefs)
    session.commit()
    session.refresh(new_prefs)
    
    return new_prefs


@app.get("/preferences/{user_id}", response_model=PreferencesResponse)
def get_preferences(user_id: int, session: Session = Depends(get_session)):
    """Get user preferences."""
    
    prefs = session.exec(
        select(UserPreferences).where(UserPreferences.user_id == user_id)
    ).first()
    
    if not prefs:
        raise HTTPException(status_code=404, detail="Preferences not found")
    
    return prefs


# ========== ML Endpoints ==========

@app.get("/ml/map", response_model=MapResponse)
def get_similarity_map(session: Session = Depends(get_session)):
    """
    Get 2D similarity map data for visualization.
    
    Returns perfume coordinates from UMAP dimensionality reduction
    along with cluster information.
    """
    
    perfumes = session.exec(select(Perfume)).all()
    
    # Filter perfumes with computed coordinates
    points = []
    for p in perfumes:
        if p.umap_x is not None and p.umap_y is not None:
            points.append(MapPoint(
                perfume_id=p.id,
                name=p.name,
                brand=p.brand,
                cluster_id=p.cluster_id,
                x=p.umap_x,
                y=p.umap_y,
                price=p.price,
                accords=p.accords
            ))
    
    # Get cluster summaries
    cluster_data = []
    for p in perfumes:
        cluster_data.append({
            "accords": p.accords,
            "notes_top": p.notes_top,
            "notes_middle": p.notes_middle,
            "notes_base": p.notes_base,
            "price": p.price,
            "name": p.name
        })
    
    cluster_ids = np.array([p.cluster_id for p in perfumes if p.cluster_id is not None])
    n_clusters = int(cluster_ids.max()) + 1 if len(cluster_ids) > 0 else 0
    
    clusters = analyze_clusters(cluster_data, cluster_ids, n_clusters)
    cluster_info = [ClusterInfo(**c) for c in clusters]
    
    return MapResponse(points=points, clusters=cluster_info)


@app.post("/ml/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    request: RecommendationRequest,
    session: Session = Depends(get_session)
):
    """
    Get personalized perfume recommendations.
    
    Combines user's inventory (weighted by ratings) and stated preferences
    to generate content-based recommendations with explanations.
    """
    
    # Get user inventory
    inventory = session.exec(
        select(UserInventory).where(UserInventory.user_id == request.user_id)
    ).all()
    
    # Get user preferences
    prefs = session.exec(
        select(UserPreferences).where(UserPreferences.user_id == request.user_id)
    ).first()
    
    if not inventory and not prefs:
        raise HTTPException(
            status_code=400,
            detail="User must have either inventory or preferences"
        )
    
    # Load note vocabulary for feature engineering
    notes = session.exec(select(NoteVocab)).all()
    note_vocab = [n.name for n in notes]
    
    # Initialize feature engineer and recommender
    feature_engineer = FeatureEngineer(note_vocab)
    recommender = PerfumeRecommender()
    
    # Build user vector from inventory
    inventory_vectors = []
    inventory_ratings = []
    
    if inventory:
        for item in inventory:
            perfume = session.get(Perfume, item.perfume_id)
            if perfume and perfume.feature_vector:
                inventory_vectors.append(np.array(perfume.feature_vector))
                inventory_ratings.append(item.rating or 3.0)  # Default to neutral
    
    # Build preference vector
    if prefs:
        pref_vector = feature_engineer.build_preference_vector(
            prefs.accord_weights,
            prefs.season_vector,
            prefs.longevity_pref,
            prefs.sillage_pref,
            prefs.budget_max
        )
    else:
        # Default neutral preferences
        pref_vector = np.zeros(feature_engineer.feature_dim)
    
    # Combine into user vector
    user_vector = recommender.build_user_vector(
        inventory_vectors,
        inventory_ratings,
        pref_vector
    )
    
    # Get all perfumes for recommendation
    all_perfumes = session.exec(select(Perfume)).all()
    perfume_vectors = []
    perfume_ids = []
    perfume_metadata = []
    
    for p in all_perfumes:
        if p.feature_vector:
            perfume_vectors.append(np.array(p.feature_vector))
            perfume_ids.append(p.id)
            perfume_metadata.append({
                "price": p.price,
                "notes_top": p.notes_top,
                "notes_middle": p.notes_middle,
                "notes_base": p.notes_base,
                "accords": p.accords,
                "season_vector": p.season_vector,
                "longevity": p.longevity,
                "sillage": p.sillage
            })
    
    # Get recommendations
    owned_ids = {item.perfume_id for item in inventory} if request.exclude_owned else set()
    disliked = set(prefs.disliked_notes) if prefs else set()
    budget_min = prefs.budget_min if prefs and request.filter_by_budget else 0
    budget_max = prefs.budget_max if prefs and request.filter_by_budget else float('inf')
    preferred_seasons = [s for s, v in prefs.season_vector.items() if v > 0.5] if prefs else []
    
    recommendations = recommender.recommend(
        user_vector=user_vector,
        all_perfume_vectors=np.array(perfume_vectors),
        all_perfume_ids=perfume_ids,
        owned_perfume_ids=owned_ids,
        disliked_notes=disliked,
        perfume_metadata=perfume_metadata,
        budget_min=budget_min,
        budget_max=budget_max,
        filter_by_season=request.filter_by_season,
        preferred_seasons=preferred_seasons,
        top_n=request.top_n,
        exclude_owned=request.exclude_owned
    )
    
    # Convert to response format
    similar_perfumes = []
    for perfume_id, similarity, explanation in recommendations:
        perfume = session.get(Perfume, perfume_id)
        if perfume:
            similar_perfumes.append(SimilarPerfume(
                perfume=PerfumeResponse.from_orm(perfume),
                similarity=similarity,
                explanation=explanation
            ))
    
    return RecommendationResponse(
        recommendations=similar_perfumes,
        based_on_inventory_count=len(inventory),
        based_on_preferences=prefs is not None
    )


@app.get("/ml/clusters", response_model=ClusterResponse)
def get_clusters(session: Session = Depends(get_session)):
    """Get cluster analysis summary."""
    
    perfumes = session.exec(select(Perfume)).all()
    
    cluster_data = []
    for p in perfumes:
        cluster_data.append({
            "accords": p.accords,
            "notes_top": p.notes_top,
            "notes_middle": p.notes_middle,
            "notes_base": p.notes_base,
            "price": p.price,
            "name": p.name
        })
    
    cluster_ids = np.array([p.cluster_id for p in perfumes if p.cluster_id is not None])
    n_clusters = int(cluster_ids.max()) + 1 if len(cluster_ids) > 0 else 0
    
    clusters = analyze_clusters(cluster_data, cluster_ids, n_clusters)
    cluster_info = [ClusterInfo(**c) for c in clusters]
    
    return ClusterResponse(
        clusters=cluster_info,
        total_perfumes=len(perfumes)
    )


@app.get("/perfumes/{perfume_id}/similar", response_model=List[SimilarPerfume])
def get_similar_perfumes(
    perfume_id: int,
    top_n: int = Query(5, ge=1, le=20),
    session: Session = Depends(get_session)
):
    """
    Find perfumes similar to a given perfume.
    
    Used for "Similar Perfumes" carousel on detail pages.
    """
    
    # Get query perfume
    query_perfume = session.get(Perfume, perfume_id)
    if not query_perfume or not query_perfume.feature_vector:
        raise HTTPException(status_code=404, detail="Perfume not found")
    
    query_vector = np.array(query_perfume.feature_vector)
    
    # Get all other perfumes
    all_perfumes = session.exec(select(Perfume)).all()
    perfume_vectors = []
    perfume_ids = []
    perfume_metadata = []
    
    for p in all_perfumes:
        if p.feature_vector and p.id != perfume_id:
            perfume_vectors.append(np.array(p.feature_vector))
            perfume_ids.append(p.id)
            perfume_metadata.append({
                "accords": p.accords,
                "notes_top": p.notes_top,
                "price": p.price
            })
    
    # Find similar perfumes
    similar = find_similar_perfumes(
        query_vector=query_vector,
        all_perfume_vectors=np.array(perfume_vectors),
        all_perfume_ids=perfume_ids,
        perfume_metadata=perfume_metadata,
        exclude_id=perfume_id,
        top_n=top_n
    )
    
    # Convert to response
    results = []
    for pid, similarity, explanation in similar:
        perfume = session.get(Perfume, pid)
        if perfume:
            results.append(SimilarPerfume(
                perfume=PerfumeResponse.from_orm(perfume),
                similarity=similarity,
                explanation=explanation
            ))
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

