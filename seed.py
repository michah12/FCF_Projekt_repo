"""
Database seeding script.

Loads mock perfume data, computes ML features, runs clustering and UMAP,
and populates the database with all necessary data including price offers.
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from sqlmodel import Session, select
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from db import engine, create_db_and_tables
from models import (
    Perfume, NoteVocab, PerfumeNote, PriceOffer,
    User, UserInventory, UserPreferences
)
from ml.features import FeatureEngineer
from ml.cluster import PerfumeClusterer, analyze_clusters

load_dotenv()


def load_mock_data() -> list:
    """Load perfume data from JSON file."""
    
    json_path = Path(__file__).parent.parent / "data" / "mock_perfumes.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Mock data not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} perfumes from {json_path}")
    return data


def build_note_vocabulary(perfume_data: list) -> list:
    """
    Extract and count all unique notes across all perfumes.
    
    Returns:
        List of note names sorted by frequency (most common first)
    """
    
    note_counter = Counter()
    
    for perfume in perfume_data:
        all_notes = (
            perfume.get("notes_top", []) +
            perfume.get("notes_middle", []) +
            perfume.get("notes_base", [])
        )
        # Normalize to lowercase
        note_counter.update([note.lower().strip() for note in all_notes])
    
    # Sort by frequency
    sorted_notes = [note for note, count in note_counter.most_common()]
    
    print(f"✓ Built vocabulary with {len(sorted_notes)} unique notes")
    return sorted_notes


def seed_perfumes(session: Session, perfume_data: list, note_vocab: list) -> list:
    """
    Insert perfumes into database and compute feature vectors.
    
    Returns:
        List of created Perfume objects
    """
    
    feature_engineer = FeatureEngineer(note_vocab, max_notes=100)
    perfumes = []
    
    print("⏳ Seeding perfumes and computing features...")
    
    for data in perfume_data:
        # Extract features
        feature_vector, meta = feature_engineer.extract_features(
            accords=data.get("accords", {}),
            notes_top=data.get("notes_top", []),
            notes_middle=data.get("notes_middle", []),
            notes_base=data.get("notes_base", []),
            season_vector=data.get("season_vector", {}),
            longevity=data.get("longevity", 3),
            sillage=data.get("sillage", 3),
            price=data.get("price", 50.0)
        )
        
        # Create perfume object
        perfume = Perfume(
            id=data.get("id"),
            name=data.get("name", "Unknown"),
            brand=data.get("brand", "Unknown"),
            year=data.get("year"),
            gender=data.get("gender"),
            price=data.get("price", 50.0),
            longevity=data.get("longevity", 3),
            sillage=data.get("sillage", 3),
            accords=data.get("accords", {}),
            season_vector=data.get("season_vector", {}),
            notes_top=data.get("notes_top", []),
            notes_middle=data.get("notes_middle", []),
            notes_base=data.get("notes_base", []),
            description=data.get("description", ""),
            feature_vector=feature_vector.tolist()  # Store as JSON
        )
        
        perfumes.append(perfume)
        session.add(perfume)
    
    session.commit()
    
    print(f"✓ Seeded {len(perfumes)} perfumes with feature vectors")
    return perfumes


def run_clustering(session: Session, perfumes: list):
    """
    Run KMeans clustering and UMAP dimensionality reduction.
    
    Updates perfume records with cluster_id, umap_x, and umap_y.
    """
    
    print("⏳ Running ML pipeline (clustering + UMAP)...")
    
    # Extract feature vectors
    feature_matrix = np.array([p.feature_vector for p in perfumes])
    
    # Initialize and fit clusterer
    n_clusters = int(os.getenv("CLUSTER_COUNT", "6"))
    random_seed = int(os.getenv("RANDOM_SEED", "42"))
    
    clusterer = PerfumeClusterer(n_clusters=n_clusters, random_state=random_seed)
    cluster_ids, coords_2d = clusterer.fit_transform(feature_matrix)
    
    # Update perfumes with results
    for i, perfume in enumerate(perfumes):
        perfume.cluster_id = int(cluster_ids[i])
        perfume.umap_x = float(coords_2d[i, 0])
        perfume.umap_y = float(coords_2d[i, 1])
        session.add(perfume)
    
    session.commit()
    
    print(f"✓ Clustered into {n_clusters} groups and computed 2D coordinates")
    
    # Print cluster summary
    perfume_metadata = []
    for p in perfumes:
        perfume_metadata.append({
            "accords": p.accords,
            "notes_top": p.notes_top,
            "notes_middle": p.notes_middle,
            "notes_base": p.notes_base,
            "price": p.price,
            "name": p.name
        })
    
    clusters = analyze_clusters(perfume_metadata, cluster_ids, n_clusters)
    
    print("\n📊 Cluster Summary:")
    for cluster in clusters:
        print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} perfumes")
        top_accords = ", ".join([f"{k}({v:.2f})" for k, v in list(cluster['top_accords'].items())[:3]])
        print(f"    Top accords: {top_accords}")
        print(f"    Representative: {', '.join(cluster['representative_perfumes'][:2])}")
    print()


def seed_note_vocabulary(session: Session, note_vocab: list):
    """Insert note vocabulary into database."""
    
    print("⏳ Seeding note vocabulary...")
    
    for note_name in note_vocab:
        note = NoteVocab(name=note_name)
        session.add(note)
    
    session.commit()
    
    print(f"✓ Seeded {len(note_vocab)} notes into vocabulary")


def seed_price_offers(session: Session, perfumes: list):
    """Generate mock price offers for perfumes."""
    
    print("⏳ Generating price offers...")
    
    shops = ["FragranceNet", "FragranceX", "Sephora", "Ulta", "Notino"]
    
    offer_count = 0
    for perfume in perfumes:
        # Generate 2-4 random offers per perfume
        num_offers = np.random.randint(2, 5)
        base_price = perfume.price or 50.0
        
        for i in range(num_offers):
            shop = shops[i % len(shops)]
            # Price variation ±15%
            price_variation = np.random.uniform(0.85, 1.15)
            offer_price = round(base_price * price_variation, 2)
            
            offer = PriceOffer(
                perfume_id=perfume.id,
                shop_name=shop,
                url=f"https://{shop.lower()}.com/perfume/{perfume.id}",
                price=offer_price,
                currency="USD"
            )
            session.add(offer)
            offer_count += 1
    
    session.commit()
    
    print(f"✓ Generated {offer_count} price offers")


def seed_demo_users(session: Session, perfumes: list):
    """Create demo users with sample inventories and preferences."""
    
    print("⏳ Creating demo users...")
    
    # Demo user 1: Loves fresh citrus scents
    user1 = User(username="demo_user")
    session.add(user1)
    session.commit()
    session.refresh(user1)
    
    # Add some perfumes to inventory
    citrus_perfumes = [p for p in perfumes if p.accords.get("citrus", 0) > 0.6][:3]
    for perfume in citrus_perfumes:
        inv = UserInventory(user_id=user1.id, perfume_id=perfume.id, rating=4)
        session.add(inv)
    
    # Add preferences
    prefs1 = UserPreferences(
        user_id=user1.id,
        accord_weights={"citrus": 0.9, "fresh": 0.8, "aquatic": 0.7},
        disliked_notes=["patchouli", "oud"],
        season_vector={"spring": 0.9, "summer": 1.0, "fall": 0.5, "winter": 0.3},
        budget_min=30,
        budget_max=150,
        longevity_pref=3,
        sillage_pref=3
    )
    session.add(prefs1)
    
    session.commit()
    
    print(f"✓ Created demo user with inventory and preferences")


def main():
    """Main seeding routine."""
    
    print("\n" + "="*60)
    print("🌸 PERFUME FINDER - Database Seeding")
    print("="*60 + "\n")
    
    # Create tables
    print("⏳ Creating database tables...")
    create_db_and_tables()
    print("✓ Database tables created\n")
    
    # Load data
    perfume_data = load_mock_data()
    
    with Session(engine) as session:
        # Check if already seeded
        existing = session.exec(select(Perfume)).first()
        if existing:
            response = input("⚠️  Database already contains data. Re-seed? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            
            # Clear existing data
            print("⏳ Clearing existing data...")
            from sqlalchemy import text
            for model in [PriceOffer, UserInventory, UserPreferences, User, PerfumeNote, Perfume, NoteVocab]:
                session.exec(text(f"DELETE FROM {model.__tablename__}"))
            session.commit()
            print("✓ Database cleared\n")
        
        # Build note vocabulary
        note_vocab = build_note_vocabulary(perfume_data)
        seed_note_vocabulary(session, note_vocab)
        
        # Seed perfumes
        perfumes = seed_perfumes(session, perfume_data, note_vocab)
        
        # Run ML pipeline
        run_clustering(session, perfumes)
        
        # Seed price offers
        seed_price_offers(session, perfumes)
        
        # Create demo users
        seed_demo_users(session, perfumes)
    
    print("="*60)
    print("✅ Seeding complete! Database is ready.")
    print("="*60 + "\n")
    print("Next steps:")
    print("  1. Start the backend: uvicorn backend.app:app --reload")
    print("  2. Start the frontend: cd frontend && npm run dev")
    print()


if __name__ == "__main__":
    main()

