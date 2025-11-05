"""
🌸 Perfume Finder - Streamlit Application

AI-Powered Fragrance Discovery Platform
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.db import engine, create_db_and_tables
from backend.models import Perfume, User, UserInventory, UserPreferences
from backend.ml.features import FeatureEngineer, cosine_similarity
from backend.ml.recommend import PerfumeRecommender
from sqlmodel import Session, select

# Page configuration
st.set_page_config(
    page_title="Perfume Finder",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
create_db_and_tables()

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# ========== Helper Functions ==========

@st.cache_data
def load_perfumes():
    """Load all perfumes from database."""
    with Session(engine) as session:
        perfumes = session.exec(select(Perfume)).all()
        return [
            {
                'id': p.id,
                'name': p.name,
                'brand': p.brand,
                'price': p.price,
                'year': p.year,
                'gender': p.gender,
                'longevity': p.longevity,
                'sillage': p.sillage,
                'accords': p.accords,
                'season_vector': p.season_vector,
                'notes_top': p.notes_top,
                'notes_middle': p.notes_middle,
                'notes_base': p.notes_base,
                'description': p.description,
                'cluster_id': p.cluster_id,
                'umap_x': p.umap_x,
                'umap_y': p.umap_y,
                'feature_vector': p.feature_vector
            }
            for p in perfumes
        ]

def get_user_inventory(user_id):
    """Get user's perfume collection."""
    with Session(engine) as session:
        inventory = session.exec(
            select(UserInventory).where(UserInventory.user_id == user_id)
        ).all()
        
        result = []
        for item in inventory:
            perfume = session.get(Perfume, item.perfume_id)
            if perfume:
                result.append({
                    'inventory_id': item.id,
                    'perfume_id': perfume.id,
                    'name': perfume.name,
                    'brand': perfume.brand,
                    'price': perfume.price,
                    'rating': item.rating,
                    'accords': perfume.accords,
                    'notes_top': perfume.notes_top,
                    'notes_middle': perfume.notes_middle,
                    'notes_base': perfume.notes_base,
                    'feature_vector': perfume.feature_vector
                })
        return result

def get_user_preferences(user_id):
    """Get user preferences."""
    with Session(engine) as session:
        prefs = session.exec(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        ).first()
        return prefs

def create_radar_chart(accords, title="Accord Profile"):
    """Create radar chart for accords."""
    accord_names = ['Citrus', 'Floral', 'Woody', 'Oriental', 'Fresh', 
                    'Gourmand', 'Aromatic', 'Fruity', 'Spicy', 'Aquatic']
    accord_keys = ['citrus', 'floral', 'woody', 'oriental', 'fresh',
                   'gourmand', 'aromatic', 'fruity', 'spicy', 'aquatic']
    
    values = [accords.get(key, 0) for key in accord_keys]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=accord_names,
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.3)',
        line=dict(color='#8b5cf6', width=2),
        name='Accords'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title=title,
        height=400
    )
    
    return fig

def create_bar_chart(notes, title="Top Notes"):
    """Create bar chart for note frequencies."""
    from collections import Counter
    note_counts = Counter(notes)
    top_notes = note_counts.most_common(10)
    
    if not top_notes:
        return None
    
    df = pd.DataFrame(top_notes, columns=['Note', 'Count'])
    
    fig = px.bar(
        df,
        x='Note',
        y='Count',
        title=title,
        color='Count',
        color_continuous_scale='Purples'
    )
    fig.update_layout(height=400)
    
    return fig

# ========== Pages ==========

def home_page():
    """Landing page with problem statement."""
    st.markdown('<h1 class="main-header">🌸 Perfume Finder</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Fragrance Discovery Platform")
    
    st.markdown("---")
    
    # Problem Statement
    st.header("The Problem We Solve")
    st.info("""
    **Finding the right perfume is challenging and expensive.**
    
    With thousands of fragrances on the market, consumers face:
    - 🤯 Information overload with too many choices
    - 💸 Expensive trial-and-error purchases
    - 🔍 Difficulty comparing similar options across brands
    - 🎯 Lack of personalization based on taste
    
    **Our Solution:** Perfume Finder uses machine learning to analyze fragrance profiles,
    cluster similar scents, and provide personalized recommendations.
    """)
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔍 Smart Search")
        st.write("Find perfumes by notes, accords, season, or price")
    
    with col2:
        st.markdown("### 📊 Visual Analysis")
        st.write("Explore with radar charts, bar charts, and similarity maps")
    
    with col3:
        st.markdown("### ✨ ML Recommendations")
        st.write("Get AI-powered suggestions based on your taste")
    
    st.markdown("---")
    
    # Technology Showcase
    st.header("Powered by Machine Learning")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **🤖 Content-Based Filtering**
        - Recommends perfumes by comparing feature vectors
        - Analyzes accords, notes, and performance metrics
        
        **📊 K-Means Clustering**
        - Groups similar fragrances into 6 clusters
        - Helps discover new options in your preferred style
        """)
    
    with tech_col2:
        st.markdown("""
        **🗺️ UMAP Visualization**
        - 2D similarity map shows relationships between perfumes
        - Interactive exploration of fragrance space
        
        **💡 Explainable AI**
        - Every recommendation comes with clear justifications
        - Understand why each perfume matches your taste
        """)
    
    st.markdown("---")
    st.success("👈 Use the sidebar to navigate between features!")

def onboarding_page():
    """Onboarding questionnaire to capture user preferences."""
    st.title("🎨 Create Your Profile")
    
    if st.session_state.user_id is not None:
        st.success(f"Welcome back, {st.session_state.username}! You can update your preferences below.")
    
    with st.form("onboarding_form"):
        st.subheader("Step 1: Basic Info")
        username = st.text_input("Your Name", value=st.session_state.username or "")
        
        st.subheader("Step 2: Favorite Accords")
        st.write("Select your preferred fragrance families:")
        
        col1, col2 = st.columns(2)
        accord_weights = {}
        
        accords = ['citrus', 'floral', 'woody', 'oriental', 'fresh', 
                   'gourmand', 'aromatic', 'fruity', 'spicy', 'aquatic']
        
        for i, accord in enumerate(accords):
            col = col1 if i < 5 else col2
            with col:
                weight = st.slider(
                    f"{accord.capitalize()}",
                    0.0, 1.0, 0.5,
                    key=f"accord_{accord}"
                )
                if weight > 0:
                    accord_weights[accord] = weight
        
        st.subheader("Step 3: Disliked Notes")
        disliked_options = ['patchouli', 'oud', 'musk', 'leather', 'vanilla', 
                           'coconut', 'amber', 'tobacco']
        disliked_notes = st.multiselect("Notes to avoid:", disliked_options)
        
        st.subheader("Step 4: Seasons")
        seasons = st.multiselect(
            "Preferred seasons:",
            ['spring', 'summer', 'fall', 'winter'],
            default=['spring', 'summer']
        )
        season_vector = {s: 1.0 for s in seasons}
        
        st.subheader("Step 5: Budget & Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            budget_min = st.number_input("Min Budget ($)", 0, 1000, 30)
            budget_max = st.number_input("Max Budget ($)", 0, 1000, 200)
        
        with col2:
            longevity_pref = st.slider("Desired Longevity", 1, 5, 3)
            sillage_pref = st.slider("Desired Sillage", 1, 5, 3)
        
        submitted = st.form_submit_button("Save Profile", type="primary")
        
        if submitted and username:
            # Create or update user
            with Session(engine) as session:
                user = session.exec(
                    select(User).where(User.username == username)
                ).first()
                
                if not user:
                    user = User(username=username)
                    session.add(user)
                    session.commit()
                    session.refresh(user)
                
                st.session_state.user_id = user.id
                st.session_state.username = user.username
                
                # Save preferences
                prefs = session.exec(
                    select(UserPreferences).where(UserPreferences.user_id == user.id)
                ).first()
                
                if prefs:
                    prefs.accord_weights = accord_weights
                    prefs.disliked_notes = disliked_notes
                    prefs.season_vector = season_vector
                    prefs.budget_min = budget_min
                    prefs.budget_max = budget_max
                    prefs.longevity_pref = longevity_pref
                    prefs.sillage_pref = sillage_pref
                else:
                    prefs = UserPreferences(
                        user_id=user.id,
                        accord_weights=accord_weights,
                        disliked_notes=disliked_notes,
                        season_vector=season_vector,
                        budget_min=budget_min,
                        budget_max=budget_max,
                        longevity_pref=longevity_pref,
                        sillage_pref=sillage_pref
                    )
                    session.add(prefs)
                
                session.commit()
            
            st.success("✅ Profile saved! Check out the other pages to explore perfumes.")
            
            # Show radar chart
            if accord_weights:
                st.subheader("Your Preference Profile")
                fig = create_radar_chart(accord_weights, "Your Accords")
                st.plotly_chart(fig, use_container_width=True)

def search_page():
    """Search and browse perfumes."""
    st.title("🔍 Search Perfumes")
    
    perfumes = load_perfumes()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input("Search by name or brand", "")
    
    with col2:
        price_range = st.slider("Price Range ($)", 0, 500, (0, 500))
    
    with col3:
        gender_filter = st.selectbox("Gender", ["All", "male", "female", "unisex"])
    
    # Apply filters
    filtered = perfumes
    
    if search_query:
        filtered = [
            p for p in filtered
            if search_query.lower() in p['name'].lower() 
            or search_query.lower() in p['brand'].lower()
        ]
    
    filtered = [
        p for p in filtered
        if price_range[0] <= (p['price'] or 0) <= price_range[1]
    ]
    
    if gender_filter != "All":
        filtered = [p for p in filtered if p['gender'] == gender_filter]
    
    st.write(f"**Found {len(filtered)} perfumes**")
    
    # Display results
    for i in range(0, len(filtered), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(filtered):
                perfume = filtered[i + j]
                with col:
                    st.markdown(f"### {perfume['name']}")
                    st.write(f"**{perfume['brand']}**")
                    if perfume['price']:
                        st.write(f"💰 ${perfume['price']:.2f}")
                    
                    # Top accords
                    top_accords = sorted(
                        perfume['accords'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    st.write("🎨 " + ", ".join([a[0] for a in top_accords]))
                    
                    if st.button("View Details", key=f"view_{perfume['id']}"):
                        st.session_state.selected_perfume_id = perfume['id']
                        st.rerun()

def perfume_detail_page():
    """Detailed perfume information."""
    if 'selected_perfume_id' not in st.session_state:
        st.warning("Please select a perfume from the Search page first.")
        return
    
    perfume_id = st.session_state.selected_perfume_id
    perfumes = load_perfumes()
    perfume = next((p for p in perfumes if p['id'] == perfume_id), None)
    
    if not perfume:
        st.error("Perfume not found")
        return
    
    # Header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title(perfume['name'])
        st.subheader(perfume['brand'])
        if perfume['year']:
            st.write(f"Released: {perfume['year']}")
    
    with col2:
        if perfume['price']:
            st.metric("Price", f"${perfume['price']:.2f}")
        
        # Add to collection button
        if st.session_state.user_id:
            rating = st.slider("Rate this perfume", 1, 5, 5, key=f"rating_{perfume_id}")
            if st.button("➕ Add to Collection", type="primary"):
                with Session(engine) as session:
                    # Check if already in inventory
                    existing = session.exec(
                        select(UserInventory).where(
                            UserInventory.user_id == st.session_state.user_id,
                            UserInventory.perfume_id == perfume_id
                        )
                    ).first()
                    
                    if not existing:
                        inv = UserInventory(
                            user_id=st.session_state.user_id,
                            perfume_id=perfume_id,
                            rating=rating
                        )
                        session.add(inv)
                        session.commit()
                        st.success("Added to your collection!")
                    else:
                        st.info("Already in your collection")
    
    st.markdown("---")
    
    # Details
    tab1, tab2, tab3 = st.tabs(["📊 Profile", "🎵 Notes", "📈 Performance"])
    
    with tab1:
        if perfume['description']:
            st.write(perfume['description'])
        
        # Radar chart
        fig = create_radar_chart(perfume['accords'], "Accord Profile")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Top Notes**")
            for note in perfume['notes_top']:
                st.write(f"• {note}")
        
        with col2:
            st.markdown("**Middle Notes**")
            for note in perfume['notes_middle']:
                st.write(f"• {note}")
        
        with col3:
            st.markdown("**Base Notes**")
            for note in perfume['notes_base']:
                st.write(f"• {note}")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Longevity", f"{perfume['longevity']}/5")
            st.progress(perfume['longevity'] / 5)
        
        with col2:
            st.metric("Sillage", f"{perfume['sillage']}/5")
            st.progress(perfume['sillage'] / 5)
        
        # Seasons
        st.markdown("**Best Seasons:**")
        best_seasons = [
            s for s, score in perfume['season_vector'].items()
            if score > 0.5
        ]
        st.write(", ".join([s.capitalize() for s in best_seasons]))

def inventory_page():
    """User's perfume collection with visualizations."""
    st.title("💼 My Collection")
    
    if not st.session_state.user_id:
        st.warning("Please create your profile first!")
        return
    
    inventory = get_user_inventory(st.session_state.user_id)
    
    if not inventory:
        st.info("Your collection is empty. Start adding perfumes from the Search page!")
        return
    
    st.write(f"**You have {len(inventory)} perfume(s) in your collection**")
    
    # Visualizations
    st.subheader("📊 Collection Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average accord profile
        accord_totals = {}
        for item in inventory:
            for accord, weight in item['accords'].items():
                accord_totals[accord] = accord_totals.get(accord, 0) + weight
        
        avg_accords = {k: v / len(inventory) for k, v in accord_totals.items()}
        fig = create_radar_chart(avg_accords, "Average Collection Profile")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top notes
        all_notes = []
        for item in inventory:
            all_notes.extend(item['notes_top'])
            all_notes.extend(item['notes_middle'])
            all_notes.extend(item['notes_base'])
        
        fig = create_bar_chart(all_notes, "Most Common Notes")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # List perfumes
    st.subheader("Your Perfumes")
    
    for item in inventory:
        with st.expander(f"🌸 {item['name']} by {item['brand']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if item['rating']:
                    st.write(f"⭐ Your rating: {'★' * item['rating']}")
                
                st.write(f"💰 ${item['price']:.2f}" if item['price'] else "Price N/A")
                
                # Top notes preview
                st.write("**Top notes:** " + ", ".join(item['notes_top'][:4]))
            
            with col2:
                if st.button("Remove", key=f"remove_{item['inventory_id']}"):
                    with Session(engine) as session:
                        inv = session.get(UserInventory, item['inventory_id'])
                        if inv:
                            session.delete(inv)
                            session.commit()
                            st.success("Removed!")
                            st.rerun()

def recommendations_page():
    """Personalized recommendations."""
    st.title("✨ Your Recommendations")
    
    if not st.session_state.user_id:
        st.warning("Please create your profile first!")
        return
    
    inventory = get_user_inventory(st.session_state.user_id)
    prefs = get_user_preferences(st.session_state.user_id)
    
    if not inventory and not prefs:
        st.info("Add some perfumes to your collection or complete your profile to get recommendations!")
        return
    
    # Settings
    with st.expander("⚙️ Recommendation Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_n = st.slider("Number of recommendations", 5, 20, 10)
        
        with col2:
            filter_budget = st.checkbox("Filter by budget", value=True)
        
        with col3:
            filter_season = st.checkbox("Filter by season", value=False)
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Analyzing your taste..."):
            # Load all perfumes
            perfumes = load_perfumes()
            perfume_vectors = np.array([p['feature_vector'] for p in perfumes if p['feature_vector']])
            perfume_ids = [p['id'] for p in perfumes if p['feature_vector']]
            
            # Build user vector
            with Session(engine) as session:
                notes = session.exec(select(Perfume)).all()
                note_vocab = list(set([
                    n for p in notes
                    for n in (p.notes_top + p.notes_middle + p.notes_base)
                ]))[:100]
            
            feature_engineer = FeatureEngineer(note_vocab)
            recommender = PerfumeRecommender()
            
            # Get inventory vectors
            inventory_vectors = []
            inventory_ratings = []
            
            for item in inventory:
                if item['feature_vector']:
                    inventory_vectors.append(np.array(item['feature_vector']))
                    inventory_ratings.append(item.get('rating', 3.0))
            
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
                pref_vector = np.zeros(feature_engineer.feature_dim)
            
            # Combine
            user_vector = recommender.build_user_vector(
                inventory_vectors,
                inventory_ratings,
                pref_vector
            )
            
            # Get recommendations
            owned_ids = {item['perfume_id'] for item in inventory}
            disliked = set(prefs.disliked_notes) if prefs else set()
            budget_min = prefs.budget_min if prefs and filter_budget else 0
            budget_max = prefs.budget_max if prefs and filter_budget else float('inf')
            preferred_seasons = [s for s, v in prefs.season_vector.items() if v > 0.5] if prefs else []
            
            perfume_metadata = []
            for p in perfumes:
                if p['feature_vector']:
                    perfume_metadata.append({
                        'price': p['price'],
                        'notes_top': p['notes_top'],
                        'notes_middle': p['notes_middle'],
                        'notes_base': p['notes_base'],
                        'accords': p['accords'],
                        'season_vector': p['season_vector'],
                        'longevity': p['longevity'],
                        'sillage': p['sillage']
                    })
            
            recommendations = recommender.recommend(
                user_vector=user_vector,
                all_perfume_vectors=perfume_vectors,
                all_perfume_ids=perfume_ids,
                owned_perfume_ids=owned_ids,
                disliked_notes=disliked,
                perfume_metadata=perfume_metadata,
                budget_min=budget_min,
                budget_max=budget_max,
                filter_by_season=filter_season,
                preferred_seasons=preferred_seasons,
                top_n=top_n
            )
            
            # Display results
            st.success(f"Found {len(recommendations)} recommendations!")
            
            for i, (perfume_id, similarity, explanation) in enumerate(recommendations, 1):
                perfume = next((p for p in perfumes if p['id'] == perfume_id), None)
                if perfume:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"### #{i}. {perfume['name']}")
                            st.write(f"**{perfume['brand']}**")
                            st.info(f"💡 {explanation}")
                            
                            # Top accords
                            top_accords = sorted(
                                perfume['accords'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:3]
                            st.write("🎨 " + ", ".join([f"{a[0]} ({a[1]:.0%})" for a in top_accords]))
                        
                        with col2:
                            st.metric("Match", f"{similarity:.0%}")
                            if perfume['price']:
                                st.write(f"💰 ${perfume['price']:.2f}")
                            
                            if st.button("View Details", key=f"rec_{perfume_id}"):
                                st.session_state.selected_perfume_id = perfume_id
                                st.rerun()
                        
                        st.markdown("---")

def similarity_map_page():
    """Interactive 2D similarity map."""
    st.title("🗺️ Perfume Similarity Map")
    
    st.info("""
    **How to use:** This map shows perfumes in 2D space using UMAP dimensionality reduction.
    Perfumes close together have similar profiles. Colors represent different clusters discovered by K-Means.
    """)
    
    perfumes = load_perfumes()
    
    # Filter to perfumes with coordinates
    map_data = [p for p in perfumes if p['umap_x'] is not None and p['umap_y'] is not None]
    
    if not map_data:
        st.warning("No map data available. Please run the seed script first.")
        return
    
    df = pd.DataFrame([{
        'x': p['umap_x'],
        'y': p['umap_y'],
        'name': p['name'],
        'brand': p['brand'],
        'cluster': f"Cluster {p['cluster_id']}" if p['cluster_id'] is not None else "Unknown",
        'price': p['price'] or 0,
        'id': p['id']
    } for p in map_data])
    
    # Cluster filter
    clusters = df['cluster'].unique()
    selected_cluster = st.selectbox("Filter by cluster:", ["All"] + list(clusters))
    
    if selected_cluster != "All":
        df = df[df['cluster'] == selected_cluster]
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['name', 'brand', 'price'],
        title=f"Perfume Similarity Map ({len(df)} perfumes)",
        width=900,
        height=600,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster analysis
    st.subheader("📊 Cluster Analysis")
    
    # Group by cluster
    cluster_stats = df.groupby('cluster').agg({
        'name': 'count',
        'price': 'mean'
    }).reset_index()
    cluster_stats.columns = ['Cluster', 'Count', 'Avg Price']
    
    st.dataframe(cluster_stats, use_container_width=True)

# ========== Main App ==========

def main():
    """Main application logic."""
    
    # Sidebar navigation
    st.sidebar.title("🌸 Navigation")
    
    # User info
    if st.session_state.user_id:
        st.sidebar.success(f"👤 {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
    else:
        st.sidebar.info("Create your profile to get started!")
    
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.radio(
        "Go to:",
        ["🏠 Home", "🎨 Onboarding", "🔍 Search", "💼 My Collection", 
         "✨ Recommendations", "🗺️ Similarity Map"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **Perfume Finder** uses machine learning to help you discover your perfect fragrance.
    
    Features:
    - K-Means clustering
    - UMAP visualization  
    - Content-based recommendations
    - Explainable AI
    """)
    
    # Route to pages
    if page == "🏠 Home":
        home_page()
    elif page == "🎨 Onboarding":
        onboarding_page()
    elif page == "🔍 Search":
        search_page()
    elif page == "💼 My Collection":
        inventory_page()
    elif page == "✨ Recommendations":
        recommendations_page()
    elif page == "🗺️ Similarity Map":
        similarity_map_page()

if __name__ == "__main__":
    main()

