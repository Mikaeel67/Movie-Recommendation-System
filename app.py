import streamlit as st
import requests
import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────
API_URL = "https://bagel-hardening-phoniness.ngrok-free.dev"

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .movie-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #e50914;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .movie-title { color: #ffffff; font-size: 18px; font-weight: bold; margin-bottom: 4px; }
    .movie-meta  { color: #a0a0b0; font-size: 13px; margin: 2px 0; }
    .score-badge {
        background: #e50914; color: white;
        padding: 2px 10px; border-radius: 20px;
        font-size: 13px; font-weight: bold;
    }
    .african-badge {
        background: #f5a623; color: black;
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: bold;
    }
    .ai-explanation {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 14px;
        margin: 12px 0;
        color: #c0e0ff;
        font-style: italic;
    }
    .section-header {
        color: #e50914; font-size: 22px; font-weight: bold;
        border-bottom: 2px solid #e50914;
        padding-bottom: 8px; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─── API HELPERS ─────────────────────────────────────────────────
def check_api_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=10)
        return res.json() if res.status_code == 200 else None
    except:
        return None


def get_recommendations(title, user_id=1, n=10):
    try:
        res = requests.post(f"{API_URL}/recommend",
                            json={"title": title, "user_id": user_id, "n": n},
                            timeout=30)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def nl_search(query, n=10):
    try:
        res = requests.post(f"{API_URL}/nl-search",
                            json={"query": query, "n": n},
                            timeout=30)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def search_movies(q):
    try:
        res = requests.get(f"{API_URL}/movies/search", params={"q": q}, timeout=10)
        return res.json() if res.status_code == 200 else None
    except:
        return None


# ─── MOVIE CARD ───────────────────────────────────────────────────
def render_movie_card(movie, score_key='hybrid_score', index=0):
    is_african  = movie.get('source') == 'African_IMDb'
    score       = movie.get(score_key, movie.get('relevance_score', 0))
    badge       = '🌍 African' if is_african else '🎬 MovieLens'
    badge_class = 'african-badge' if is_african else 'score-badge'
    country     = f" • {movie['country']}" if movie.get('country') and movie['country'] != 'Unknown' else ''
    desc        = movie.get('description', '')
    if desc == 'No description available':
        desc = ''

    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">
            {'🏆' if index == 0 else f'{index+1}.'} {movie['title']}
            &nbsp;<span class="{badge_class}">{badge}</span>
        </div>
        <div class="movie-meta">🎭 {movie.get('genre', 'N/A')}</div>
        <div class="movie-meta">⭐ Rating: {round(float(movie.get('rating', 0) or 0), 2)}{country}</div>
        {f'<div class="movie-meta">📝 {desc[:120]}...</div>' if desc else ''}
        <div class="movie-meta">🎯 Score: <span class="score-badge">{round(float(score or 0), 3)}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown("*African + Global Movies | Hybrid ML*")
    st.divider()

    health = check_api_health()
    if health:
        st.success("✅ API Connected")
        st.markdown(f"**Total Movies:** {health['movies']:,}")
        st.markdown(f"**MovieLens:** {health['sources'].get('MovieLens', 0):,}")
        st.markdown(f"**African IMDb:** {health['sources'].get('African_IMDb', 0):,}")
        st.markdown(f"**FAISS:** {'✅' if health.get('faiss_enabled') else '⚠️ TF-IDF fallback'}")
        st.markdown(f"**Gemini AI:** {'✅' if health.get('gemini_enabled') else '⚠️ Not configured'}")
    else:
        st.error("❌ API Offline — Start the backend first")

    st.divider()
    st.markdown("### 👤 User Settings")
    user_id   = st.number_input("Your User ID", min_value=1, max_value=671, value=1, step=1)
    n_results = st.slider("Number of Results", min_value=3, max_value=20, value=10)

    st.divider()
    st.markdown("### 📊 About")
    st.markdown("""
    **Models Used:**
    - 🤖 Hybrid Filtering (CBF + CF)
    - 🔢 SVD Collaborative Filtering
    - 📐 TF-IDF Content Similarity
    - 🧠 FAISS Vector Search
    - ✨ Gemini AI NL Understanding

    **Datasets:**
    - 🌍 Scraped African/Nollywood movies
    - 🎬 MovieLens (Kaggle)
    """)


# ─── MAIN ────────────────────────────────────────────────────────
st.markdown("# 🎬 Movie Recommendation System")
st.markdown("*Personalized recommendations — African & Global cinema powered by Hybrid ML + Gemini AI*")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🎯 Get Recommendations",
    "🔍 Natural Language Search",
    "🌍 Browse Movies"
])


# ── TAB 1: RECOMMENDATIONS ───────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">🎯 Hybrid Movie Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown("Enter a movie title to get personalized recommendations.")

    col1, col2 = st.columns([3, 1])
    with col1:
        movie_title = st.text_input("Movie Title",
            placeholder="e.g. Toy Story, The Lion King, Lionheart...",
            label_visibility="collapsed")
    with col2:
        recommend_btn = st.button("🎬 Recommend", use_container_width=True, type="primary")

    if recommend_btn and movie_title:
        with st.spinner(f'Finding movies similar to "{movie_title}"...'):
            result = get_recommendations(movie_title, user_id=user_id, n=n_results)

        if result:
            st.success(f"✅ {result['total_results']} recommendations for **{result['query']}** | User {result['user_id']}")
            st.divider()
            for i, movie in enumerate(result['recommendations']):
                render_movie_card(movie, score_key='hybrid_score', index=i)
        else:
            st.warning(f'❌ "{movie_title}" not found. Try another title.')
            suggestions = search_movies(movie_title)
            if suggestions and suggestions.get('results'):
                st.markdown("**Did you mean:**")
                for s in suggestions['results'][:3]:
                    st.markdown(f"- {s['title']} ({s.get('genre', '')})")

    elif recommend_btn:
        st.warning("Please enter a movie title.")


# ── TAB 2: NL SEARCH ─────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔍 Natural Language Search</div>',
                unsafe_allow_html=True)
    st.markdown("Describe what you want in plain English — Gemini AI understands you.")

    col1, col2 = st.columns([3, 1])
    with col1:
        nl_query = st.text_input("Search Query",
            placeholder="e.g. funny Nigerian comedy, romantic African drama, sci-fi thriller...",
            label_visibility="collapsed")
    with col2:
        search_btn = st.button("🔍 Search", use_container_width=True, type="primary")

    # Quick example buttons
    st.markdown("**💡 Try these:**")
    ex1, ex2, ex3, ex4 = st.columns(4)
    with ex1:
        if st.button("🇳🇬 Nigerian comedy"):
            nl_query = "funny Nigerian comedy Nollywood"
    with ex2:
        if st.button("🎭 African drama"):
            nl_query = "emotional African drama"
    with ex3:
        if st.button("🚀 Sci-fi adventure"):
            nl_query = "sci-fi space adventure"
    with ex4:
        if st.button("💕 Romance"):
            nl_query = "romantic love story"

    if (search_btn or nl_query) and nl_query:
        with st.spinner(f'Searching for "{nl_query}"...'):
            result = nl_search(nl_query, n=n_results)

        if result and result.get('results'):
            st.success(f"✅ {result['total_results']} results | Method: **{result.get('search_method', 'tfidf').upper()}**")

            # Show Gemini AI explanation
            if result.get('ai_explanation'):
                st.markdown(f"""
                <div class="ai-explanation">
                    ✨ <strong>Gemini AI says:</strong><br>{result['ai_explanation']}
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            for i, movie in enumerate(result['results']):
                render_movie_card(movie, score_key='relevance_score', index=i)
        else:
            st.warning("No results found. Try a different query.")


# ── TAB 3: BROWSE ────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🌍 Browse Movies</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        browse_query = st.text_input("Search by title, genre or country",
                                     placeholder="e.g. Nigeria, Comedy, Action...")
    with col2:
        source_filter = st.selectbox("Filter by Source", ["All", "African_IMDb", "MovieLens"])

    if browse_query:
        with st.spinner("Searching..."):
            results = search_movies(browse_query)

        if results and results.get('results'):
            movies = results['results']
            if source_filter != "All":
                movies = [m for m in movies if m.get('source') == source_filter]

            st.success(f"Found {len(movies)} movies")
            df = pd.DataFrame(movies)[['title', 'genre', 'rating', 'country', 'source']]
            df.columns = ['Title', 'Genre', 'Rating', 'Country', 'Source']
            df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No movies found.")
    else:
        if health:
            st.markdown("### 📊 Dataset Overview")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Movies", f"{health['movies']:,}")
            with c2:
                st.metric("MovieLens", f"{health['sources'].get('MovieLens', 0):,}")
            with c3:
                st.metric("African IMDb", f"{health['sources'].get('African_IMDb', 0):,}")
