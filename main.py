from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os

# ─── CONFIGURATION ───────────────────────────────────────────────
# Update this path to match your environment
MODELS_DIR = os.environ.get(
    'MODELS_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Models')
)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE')

# ─── LOAD MODEL ARTIFACTS ────────────────────────────────────────
print("⏳ Loading model artifacts...")

merged_df    = joblib.load(os.path.join(MODELS_DIR, 'merged_df.pkl'))
cosine_sim   = joblib.load(os.path.join(MODELS_DIR, 'cosine_sim.pkl'))
title_to_idx = joblib.load(os.path.join(MODELS_DIR, 'title_to_idx.pkl'))
svd_model    = joblib.load(os.path.join(MODELS_DIR, 'svd_model.pkl'))

# Load FAISS index if available
faiss_available = False
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    faiss_index   = faiss.read_index(os.path.join(MODELS_DIR, 'faiss.index'))
    encoder       = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_available = True
    print("✅ FAISS index loaded!")
except Exception as e:
    print(f"⚠️  FAISS not available, using TF-IDF fallback: {e}")

# Load Gemini if available
gemini_available = False
try:
    import google.generativeai as genai
    if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE':
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_available = True
        print("✅ Gemini API loaded!")
    else:
        print("⚠️  Gemini API key not set, NL search will use TF-IDF fallback")
except Exception as e:
    print(f"⚠️  Gemini not available: {e}")

print(f"✅ Artifacts loaded! Dataset: {len(merged_df):,} movies")

# ─── FASTAPI APP ─────────────────────────────────────────────────
app = FastAPI(
    title="🎬 Movie Recommendation API",
    description="""
    ## Hybrid Movie Recommendation System
    Combining **Content-Based Filtering**, **Collaborative Filtering (SVD)**,
    **FAISS Vector Search**, and **Gemini AI** natural language understanding.

    ### Dataset
    - 🌍 **African IMDb** (scraped from IMDb — Nollywood + African cinema)
    - 🎬 **MovieLens** (Kaggle — global movies + user ratings)
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── SCHEMAS ─────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    title: str
    user_id: Optional[int] = 1
    n: Optional[int] = 10
    cbf_weight: Optional[float] = 0.4
    cf_weight: Optional[float] = 0.6


class NLSearchRequest(BaseModel):
    query: str
    n: Optional[int] = 10


# ─── HELPER: HYBRID RECOMMENDER ──────────────────────────────────
from sklearn.preprocessing import MinMaxScaler

def hybrid_recommend(title: str, user_id: int, n: int = 10,
                     cbf_weight: float = 0.4, cf_weight: float = 0.6):
    title_lower = title.lower().strip()

    if title_lower not in title_to_idx:
        matches = [t for t in title_to_idx.index if title_lower in t]
        if matches:
            title_lower = matches[0]
        else:
            return None, f'Movie "{title}" not found in dataset.'

    idx = title_to_idx[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n*3+1]

    candidate_indices = [i[0] for i in sim_scores]
    candidates = merged_df.iloc[candidate_indices][
        ['movieId', 'title', 'genre', 'rating', 'country', 'source', 'description']
    ].copy()
    candidates['cbf_score'] = [float(s[1]) for s in sim_scores]

    scaler = MinMaxScaler(feature_range=(0, 5))
    candidates['cbf_score_norm'] = scaler.fit_transform(
        candidates[['cbf_score']]
    ).flatten()

    def get_cf_score(row):
        if row['source'] == 'MovieLens':
            return svd_model.predict(user_id, row['movieId']).est
        return float(row['rating']) if pd.notna(row['rating']) else 3.0

    candidates['cf_score']     = candidates.apply(get_cf_score, axis=1)
    candidates['hybrid_score'] = (
        cbf_weight * candidates['cbf_score_norm'] +
        cf_weight  * candidates['cf_score']
    )

    result = candidates.sort_values('hybrid_score', ascending=False).head(n)
    result = result[['movieId', 'title', 'genre', 'rating', 'country',
                     'source', 'description', 'hybrid_score']].copy()
    result['hybrid_score'] = result['hybrid_score'].round(4)
    return result.reset_index(drop=True), None


# ─── HELPER: FAISS VECTOR SEARCH ─────────────────────────────────
def faiss_search(query: str, n: int = 10):
    """Semantic vector search using FAISS + sentence-transformers."""
    query_vector = encoder.encode([query.lower()]).astype('float32')
    distances, indices = faiss_index.search(query_vector, n)

    result = merged_df.iloc[indices[0]][
        ['movieId', 'title', 'genre', 'rating', 'country', 'source', 'description']
    ].copy()
    result['relevance_score'] = (1 / (1 + distances[0])).round(4)
    return result.reset_index(drop=True)


# ─── HELPER: TFIDF FALLBACK SEARCH ───────────────────────────────
def tfidf_search(query: str, n: int = 10):
    """TF-IDF based NL search fallback."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    corpus = merged_df['content'].fillna('').tolist()
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_corpus = vectorizer.fit_transform(corpus)
    query_vec    = vectorizer.transform([query.lower()])

    scores = cos_sim(query_vec, tfidf_corpus).flatten()
    top_indices = scores.argsort()[::-1][:n]

    result = merged_df.iloc[top_indices][
        ['movieId', 'title', 'genre', 'rating', 'country', 'source', 'description']
    ].copy()
    result['relevance_score'] = scores[top_indices].round(4)
    result = result[result['relevance_score'] > 0]
    return result.reset_index(drop=True)


# ─── HELPER: GEMINI NL PARSER ────────────────────────────────────
def parse_query_with_gemini(query: str) -> dict:
    """Use Gemini to parse NL query into structured filters."""
    prompt = f"""
    You are a movie search assistant. Parse the user's natural language movie query 
    into a structured JSON filter object.
    
    User query: "{query}"
    
    Return ONLY a valid JSON object with these fields (use null if not mentioned):
    {{
        "genre": ["list of genres or null"],
        "country": "country name or null",
        "mood": "funny/sad/thrilling/romantic/etc or null",
        "keywords": ["important keywords from the query"],
        "enhanced_query": "rewritten search query optimized for movie search"
    }}
    
    Examples:
    - "funny Nigerian comedy" -> {{"genre": ["Comedy"], "country": "Nigeria", "mood": "funny", "keywords": ["Nigerian", "comedy"], "enhanced_query": "Nigerian comedy funny Nollywood"}}
    - "romantic African drama" -> {{"genre": ["Drama", "Romance"], "country": null, "mood": "romantic", "keywords": ["African", "romantic", "drama"], "enhanced_query": "African romantic drama emotional"}}
    
    Return ONLY the JSON, no explanation.
    """
    try:
        response = gemini_model.generate_content(prompt)
        import json, re
        text = response.text.strip()
        text = re.sub(r'```json|```', '', text).strip()
        return json.loads(text)
    except Exception as e:
        print(f"Gemini parse error: {e}")
        return {"enhanced_query": query, "keywords": [query]}


def generate_gemini_explanation(query: str, movies: list) -> str:
    """Use Gemini to explain why movies were recommended."""
    movie_list = "\n".join([
        f"- {m['title']} ({m.get('genre','')}, {m.get('country','')})"
        for m in movies[:5]
    ])
    prompt = f"""
    A user searched for: "{query}"
    
    These movies were recommended:
    {movie_list}
    
    Write a brief, friendly 2-3 sentence explanation of why these movies match 
    what the user is looking for. Be specific about genres and themes.
    Keep it conversational and encouraging.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except:
        return ""


# ─── ROUTES ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message"    : "🎬 Movie Recommendation API is running!",
        "faiss"      : faiss_available,
        "gemini"     : gemini_available,
        "endpoints"  : {
            "health"   : "GET  /health",
            "recommend": "POST /recommend",
            "nl_search": "POST /nl-search",
            "movies"   : "GET  /movies",
            "search"   : "GET  /movies/search?q=<query>"
        }
    }


@app.get("/health")
def health():
    return {
        "status"         : "ok",
        "movies"         : len(merged_df),
        "sources"        : merged_df['source'].value_counts().to_dict(),
        "faiss_enabled"  : faiss_available,
        "gemini_enabled" : gemini_available
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """Hybrid recommendation — Content-Based + Collaborative Filtering."""
    result, error = hybrid_recommend(
        title      = req.title,
        user_id    = req.user_id,
        n          = req.n,
        cbf_weight = req.cbf_weight,
        cf_weight  = req.cf_weight
    )
    if error:
        raise HTTPException(status_code=404, detail=error)

    return {
        "query"          : req.title,
        "user_id"        : req.user_id,
        "total_results"  : len(result),
        "recommendations": result.to_dict(orient='records')
    }


@app.post("/nl-search")
def natural_language_search(req: NLSearchRequest):
    """
    Natural Language Search endpoint.
    Uses Gemini to parse query → FAISS for semantic search → returns results + AI explanation.
    Falls back to TF-IDF if Gemini/FAISS unavailable.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    explanation = ""
    search_method = "tfidf"

    # Step 1: Parse query with Gemini
    if gemini_available:
        parsed = parse_query_with_gemini(req.query)
        enhanced_query = parsed.get('enhanced_query', req.query)
    else:
        enhanced_query = req.query

    # Step 2: Search using FAISS or TF-IDF
    if faiss_available:
        result = faiss_search(enhanced_query, n=req.n)
        search_method = "faiss"
    else:
        result = tfidf_search(enhanced_query, n=req.n)

    if result.empty:
        raise HTTPException(status_code=404, detail="No matching movies found.")

    # Step 3: Generate Gemini explanation
    if gemini_available:
        explanation = generate_gemini_explanation(
            req.query,
            result.to_dict(orient='records')
        )

    return {
        "query"        : req.query,
        "total_results": len(result),
        "search_method": search_method,
        "ai_explanation": explanation,
        "results"      : result.to_dict(orient='records')
    }


@app.get("/movies")
def list_movies(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, le=100),
    source: Optional[str] = None
):
    """List all movies with pagination. Filter by source: MovieLens or African_IMDb."""
    df = merged_df.copy()
    if source:
        df = df[df['source'].str.lower() == source.lower()]

    total  = len(df)
    start  = (page - 1) * limit
    subset = df.iloc[start:start+limit][
        ['movieId', 'title', 'genre', 'rating', 'country', 'source']
    ]
    return {"total": total, "page": page, "limit": limit,
            "movies": subset.to_dict(orient='records')}


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=1)):
    """Quick keyword search by title, genre or country."""
    q_lower = q.lower()
    mask = (
        merged_df['title'].str.lower().str.contains(q_lower, na=False) |
        merged_df['genre'].str.lower().str.contains(q_lower, na=False) |
        merged_df['country'].str.lower().str.contains(q_lower, na=False)
    )
    results = merged_df[mask][
        ['movieId', 'title', 'genre', 'rating', 'country', 'source']
    ].head(20)
    return {"query": q, "total": len(results),
            "results": results.to_dict(orient='records')}
