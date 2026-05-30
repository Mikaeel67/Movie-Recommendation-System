from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os
import re
import json

# ─── CONFIGURATION ───────────────────────────────────────────────
MODELS_DIR     = '/content/drive/MyDrive/movie_recommender/models'
GEMINI_API_KEY = 'AQ.Ab8RN6IsZdYUPbMBbAwpveSfQ0N_72s-5AJIZP_fge022KfAE'

# ─── LOAD MODEL ARTIFACTS ────────────────────────────────────────
print("⏳ Loading model artifacts...")

merged_df    = joblib.load(os.path.join(MODELS_DIR, 'merged_df.pkl'))
cosine_sim   = joblib.load(os.path.join(MODELS_DIR, 'cosine_sim.pkl'))
title_to_idx = joblib.load(os.path.join(MODELS_DIR, 'title_to_idx.pkl'))
svd_model    = joblib.load(os.path.join(MODELS_DIR, 'svd_model.pkl'))

# ─── LOAD FAISS ──────────────────────────────────────────────────
faiss_available = False
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    faiss_index     = faiss.read_index(os.path.join(MODELS_DIR, 'faiss.index'))
    encoder         = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_available = True
    print("✅ FAISS index loaded!")
except Exception as e:
    print(f"⚠️  FAISS not available, using TF-IDF fallback: {e}")

# ─── LOAD GEMINI ──────────────────────────────────────────────
gemini_available = False
try:
    from google import genai
    from google.genai import types
    client           = genai.Client(api_key=GEMINI_API_KEY)
    gemini_available = True
    print("✅ Gemini API loaded!")
except Exception as e:
    print(f"⚠️  Gemini not available: {e}")

# ─── FASTAPI APP ─────────────────────────────────────────────────
app = FastAPI(
    title="🎬 Movie Recommendation API",
    description="Hybrid Movie Recommender — African + MovieLens dataset with FAISS + Gemini AI",
    version="2.0.0"
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

# ─── HYBRID RECOMMENDER ──────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler

def hybrid_recommend(title, user_id, n=10, cbf_weight=0.4, cf_weight=0.6):
    title_lower = title.lower().strip()
    if title_lower not in title_to_idx:
        matches = [t for t in title_to_idx.index if title_lower in t]
        if matches:
            title_lower = matches[0]
        else:
            return None, f'Movie "{title}" not found.'

    idx        = title_to_idx[title_lower]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n*3+1]
    candidates = merged_df.iloc[[i[0] for i in sim_scores]][
        ['movieId', 'title', 'genre', 'rating', 'country', 'source', 'description']
    ].copy()
    candidates['cbf_score'] = [float(s[1]) for s in sim_scores]

    scaler = MinMaxScaler(feature_range=(0, 5))
    candidates['cbf_score_norm'] = scaler.fit_transform(candidates[['cbf_score']]).flatten()

    def get_cf_score(row):
        if row['source'] == 'MovieLens':
            return svd_model.predict(user_id, row['movieId']).est
        return float(row['rating']) if pd.notna(row['rating']) else 3.0

    candidates['cf_score']     = candidates.apply(get_cf_score, axis=1)
    candidates['hybrid_score'] = (cbf_weight * candidates['cbf_score_norm'] + cf_weight * candidates['cf_score'])
    result = candidates.sort_values('hybrid_score', ascending=False).head(n)
    result = result[['movieId', 'title', 'genre', 'rating', 'country', 'source', 'description', 'hybrid_score']].copy()
    result['hybrid_score'] = result['hybrid_score'].round(4)
    return result.reset_index(drop=True), None

# ─── FAISS SEARCH ────────────────────────────────────────────────
def faiss_search(query, n=10):
    query_vector       = encoder.encode([query.lower()]).astype('float32')
    distances, indices = faiss_index.search(query_vector, n)
    result             = merged_df.iloc[indices[0]][
        ['movieId', 'title', 'genre', 'rating', 'country', 'source', 'description']
    ].copy()
    result['relevance_score'] = (1 / (1 + distances[0])).round(4)
    return result.reset_index(drop=True)

# ─── TFIDF FALLBACK ──────────────────────────────────────────────
def tfidf_search(query, n=10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    corpus       = merged_df['content'].fillna('').tolist()
    vectorizer   = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_corpus = vectorizer.fit_transform(corpus)
    query_vec    = vectorizer.transform([query.lower()])
    scores       = cos_sim(query_vec, tfidf_corpus).flatten()
    top_indices  = scores.argsort()[::-1][:n]
    result       = merged_df.iloc[top_indices][
        ['movieId', 'title', 'genre', 'rating', 'country', 'source', 'description']
    ].copy()
    result['relevance_score'] = scores[top_indices].round(4)
    return result[result['relevance_score'] > 0].reset_index(drop=True)

# ─── GEMINI FUNCTIONS ────────────────────────────────────────────
def parse_query_with_gemini(query):
    prompt = f"""
    Parse this movie search query into a JSON filter object.
    Query: "{query}"
    Return ONLY valid JSON with no markdown or backticks:
    {{
        "genre": ["list of genres"],
        "country": "country name or null",
        "keywords": ["important keywords"],
        "enhanced_query": "rewritten query optimized for movie search"
    }}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt
        )
        text   = re.sub(r'```json|```', '', response.text).strip()
        parsed = json.loads(text)
        print(f"✅ Gemini parsed: {parsed.get('enhanced_query')}")
        return parsed
    except Exception as e:
        print(f"Gemini parse error: {e}")
        return {"enhanced_query": query, "keywords": [query]}


def generate_gemini_explanation(query, movies):
    movie_list = "\n".join([
        f"- {m['title']} ({m.get('genre','')}, {m.get('country','')})"
        for m in movies[:5]
    ])
    prompt = f"""
    A user searched for: "{query}"
    These movies were recommended:
    {movie_list}
    Write a brief 2-3 sentence friendly explanation of why these movies 
    match the user's request. Be specific about genres and themes.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt
        )
        print(f"✅ Gemini explanation generated")
        return response.text.strip()
    except Exception as e:
        print(f"Gemini explanation error: {e}")
        return ""

# ─── ROUTES ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "🎬 Movie Recommendation API is running!",
        "faiss"  : faiss_available,
        "gemini" : gemini_available,
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
    result, error = hybrid_recommend(req.title, req.user_id, req.n, req.cbf_weight, req.cf_weight)
    if error:
        raise HTTPException(status_code=404, detail=error)
    return {"query": req.title, "user_id": req.user_id, "total_results": len(result), "recommendations": result.to_dict(orient='records')}

@app.post("/nl-search")
def natural_language_search(req: NLSearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    explanation   = ""
    search_method = "tfidf"

    # Parse with Gemini
    if gemini_available:
        parsed         = parse_query_with_gemini(req.query)
        enhanced_query = parsed.get('enhanced_query', req.query)
    else:
        enhanced_query = req.query

    # Search with FAISS or TF-IDF
    if faiss_available:
        result        = faiss_search(enhanced_query, n=req.n)
        search_method = "faiss"
    else:
        result = tfidf_search(enhanced_query, n=req.n)

    if result.empty:
        raise HTTPException(status_code=404, detail="No matching movies found.")

    # Gemini explanation
    if gemini_available:
        explanation = generate_gemini_explanation(req.query, result.to_dict(orient='records'))

    return {
        "query"         : req.query,
        "total_results" : len(result),
        "search_method" : search_method,
        "ai_explanation": explanation,
        "results"       : result.to_dict(orient='records')
    }

@app.get("/movies")
def list_movies(page: int = Query(default=1, ge=1), limit: int = Query(default=20, le=100), source: Optional[str] = None):
    df     = merged_df.copy()
    if source:
        df = df[df['source'].str.lower() == source.lower()]
    total  = len(df)
    subset = df.iloc[(page-1)*limit : page*limit][['movieId', 'title', 'genre', 'rating', 'country', 'source']]
    return {"total": total, "page": page, "limit": limit, "movies": subset.to_dict(orient='records')}

@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=1)):
    q_lower = q.lower()
    mask    = (
        merged_df['title'].str.lower().str.contains(q_lower, na=False) |
        merged_df['genre'].str.lower().str.contains(q_lower, na=False) |
        merged_df['country'].str.lower().str.contains(q_lower, na=False)
    )
    results = merged_df[mask][['movieId', 'title', 'genre', 'rating', 'country', 'source']].head(20)
    return {"query": q, "total": len(results), "results": results.to_dict(orient='records')}
