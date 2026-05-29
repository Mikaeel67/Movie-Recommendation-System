# 🎬 Movie Recommendation System
### End-to-End AI-Powered Movie Recommender — African & Global Cinema

> **Deep Tech Mentorship Program** | ID: FE/23/15128075 | Alyasau Mikaila

---

## 🌟 Project Overview

A production-ready, end-to-end Movie Recommendation System that combines **Hybrid Machine Learning**, **FAISS Vector Search**, and **Gemini AI** to deliver personalized movie recommendations across African and global cinema.

What makes this project unique:
- 🌍 **Custom African dataset** — scraped directly from IMDb (Nollywood + 10 African countries)
- 🤖 **Hybrid ML model** — Content-Based Filtering + Collaborative Filtering (SVD)
- 🧠 **Gemini AI integration** — Natural language search + AI-generated explanations
- ⚡ **FAISS vector search** — Semantic similarity using sentence-transformers
- 🚀 **Production-grade stack** — FastAPI backend + Streamlit frontend

---

## 📊 Dataset

| Source | Movies | Description |
|---|---|---|
| **African IMDb (scraped)** | ~473 | Nollywood + African cinema from 10 countries |
| **MovieLens (Kaggle)** | ~10,009 | Global movies with 105K+ user ratings |
| **Total** | **~10,482** | Merged, deduplicated, preprocessed |

**African countries covered:** Nigeria, South Africa, Ghana, Kenya, Ethiopia, Egypt, Tanzania, Uganda, Senegal, Cameroon

---

## 🏗️ System Architecture

```
User Query (Natural Language)
        ↓
   Gemini API (parses intent)
        ↓
   FAISS Vector Search (finds similar movies)
        ↓
   Hybrid Recommender (CBF 40% + SVD CF 60%)
        ↓
   Gemini API (explains recommendations)
        ↓
   FastAPI Backend → Streamlit Frontend
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Scraping | BeautifulSoup4 + Selenium |
| Data Processing | Pandas, NumPy |
| ML / Modeling | Scikit-learn (TF-IDF, Cosine Similarity) |
| Collaborative Filtering | Surprise (SVD) |
| Vector Search | FAISS + Sentence-Transformers |
| LLM Integration | Gemini API (Google AI Studio) |
| API Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Dataset | IMDb (scraped) + Kaggle MovieLens |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Mac/Linux
export GEMINI_API_KEY=your_api_key_here
```

### 4. Run the backend
```bash
uvicorn api.main:app --reload
```

### 5. Run the frontend (new terminal)
```bash
streamlit run frontend/app.py
```

### 6. Open the app
- Streamlit UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## 📁 Project Structure

```
movie-recommender/
│
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── african_movies_raw.csv
│   │   ├── movies.csv
│   │   └── ratings.csv
│   └── processed/              # Cleaned & merged data
│       ├── movies_merged.csv
│       └── ratings_clean.csv
│
├── notebooks/
│   └── movie_recommendation_system.ipynb  # Full ML pipeline
│
├── models/                     # Saved model artifacts
│   ├── svd_model.pkl
│   ├── cosine_sim.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── title_to_idx.pkl
│   ├── merged_df.pkl
│   └── faiss.index
│
├── scraper/
│   └── showmax_scraper.py      # IMDb African movies scraper
│
├── api/
│   └── main.py                 # FastAPI backend
│
├── frontend/
│   └── app.py                  # Streamlit frontend
│
├── requirements.txt
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API health + dataset stats |
| POST | `/recommend` | Hybrid movie recommendations |
| POST | `/nl-search` | Natural language search (Gemini + FAISS) |
| GET | `/movies` | Browse all movies (paginated) |
| GET | `/movies/search?q=` | Keyword search |

### Example: Get Recommendations
```json
POST /recommend
{
  "title": "Toy Story",
  "user_id": 1,
  "n": 10
}
```

### Example: Natural Language Search
```json
POST /nl-search
{
  "query": "funny Nigerian comedy with strong female lead",
  "n": 10
}
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| RMSE (SVD) | ~0.87 |
| MAE (SVD) | ~0.67 |
| Precision@10 | ~0.72 |
| Recall@10 | ~0.41 |

---

## 👤 Author

**Alyasau Mikaila**
- Deep Tech ID: FE/23/15128075
- Program: Data Science & Machine Learning Mentorship
- Institution: Abubakar Tafawa Balewa University (ATBU), Bauchi

---

## 📄 License
MIT License
