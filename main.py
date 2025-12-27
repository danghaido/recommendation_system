# main.py
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chatbot import (
    DEFAULT_MODEL_CONFIG,
    create_context_from_movies,
    create_prompt,
    query_llm_model,
    retrieve_relevant_movies,
)
from evaluation import MovieRecommenderEvaluator, display_evaluation_info

# Import ChromaDB helper functions
try:
    from chromadb_helper import (
        get_chromadb_client,
        get_chromadb_collection,
        get_movie_embedding_by_index,
        search_similar_movies_by_embedding,
        search_similar_movies_by_text,
    )
    from chromadb_helper import (
        load_sentence_model as load_model_chromadb,
    )

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    st.warning("ChromaDB helper not available. Install chromadb package or use local embeddings.")

DATA_PATH = "csv/processed_data.csv"
USE_CHROMADB = True


@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    # basic clean
    df = df.fillna({"overview": "", "genres": "", "cast": "", "director": "", "release_date": ""})

    # parse genres_list từ string về list (nếu cần)
    import ast

    if "genres_list" in df.columns:
        df["genres_list"] = df["genres_list"].apply(
            lambda x: ast.literal_eval(x)
            if isinstance(x, str) and x.startswith("[")
            else (x if isinstance(x, list) else [])
        )

    # Extract year from release_date
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype(str)

    # Tạo combined text ĐẦY ĐỦ cho embedding (dùng chung cho cả recommendation và chatbot)
    # Bao gồm: Title, Year, Director (x2 để tăng trọng số), Genres, Cast, Rating, Overview
    df["combined"] = (
        "Title: "
        + df["title"].astype(str)
        + ". Year: "
        + df["year"].astype(str)
        + ". Director: "
        + df["director"].astype(str)
        + ". Director: "
        + df["director"].astype(str)  # Lặp lại để boost trọng số
        + ". Genres: "
        + df["genres"].astype(str)
        + ". Cast: "
        + df["cast"].astype(str)
        + ". Rating: "
        + df["vote_average"].astype(str)
        + "/10. Overview: "
        + df["overview"].astype(str)
    )
    return df


@st.cache_resource(show_spinner=False)
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def compute_embeddings(combined_texts, _model):
    """Compute embeddings once for both recommendation system and chatbot"""
    print(f"Computing embeddings for {len(combined_texts)} movies...")
    embeddings = _model.encode(combined_texts, show_progress_bar=False)
    print("Embeddings computed and cached!")
    return embeddings


@st.cache_data(show_spinner=False)
def fit_tfidf_on_cast_director(corpus, max_features=5000):
    # fit on "cast" + "director" strings (fast and useful for name-based similarity)
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    mat = tfidf.fit_transform(corpus)
    return tfidf, mat


def cos_sim_vec(a, M):
    return cosine_similarity(a.reshape(1, -1), M).flatten()


# ---------- Load everything ----------
st.set_page_config(layout="wide", page_title="🎬 Movie Recommender", page_icon="🎬", initial_sidebar_state="collapsed")


# Load CSS from external file
def load_css():
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("styles.css not found. Using default styling.")


load_css()

st.title("🎬 Movie Recommender System")

# Load data
df = load_data(DATA_PATH)

# Try ChromaDB first, fallback to local embeddings
chromadb_client = None
chromadb_collection = None
embeddings = None
model = None

if USE_CHROMADB and CHROMADB_AVAILABLE:
    st.info("🔄 Attempting to connect to ChromaDB Cloud...")
    try:
        chromadb_client = get_chromadb_client()
        if chromadb_client is not None:
            collection_name = st.secrets.get("chromadb", {}).get("collection_name", "movie_embeddings")
            chromadb_collection = get_chromadb_collection(chromadb_client, collection_name)
            model = load_model_chromadb()

            if chromadb_collection is not None:
                st.success("✅ Connected to ChromaDB Cloud successfully!")
            else:
                st.warning("⚠️ ChromaDB collection not found. Falling back to local embeddings...")
                chromadb_client = None
        else:
            st.warning("⚠️ ChromaDB connection failed. Falling back to local embeddings...")
    except Exception as e:
        st.warning(f"⚠️ ChromaDB error: {e}. Falling back to local embeddings...")
        chromadb_client = None
        chromadb_collection = None

# Load local embeddings if ChromaDB not available
if chromadb_collection is None:
    if model is None:
        model = load_sentence_model()
    with st.spinner("⏳ Computing embeddings for recommendation system... (this may take 30-60 seconds on first run)"):
        embeddings = compute_embeddings(df["combined"].tolist(), model)

cast_director_corpus = (df["cast"].astype(str) + " " + df["director"].astype(str)).tolist()
tfidf_vectorizer, tfidf_matrix = fit_tfidf_on_cast_director(cast_director_corpus)

indices = pd.Series(df.index, index=df["title"].astype(str)).drop_duplicates()

# ---------- Session state: search history ----------
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []

if "search_results" not in st.session_state:
    st.session_state["search_results"] = None

if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

if "filter_results" not in st.session_state:
    st.session_state["filter_results"] = None

if "similar_results" not in st.session_state:
    st.session_state["similar_results"] = None

if "history_recommendations" not in st.session_state:
    st.session_state["history_recommendations"] = None

if "context_results" not in st.session_state:
    st.session_state["context_results"] = None

if "liked_movies" not in st.session_state:
    st.session_state["liked_movies"] = []

if "liked_recommendations" not in st.session_state:
    st.session_state["liked_recommendations"] = None


def push_history(q):
    hist = st.session_state["search_history"]
    if q in hist:
        hist.remove(q)
    hist.insert(0, q)
    # keep up to 5
    st.session_state["search_history"] = hist[:5]
    # Reset history recommendations khi có search mới
    st.session_state["history_recommendations"] = None


# ---------- Recommendation functions with ChromaDB and local fallback ----------


def map_titles_to_indices(titles):
    """Map list of titles to DataFrame indices"""
    result_indices = []
    for title in titles:
        if title in indices.index:
            idx_value = indices[title]
            # Ensure it's a scalar (not Series)
            if hasattr(idx_value, "iloc"):
                # It's a Series, take first value
                idx_value = idx_value.iloc[0]
            # Convert to Python int
            result_indices.append(int(idx_value))
    return result_indices


def recommend_by_title(title, topn=6):
    """Recommend movies similar to a given title"""
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]

    # Try ChromaDB first
    if chromadb_collection is not None:
        try:
            movie_embedding = get_movie_embedding_by_index(chromadb_collection, idx)
            if movie_embedding is not None:
                similar_titles = search_similar_movies_by_embedding(
                    chromadb_collection, movie_embedding, top_k=min(100, topn * 3), excluded_indices=[idx]
                )
                # Map titles to indices
                similar_indices = map_titles_to_indices(similar_titles)
                similar_indices = [i for i in similar_indices if i != idx][:topn]
                if similar_indices:
                    result_df = df.iloc[similar_indices][
                        ["title", "genres", "cast", "director", "vote_average", "overview"]
                    ]
                    return result_df.sort_values("vote_average", ascending=False).head(topn)
        except Exception as e:
            st.warning(f"ChromaDB search failed, using local fallback: {e}")

    # Fallback to local embeddings
    if embeddings is not None:
        emb = embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(emb, embeddings).flatten()
        top_idx = np.argsort(sims)[::-1][1 : topn + 1]
        return df.iloc[top_idx][["title", "genres", "cast", "director", "vote_average", "overview"]]

    return pd.DataFrame()


def recommend_by_director(director_name, topn=6):
    dir_vec = tfidf_vectorizer.transform([director_name])
    sims = cosine_similarity(dir_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:topn]
    return df.iloc[top_idx][["title", "genres", "cast", "director", "vote_average", "overview"]]


def recommend_by_genre(genre_name, topn=6):
    """Recommend movies by genre"""
    mask = df["genres_list"].apply(lambda x: genre_name in x if isinstance(x, list) else False)
    if mask.sum() == 0:
        return pd.DataFrame()

    # Try ChromaDB first
    if chromadb_collection is not None:
        try:
            query_text = f"Movies with genre {genre_name}. {genre_name} films and cinema."
            similar_titles = search_similar_movies_by_text(
                chromadb_collection, query_text, model, top_k=min(100, topn * 5)
            )
            # Map titles to indices
            similar_indices = map_titles_to_indices(similar_titles)
            if similar_indices:
                result_df = df.iloc[similar_indices]
                result_df = result_df[
                    result_df["genres_list"].apply(lambda x: genre_name in x if isinstance(x, list) else False)
                ]
                if not result_df.empty:
                    result_df = result_df.sort_values("vote_average", ascending=False)
                    return result_df[["title", "genres", "cast", "director", "vote_average", "overview"]].head(topn)
        except Exception as e:
            st.warning(f"ChromaDB search failed, using local fallback: {e}")

    # Fallback: return highest rated movies of this genre
    filtered = df[mask].sort_values("vote_average", ascending=False).head(topn)
    return filtered[["title", "genres", "cast", "director", "vote_average", "overview"]]


def recommend_by_filters(director_name=None, genre_name=None, topn=8):
    """
    Gợi ý phim theo đạo diễn và/hoặc thể loại.
    Có thể chọn 1 trong 2 hoặc cả 2.
    """
    mask = pd.Series([True] * len(df), index=df.index)

    # Filter by director
    if director_name:
        mask = mask & (df["director"] == director_name)

    # Filter by genre
    if genre_name:
        genre_mask = df["genres_list"].apply(lambda x: genre_name in x if isinstance(x, list) else False)
        mask = mask & genre_mask

    if mask.sum() == 0:
        return pd.DataFrame()

    # Get filtered movies and sort by vote_average
    filtered_df = df[mask].sort_values("vote_average", ascending=False).head(topn)
    return filtered_df[["title", "genres", "cast", "director", "vote_average", "overview"]]


def recommend_by_search_history(history_list, topn=7):
    """
    Gợi ý phim dựa trên lịch sử tìm kiếm.
    """
    if not history_list:
        return pd.DataFrame()

    # Try ChromaDB first
    if chromadb_collection is not None:
        try:
            query_text = "Movies similar to: " + ", ".join(history_list) + ". Recommendations based on these titles."
            similar_titles = search_similar_movies_by_text(
                chromadb_collection, query_text, model, top_k=min(100, topn * 3)
            )
            # Map titles to indices
            similar_indices = map_titles_to_indices(similar_titles)
            if similar_indices:
                result_df = df.iloc[similar_indices][
                    ["title", "genres", "cast", "director", "vote_average", "overview"]
                ]
                result_df = result_df.sort_values("vote_average", ascending=False).head(topn)
                return result_df
        except Exception as e:
            st.warning(f"ChromaDB search failed, using local fallback: {e}")

    # Fallback to local embeddings
    if embeddings is not None:
        query_emb = model.encode([" ".join(history_list)])
        sims = cosine_similarity(query_emb, embeddings).flatten()
        top_idx = np.argsort(sims)[::-1][:topn]
        return df.iloc[top_idx][["title", "genres", "cast", "director", "vote_average", "overview"]]

    return pd.DataFrame()


def recommend_by_liked_movies(liked_list, topn=10):
    """
    Gợi ý phim dựa trên danh sách phim đã thích.
    """
    if not liked_list:
        return pd.DataFrame()

    liked_indices = []
    for movie_title in liked_list:
        if movie_title in indices:
            liked_indices.append(indices[movie_title])

    if not liked_indices:
        return pd.DataFrame()

    # Try ChromaDB first
    if chromadb_collection is not None:
        try:
            liked_embeddings = []
            for idx in liked_indices:
                emb = get_movie_embedding_by_index(chromadb_collection, idx)
                if emb is not None:
                    # Ensure it's a 1D numpy array
                    if isinstance(emb, np.ndarray):
                        liked_embeddings.append(emb.flatten())

            if not liked_embeddings:
                query_text = "Movies similar to: " + ", ".join(liked_list)
                similar_titles = search_similar_movies_by_text(
                    chromadb_collection, query_text, model, top_k=min(150, topn * 5)
                )
                # Map titles to indices
                similar_indices = map_titles_to_indices(similar_titles)
                similar_indices = [i for i in similar_indices if i not in liked_indices][:topn]
                if similar_indices:
                    result_df = df.iloc[similar_indices][
                        ["title", "genres", "cast", "director", "vote_average", "overview"]
                    ]
                    return result_df.sort_values("vote_average", ascending=False).head(topn)
            else:
                # Stack embeddings and compute mean
                avg_embedding = np.mean(np.stack(liked_embeddings), axis=0)
                similar_titles = search_similar_movies_by_embedding(
                    chromadb_collection,
                    avg_embedding,
                    top_k=min(200, topn * 5 + len(liked_indices)),
                    excluded_indices=liked_indices,
                )
                # Map titles to indices
                similar_indices = map_titles_to_indices(similar_titles)
                similar_indices = [i for i in similar_indices if i not in liked_indices][:topn]
                if similar_indices:
                    result_df = df.iloc[similar_indices][
                        ["title", "genres", "cast", "director", "vote_average", "overview"]
                    ]
                    return result_df.sort_values("vote_average", ascending=False).head(topn)
        except Exception as e:
            st.warning(f"ChromaDB search failed, using local fallback: {e}")

    # Fallback to local embeddings
    if embeddings is not None:
        avg_emb = np.mean(embeddings[liked_indices], axis=0).reshape(1, -1)
        sims = cosine_similarity(avg_emb, embeddings).flatten()
        # Exclude liked movies
        for idx in liked_indices:
            sims[idx] = -1
        top_idx = np.argsort(sims)[::-1][:topn]
        return df.iloc[top_idx][["title", "genres", "cast", "director", "vote_average", "overview"]]

    return pd.DataFrame()


def recommend_by_context(title, context_time=None, context_mood=None, topn=10):
    """
    Gợi ý phim dựa trên context (thời gian, tâm trạng)

    Args:
        title: Tên phim gốc
        context_time: "Morning", "Afternoon", "Evening", "Night"
        context_mood: "Relaxing", "Exciting", "Romantic", "Thoughtful"
        topn: Số phim gợi ý
    """
    # Lấy gợi ý cơ bản
    base_recs = recommend_by_title(title, topn=30)

    if base_recs.empty:
        return base_recs

    # Filter theo thời gian
    if context_time:
        if context_time == "Morning":
            # Sáng: Comedy, Family, Animation
            base_recs = base_recs[base_recs["genres"].str.contains("Comedy|Family|Animation", na=False, case=False)]
        elif context_time == "Afternoon":
            # Chiều: Adventure, Action, Documentary
            base_recs = base_recs[
                base_recs["genres"].str.contains("Adventure|Action|Documentary", na=False, case=False)
            ]
        elif context_time == "Evening":
            # Tối: Drama, Romance, Mystery
            base_recs = base_recs[base_recs["genres"].str.contains("Drama|Romance|Mystery", na=False, case=False)]
        elif context_time == "Night":
            # Đêm: Horror, Thriller, Crime
            base_recs = base_recs[base_recs["genres"].str.contains("Horror|Thriller|Crime", na=False, case=False)]

    # Filter theo tâm trạng
    if context_mood:
        if context_mood == "Relaxing":
            # Thư giãn: Comedy, Romance, Family
            base_recs = base_recs[base_recs["genres"].str.contains("Comedy|Romance|Family", na=False, case=False)]
        elif context_mood == "Exciting":
            # Kích động: Action, Adventure, Science Fiction
            base_recs = base_recs[
                base_recs["genres"].str.contains("Action|Adventure|Science Fiction", na=False, case=False)
            ]
        elif context_mood == "Romantic":
            # Lãng mạn: Romance, Drama
            base_recs = base_recs[base_recs["genres"].str.contains("Romance|Drama", na=False, case=False)]
        elif context_mood == "Thoughtful":
            # Suy ngẫm: Drama, Documentary, Mystery
            base_recs = base_recs[base_recs["genres"].str.contains("Drama|Documentary|Mystery", na=False, case=False)]

    # Ưu tiên phim có rating cao
    base_recs = base_recs.sort_values("vote_average", ascending=False)

    return base_recs.head(topn)


def display_movie_list(movie_df, key_prefix="movie", show_like_button=True):
    """Hiển thị danh sách phim với nút Mô tả và nút Like - Enhanced UI."""
    for movie_idx, (idx, row) in enumerate(movie_df.iterrows()):
        is_liked = row["title"] in st.session_state["liked_movies"]

        st.markdown(
            f"""
        <div class="movie-card">
            <h3>🎬 {row["title"]}</h3>
            <p class="movie-genre"><span>Genre:</span> {row["genres"]}</p>
            <p class="movie-rating">⭐ {row["vote_average"]}/10</p>
            <p class="movie-cast"><strong>🎭 Cast:</strong> {row["cast"][:100]}{"..." if len(str(row["cast"])) > 100 else ""}</p>
            <p class="movie-director"><strong>🎥 Director:</strong> {row["director"]}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            btn_key = f"{key_prefix}_{movie_idx}"
            if st.button("📖 Description", key=btn_key, use_container_width=True):
                st.session_state[f"show_overview_{btn_key}"] = not st.session_state.get(
                    f"show_overview_{btn_key}", False
                )

        with col2:
            if show_like_button:
                like_btn_key = f"like_{key_prefix}_{movie_idx}"
                if is_liked:
                    if st.button("💔 Unlike", key=like_btn_key, use_container_width=True):
                        st.session_state["liked_movies"].remove(row["title"])
                        st.session_state["liked_recommendations"] = None
                        st.rerun()
                else:
                    if st.button("❤️ Like", key=like_btn_key, use_container_width=True):
                        if row["title"] not in st.session_state["liked_movies"]:
                            st.session_state["liked_movies"].append(row["title"])
                            st.session_state["liked_recommendations"] = None
                        st.rerun()

        if st.session_state.get(f"show_overview_{btn_key}", False):
            st.markdown(
                f"""
            <div class="movie-overview">
                <p><strong style="color: #667eea;">📝 Summary:</strong><br>{row["overview"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )


def display_movie_cards(movie_df, key_prefix="card", show_like_button=True):
    """Hiển thị danh sách phim dạng card đẹp hơn cho gợi ý lịch sử."""
    cols_per_row = 3
    rows = len(movie_df) // cols_per_row + (1 if len(movie_df) % cols_per_row != 0 else 0)
    movie_list = list(movie_df.iterrows())

    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            movie_idx = row_idx * cols_per_row + col_idx
            if movie_idx < len(movie_list):
                idx, row = movie_list[movie_idx]
                is_liked = row["title"] in st.session_state["liked_movies"]

                with cols[col_idx]:
                    st.markdown(
                        f"""
                    <div class="movie-card-grid">
                        <h4>🎬 {row["title"]}</h4>
                        <p style="margin: 8px 0; font-size: 14px; color: #4a5568;">
                            <span style="font-weight: 600; color: #667eea;">Genre:</span> {row["genres"][:50]}{"..." if len(str(row["genres"])) > 50 else ""}
                        </p>
                        <p style="margin: 8px 0; font-size: 15px; font-weight: 600; color: #e97b20;">
                            ⭐ {row["vote_average"]}/10
                        </p>
                        <p style="margin: 8px 0; font-size: 13px; color: #718096;">
                            <span style="font-weight: 600; color: #2d3748;">🎥 Director:</span> {row["director"][:30]}{"..." if len(str(row["director"])) > 30 else ""}
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        btn_key = f"{key_prefix}_{movie_idx}"
                        if st.button("📖 Description", key=btn_key, use_container_width=True):
                            st.session_state[f"show_overview_{btn_key}"] = not st.session_state.get(
                                f"show_overview_{btn_key}", False
                            )

                    with btn_col2:
                        if show_like_button:
                            like_btn_key = f"like_{key_prefix}_{movie_idx}"
                            if is_liked:
                                if st.button("💔 Unlike", key=like_btn_key, use_container_width=True):
                                    st.session_state["liked_movies"].remove(row["title"])
                                    st.session_state["liked_recommendations"] = None
                                    st.rerun()
                            else:
                                if st.button("❤️ Like", key=like_btn_key, use_container_width=True):
                                    if row["title"] not in st.session_state["liked_movies"]:
                                        st.session_state["liked_movies"].append(row["title"])
                                        st.session_state["liked_recommendations"] = None
                                    st.rerun()

                    if st.session_state.get(f"show_overview_{btn_key}", False):
                        st.info(f"**Summary:** {row['overview']}")


# ---------- UI: tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Search", "Statistics", "Evaluation", "Chatbot"])

with tab1:
    st.markdown(
        '<div class="page-header"><h1>🏠 Home — Quick Recommendations</h1><p>Discover your next favorite movie with our smart recommendation system</p></div>',
        unsafe_allow_html=True,
    )

    # Top directors & genres list
    top_directors = df["director"].value_counts().head(50).index.tolist()
    top_genres = df["genres_list"].dropna().explode().value_counts().head(30).index.tolist()

    st.subheader("🎬 Filter by Director and/or Genre")
    st.write("You can choose one or both criteria to filter")

    col1, col2 = st.columns(2)

    with col1:
        dir_sel = st.selectbox("Director (optional)", ["--- None ---"] + top_directors, index=0)

    with col2:
        genre_sel = st.selectbox("Genre (optional)", ["--- None ---"] + top_genres, index=0)

    if st.button("🔍 Find Movies", type="primary"):
        selected_dir = None if dir_sel == "--- None ---" else dir_sel
        selected_genre = None if genre_sel == "--- None ---" else genre_sel

        if not selected_dir and not selected_genre:
            st.warning("⚠️ Please select at least one criteria (director or genre)")
        else:
            res = recommend_by_filters(selected_dir, selected_genre, topn=10)
            st.session_state["filter_results"] = res

    # Display filter results
    if st.session_state["filter_results"] is not None:
        res = st.session_state["filter_results"]
        if res.empty:
            st.warning("❌ No matching movies found")
        else:
            st.success(f"✅ Found {len(res)} movies")
            display_movie_list(res, key_prefix="filter")

    st.markdown("---")
    st.subheader("🎥 Quick Search by Movie & Similar Recommendations")
    sel_title = st.selectbox("Select a movie (search by title)", [""] + df["title"].head(500).tolist())
    if st.button("💡 Recommend Similar Movies"):
        if sel_title:
            res = recommend_by_title(sel_title, topn=8)
            st.session_state["similar_results"] = res
            st.session_state["similar_title"] = sel_title
        else:
            st.warning("⚠️ Please select a movie.")

    # Display similar movies results
    if st.session_state["similar_results"] is not None:
        res = st.session_state["similar_results"]
        st.success(f"✅ Similar movies to **{st.session_state.get('similar_title', '')}**")
        display_movie_list(res, key_prefix="similar")

    st.markdown("---")
    st.subheader("🌟 Context-Aware Recommendations (Time & Mood)")
    st.write("Get movie recommendations based on your current time and mood")

    col1, col2, col3 = st.columns(3)

    with col1:
        context_movie = st.selectbox(
            "Select base movie", [""] + df["title"].head(500).tolist(), key="context_movie_select"
        )

    with col2:
        context_time = st.selectbox("⏰ Time of Day", ["--- None ---", "Morning", "Afternoon", "Evening", "Night"])

    with col3:
        context_mood = st.selectbox("😊 Mood", ["--- None ---", "Relaxing", "Exciting", "Romantic", "Thoughtful"])

    if st.button("🎯 Get Context-Aware Recommendations", type="primary"):
        if context_movie:
            selected_time = None if context_time == "--- None ---" else context_time
            selected_mood = None if context_mood == "--- None ---" else context_mood

            res = recommend_by_context(context_movie, selected_time, selected_mood, topn=8)
            st.session_state["context_results"] = res
            st.session_state["context_params"] = {"movie": context_movie, "time": selected_time, "mood": selected_mood}
        else:
            st.warning("⚠️ Please select a base movie.")

    # Display context-aware results
    if st.session_state.get("context_results") is not None:
        res = st.session_state["context_results"]
        params = st.session_state.get("context_params", {})

        context_info = []
        if params.get("time"):
            context_info.append(f"⏰ {params['time']}")
        if params.get("mood"):
            context_info.append(f"😊 {params['mood']}")

        context_str = " | ".join(context_info) if context_info else "No context"

        if res.empty:
            st.warning(f"❌ No movies found matching context: {context_str}")
        else:
            st.success(f"✅ Recommendations for **{params.get('movie', '')}** - Context: {context_str}")
            display_movie_list(res, key_prefix="context")

with tab2:
    st.markdown(
        '<div class="page-header"><h1>🔍 Search Movies</h1><p>Find your favorite movies and track your search history</p></div>',
        unsafe_allow_html=True,
    )

    # Search input
    q = st.text_input("🔎 Search for a movie title", placeholder="Enter movie title...", key="movie_search")

    if q:
        # Normalize query and data for search
        q_normalized = (
            (
                q.lower()
                .str.replace(" ", "", regex=False)
                .str.replace("-", "", regex=False)
                .str.replace(":", "", regex=False)
            )
            if hasattr(q, "str")
            else q.lower().replace(" ", "").replace("-", "").replace(":", "")
        )

        df_normalized = (
            df["title"]
            .str.lower()
            .str.replace(" ", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.replace(":", "", regex=False)
        )
        mask = df_normalized.str.contains(q_normalized, na=False)
        res = df[mask].head(50)

        # Lưu kết quả vào session state
        st.session_state["search_results"] = res
        st.session_state["last_query"] = q

        if len(res) > 0:
            # push to history
            push_history(q)
        else:
            st.session_state["search_results"] = None

    # Display search results from session state
    if st.session_state["search_results"] is not None and len(st.session_state["search_results"]) > 0:
        res = st.session_state["search_results"]
        q = st.session_state["last_query"]

        st.success(f"✅ Search results for: **{q}** — {len(res)} results (max 50 displayed)")
        display_movie_list(res, key_prefix="search")
    elif st.session_state.get("last_query") and st.session_state["search_results"] is not None:
        st.warning(f"❌ No movies found with keyword: **{st.session_state['last_query']}**")

    st.markdown("---")

    # Liked Movies Section
    st.subheader("❤️ Your Liked Movies")
    if st.session_state["liked_movies"]:
        st.success(f"You have liked {len(st.session_state['liked_movies'])} movies")

        # Display liked movies
        with st.expander("📋 View Liked Movies", expanded=False):
            for idx, movie_title in enumerate(st.session_state["liked_movies"], 1):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"{idx}. {movie_title}")
                with col2:
                    if st.button("🗑️", key=f"remove_liked_{idx}", help="Remove from liked"):
                        st.session_state["liked_movies"].remove(movie_title)
                        st.session_state["liked_recommendations"] = None
                        st.rerun()

        st.markdown("---")
        st.subheader("💖 Because you liked these movies, you might also enjoy:")

        if st.session_state["liked_recommendations"] is None:
            with st.spinner("Analyzing your preferences..."):
                recommendations = recommend_by_liked_movies(st.session_state["liked_movies"], topn=9)
                st.session_state["liked_recommendations"] = recommendations
        else:
            recommendations = st.session_state["liked_recommendations"]

        if not recommendations.empty:
            display_movie_cards(recommendations, key_prefix="liked_recs")
        else:
            st.info("Not enough data to recommend suitable movies")
    else:
        st.info("You haven't liked any movies yet. Start liking movies to get personalized recommendations!")

    st.markdown("---")
    st.subheader("📜 Search History (5 most recent)")
    if st.session_state["search_history"]:
        for idx, h in enumerate(st.session_state["search_history"], 1):
            st.write(f"{idx}. {h}")

        # Recommendations based on history
        st.markdown("---")
        st.subheader("💡 Based on your search history, you might like these movies:")

        # Calculate recommendations if not yet available or history changed
        if st.session_state["history_recommendations"] is None:
            with st.spinner("Analyzing your search history..."):
                recommendations = recommend_by_search_history(st.session_state["search_history"], topn=9)
                st.session_state["history_recommendations"] = recommendations
        else:
            recommendations = st.session_state["history_recommendations"]

        if not recommendations.empty:
            display_movie_cards(recommendations, key_prefix="history")
        else:
            st.info("Not enough data to recommend suitable movies")
    else:
        st.info("No search history yet")

with tab3:
    st.markdown(
        '<div class="page-header"><h1>📊 Data Statistics & Visualizations</h1><p>Explore comprehensive analytics and insights from our movie database</p></div>',
        unsafe_allow_html=True,
    )

    st.write("Explore various statistical visualizations of the movie dataset")

    # Section 1: Static Images
    st.subheader("📈 Static Charts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribution of Movie Ratings**")
        try:
            st.image("graph/Histogram.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Histogram.png")

    with col2:
        st.markdown("**Total Revenue by Genre**")
        try:
            st.image("graph/bar_chart_revenue_by_genre.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: bar_chart_revenue_by_genre.png")

    st.markdown("---")

    st.markdown("**Heatmap - Genre Distribution by Year (2010-2020)**")
    try:
        st.image("graph/Heatmap_movie_category.png", use_container_width=True)
    except Exception:
        st.warning("Image not found: Heatmap_movie_category.png")

    st.markdown("---")

    # Section 2: Interactive Plotly Charts
    st.subheader("📊 Revenue & Budget")

    tab_chart1, tab_chart2 = st.tabs(["Revenue & Budget Over Time", "Cumulative Revenue"])

    with tab_chart1:
        st.markdown("**Revenue & Budget Comparison Over Time**")
        try:
            with open("graph/revenue_budget_over_time.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
        except FileNotFoundError:
            st.error("File not found: revenue_budget_over_time.html")
            st.info("Run `python line_chart_plotly.py` to generate this chart")

    with tab_chart2:
        st.markdown("**Cumulative Revenue Over Time**")
        try:
            with open("graph/cumulative_revenue_over_time.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
        except FileNotFoundError:
            st.error("File not found: cumulative_revenue_over_time.html")
            st.info("Run `python line_chart_plotly.py` to generate this chart")

    st.markdown("---")

    # Dataset Info
    st.subheader("ℹ️ Dataset Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Movies", f"{len(df):,}")
    with col2:
        st.metric("Unique Directors", f"{df['director'].nunique():,}")
    with col3:
        st.metric("Unique Genres", f"{df['genres_list'].explode().nunique():,}")

with tab4:
    st.markdown(
        '<div class="page-header"><h1>📊 System Evaluation</h1><p>Measure and analyze recommendation system performance</p></div>',
        unsafe_allow_html=True,
    )

    display_evaluation_info()
    st.markdown("---")

    st.subheader("⚙️ Settings")
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Number of test movies", 10, 200, 50, 10)
    with col2:
        k = st.slider("Top-K", 5, 20, 10, 1)

    if st.button("▶️ Run Evaluation", type="primary"):
        evaluator = MovieRecommenderEvaluator(df, recommend_by_title)
        results = evaluator.run_full_evaluation(n_samples=n_samples, k=k)

        # Save results
        st.session_state["eval_results"] = results

        # Export CSV
        if st.button("💾 Export Results to CSV"):
            result_df = pd.DataFrame([{**results["content"], **results["diversity"]}])
            result_df.to_csv("evaluation_results.csv", index=False)
            st.success("✅ Saved to evaluation_results.csv")

with tab5:
    st.markdown(
        '<div class="page-header"><h1>🤖 AI Movie Assistant</h1><p>Chat with our AI to get personalized movie recommendations and insights</p></div>',
        unsafe_allow_html=True,
    )

    # Initialize chatbot session state
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if "chat_api_key_verified" not in st.session_state:
        st.session_state["chat_api_key_verified"] = False

    if "chat_api_key" not in st.session_state:
        st.session_state["chat_api_key"] = ""

    if "chat_embeddings" not in st.session_state:
        st.session_state["chat_embeddings"] = None

    if "chat_model_provider" not in st.session_state:
        st.session_state["chat_model_provider"] = "huggingface"

    if "chat_model_name" not in st.session_state:
        st.session_state["chat_model_name"] = DEFAULT_MODEL_CONFIG["huggingface"]["model_name"]

    if "chat_base_url" not in st.session_state:
        st.session_state["chat_base_url"] = ""

    # Sidebar configuration in expander
    with st.expander("⚙️ API Configuration", expanded=not st.session_state["chat_api_key_verified"]):
        if not st.session_state["chat_api_key_verified"]:
            st.warning("🔑 Configure your LLM provider to start chatting")

            # Model provider selection
            provider = st.selectbox(
                "Select Provider",
                ["huggingface", "openai", "gemini", "custom"],
                help="Choose your LLM provider",
                key="provider_select",
            )

            # Model name input
            default_model = DEFAULT_MODEL_CONFIG[provider]["model_name"]
            model_name_input = st.text_input(
                "Model Name",
                value=default_model,
                help="e.g., gpt-3.5-turbo, gemini/gemini-pro, huggingface/model-name",
                key="model_name_input",
            )

            # Base URL (only for custom)
            base_url_input = ""
            if provider == "custom":
                base_url_input = st.text_input(
                    "Base URL",
                    value="http://localhost:8000/v1",
                    help="Your self-hosted model endpoint",
                    key="base_url_input",
                )

            # API Key input
            api_key_help_text = {
                "huggingface": "Get from https://huggingface.co/settings/tokens",
                "openai": "Get from https://platform.openai.com/api-keys",
                "gemini": "Get from https://makersuite.google.com/app/apikey",
                "custom": "Your API key for the custom endpoint",
            }

            api_key_input = st.text_input(
                f"{provider.title()} API Key",
                type="password",
                help=api_key_help_text.get(provider, "Your API key"),
                key="chatbot_api_key_input",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Verify & Start", type="primary", key="verify_btn"):
                    if api_key_input.strip():
                        # Test API key with a simple request
                        test_prompt = create_prompt("Say hello", "No context needed for test.", provider)
                        with st.spinner("Verifying API key and model..."):
                            response = query_llm_model(
                                api_key_input,
                                test_prompt,
                                model_name=model_name_input,
                                base_url=base_url_input if base_url_input else None,
                                max_tokens=10,
                            )

                        if not response.startswith("Error"):
                            st.session_state["chat_api_key"] = api_key_input
                            st.session_state["chat_model_provider"] = provider
                            st.session_state["chat_model_name"] = model_name_input
                            st.session_state["chat_base_url"] = base_url_input
                            st.session_state["chat_api_key_verified"] = True

                            # Only compute embeddings if ChromaDB not available
                            if chromadb_collection is not None:
                                # Using ChromaDB for RAG - no local embeddings needed
                                st.session_state["chat_embeddings"] = None
                                st.info("🌐 Chatbot will use ChromaDB Cloud for RAG")
                            elif embeddings is not None:
                                # Use already computed embeddings
                                st.session_state["chat_embeddings"] = embeddings
                            else:
                                # Compute embeddings for chatbot if not available
                                st.session_state["chat_embeddings"] = compute_embeddings(df["combined"].tolist(), model)

                            st.success("✅ Model verified successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Verification failed: {response}")
                    else:
                        st.error("❌ Please enter a valid API key")

            st.info("""
            **API Key Resources:**
            - **Huggingface**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
            - **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
            - **Gemini**: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
            - **Custom**: Your self-hosted endpoint credentials
            """)

        else:
            st.success("✅ API Key Verified")
            st.info(f"🤖 Provider: **{st.session_state['chat_model_provider'].title()}**")
            st.info(f"📦 Model: **{st.session_state['chat_model_name']}**")
            if st.session_state["chat_base_url"]:
                st.info(f"🔗 Endpoint: **{st.session_state['chat_base_url']}**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Change Configuration", key="change_key_btn"):
                    st.session_state["chat_api_key_verified"] = False
                    st.session_state["chat_api_key"] = ""
                    st.session_state["chat_messages"] = []
                    st.rerun()

            with col2:
                if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
                    st.session_state["chat_messages"] = []
                    st.rerun()

    # Display example questions if not verified
    if not st.session_state["chat_api_key_verified"]:
        st.markdown("---")
        st.subheader("💡 Example Questions You Can Ask:")
        st.markdown("""
        - What are the best action movies in the database?
        - Tell me about movies directed by Christopher Nolan
        - Recommend some high-rated comedy movies
        - What movies feature Tom Hanks?
        - Show me romantic movies with good ratings
        - Compare Marvel and DC movies
        - What are some underrated gems?
        """)
        st.stop()

    # Main chat interface
    st.markdown("---")

    # Display chat history
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display retrieved movies if available
            if message["role"] == "assistant" and "retrieved_movies" in message:
                with st.expander("📚 Retrieved Movies from Database", expanded=False):
                    for idx, row in message["retrieved_movies"].iterrows():
                        st.markdown(f"**{row['title']}** ({row['genres']}) - ⭐ {row['vote_average']}/10")
                        st.caption(f"Director: {row['director']} | Cast: {row['cast']}")

    # Chat input
    if user_input := st.chat_input("Ask me about movies...", key="chatbot_input"):
        # Add user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching database and generating response..."):
                # Step 1: Retrieve relevant movies (RAG with ChromaDB support)
                relevant_movies, scores = retrieve_relevant_movies(
                    user_input,
                    st.session_state["chat_embeddings"],
                    df,
                    model,
                    top_k=50,
                    chromadb_collection=chromadb_collection,
                )

                # Debug: Show how many movies were retrieved
                st.info(f"🎬 Retrieved {len(relevant_movies)} movies from database")

                # Step 2: Create context
                context = create_context_from_movies(relevant_movies)

                # Step 3: Create prompt
                prompt = create_prompt(user_input, context, st.session_state["chat_model_provider"])

                # Step 4: Query LLM
                response = query_llm_model(
                    st.session_state["chat_api_key"],
                    prompt,
                    model_name=st.session_state["chat_model_name"],
                    base_url=st.session_state["chat_base_url"] if st.session_state["chat_base_url"] else None,
                    max_tokens=500,
                )

                # Display response
                st.markdown(response)

                # Display retrieved movies
                with st.expander("📚 Retrieved Movies from Database", expanded=False):
                    for idx, row in relevant_movies.iterrows():
                        st.markdown(f"**{row['title']}** ({row['genres']}) - ⭐ {row['vote_average']}/10")
                        st.caption(f"Director: {row['director']} | Cast: {row['cast']}")

                # Save assistant message
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": response, "retrieved_movies": relevant_movies}
                )
