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

DATA_PATH = "csv/processed_data.csv"


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
    print(f"🔄 Computing embeddings for {len(combined_texts)} movies...")
    embeddings = _model.encode(combined_texts, show_progress_bar=False)
    print("✅ Embeddings computed and cached!")
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
st.set_page_config(layout="wide", page_title="Movie Recommender Demo")
st.title("Movie Recommender — Demo")

df = load_data(DATA_PATH)
model = load_sentence_model()
# Compute embeddings once for both recommendation system and chatbot
embeddings = compute_embeddings(df["combined"].tolist(), model)
cast_director_corpus = (df["cast"].astype(str) + " " + df["director"].astype(str)).tolist()
tfidf_vectorizer, tfidf_matrix = fit_tfidf_on_cast_director(cast_director_corpus)

# index lookup
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


def push_history(q):
    hist = st.session_state["search_history"]
    if q in hist:
        hist.remove(q)
    hist.insert(0, q)
    # keep up to 5
    st.session_state["search_history"] = hist[:5]
    # Reset history recommendations khi có search mới
    st.session_state["history_recommendations"] = None


# ---------- Recommendation functions ----------
def recommend_by_title(title, topn=6):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sims = cosine_similarity([embeddings[idx]], embeddings).flatten()
    top_idx = np.argsort(sims)[::-1][1 : topn + 1]  # skip itself
    return df.iloc[top_idx][["title", "genres", "cast", "director", "vote_average", "overview"]]


def recommend_by_director(director_name, topn=6):
    dir_vec = tfidf_vectorizer.transform([director_name])
    sims = cosine_similarity(dir_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:topn]
    return df.iloc[top_idx][["title", "genres", "cast", "director", "vote_average", "overview"]]


def recommend_by_genre(genre_name, topn=6):
    # filter movies that have the genre in genres_list
    mask = df["genres_list"].apply(lambda x: genre_name in x if isinstance(x, list) else False)
    if mask.sum() == 0:
        return pd.DataFrame()
    # compute centroid of those movies then find nearest neighbors in embedding space
    centroid = embeddings[mask].mean(axis=0)
    sims = cosine_similarity([centroid], embeddings).flatten()
    top_idx = np.argsort(sims)[::-1][:topn]
    return df.iloc[top_idx][["title", "genres", "cast", "director", "vote_average", "overview"]]


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
    Tính embedding trung bình của các phim đã tìm và tìm phim tương tự.
    """
    if not history_list:
        return pd.DataFrame()

    # Tìm tất cả phim matching với history
    all_matched_indices = []
    for query in history_list:
        q_normalized = query.lower().replace(" ", "").replace("-", "").replace(":", "")
        df_normalized = (
            df["title"]
            .str.lower()
            .str.replace(" ", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.replace(":", "", regex=False)
        )
        mask = df_normalized.str.contains(q_normalized, na=False)
        matched_idx = df[mask].index.tolist()
        all_matched_indices.extend(matched_idx)

    if not all_matched_indices:
        return pd.DataFrame()

    # Loại bỏ duplicate
    all_matched_indices = list(set(all_matched_indices))

    # Tính embedding trung bình
    matched_embeddings = embeddings[all_matched_indices]
    centroid = matched_embeddings.mean(axis=0)

    # Tìm phim tương tự (loại bỏ các phim đã có trong history)
    sims = cosine_similarity([centroid], embeddings).flatten()

    # Loại bỏ các phim đã match
    for idx in all_matched_indices:
        sims[idx] = -1

    top_idx = np.argsort(sims)[::-1][:topn]
    return df.iloc[top_idx][["title", "genres", "cast", "director", "vote_average", "overview"]]


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


def display_movie_list(movie_df, key_prefix="movie"):
    """
    Hiển thị danh sách phim với nút Mô tả để xem overview.
    """
    for idx, row in movie_df.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{row['title']}** - {row['genres']} - ⭐ {row['vote_average']}")
            st.caption(f"**Cast:** {row['cast']}")
            st.caption(f"**Director:** {row['director']}")
        with col2:
            btn_key = f"{key_prefix}_{idx}"
            if st.button("📖 Description", key=btn_key):
                st.session_state[f"show_overview_{btn_key}"] = not st.session_state.get(
                    f"show_overview_{btn_key}", False
                )

        # Hiển thị overview nếu được chọn
        if st.session_state.get(f"show_overview_{btn_key}", False):
            st.info(f"**Summary:** {row['overview']}")

        st.markdown("---")


def display_movie_cards(movie_df, key_prefix="card"):
    """
    Hiển thị danh sách phim dạng card đẹp hơn cho gợi ý lịch sử.
    """
    # Hiển thị 3 phim mỗi hàng
    cols_per_row = 3
    rows = len(movie_df) // cols_per_row + (1 if len(movie_df) % cols_per_row != 0 else 0)

    movie_list = list(movie_df.iterrows())

    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            movie_idx = row_idx * cols_per_row + col_idx
            if movie_idx < len(movie_list):
                idx, row = movie_list[movie_idx]
                with cols[col_idx]:
                    # Card container với border
                    st.markdown(
                        f"""
                    <div style='padding: 15px; border: 1px solid #ddd; border-radius: 10px; background-color: #f9f9f9; height: 100%;'>
                        <h4 style='margin: 0 0 10px 0; color: #1f77b4;'>{row["title"]}</h4>
                        <p style='margin: 5px 0; font-size: 14px;'><b>Genre:</b> {row["genres"]}</p>
                        <p style='margin: 5px 0; font-size: 14px;'><b>Rating:</b> ⭐ {row["vote_average"]}</p>
                        <p style='margin: 5px 0; font-size: 13px; color: #666;'><b>Director:</b> {row["director"]}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Description button
                    btn_key = f"{key_prefix}_{idx}"
                    if st.button("📖 View Description", key=btn_key, use_container_width=True):
                        st.session_state[f"show_overview_{btn_key}"] = not st.session_state.get(
                            f"show_overview_{btn_key}", False
                        )

                    # Display overview if selected
                    if st.session_state.get(f"show_overview_{btn_key}", False):
                        st.info(f"**Summary:** {row['overview']}")

                    st.markdown("<br>", unsafe_allow_html=True)


# ---------- UI: tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Search", "Statistics", "Evaluation", "Chatbot"])

with tab1:
    st.header("Home — Quick Recommendations")

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
    st.header("Search — Find Movies & Search History (5 most recent)")

    # Search input
    q = st.text_input("Enter movie name (partial name allowed)", "")

    if st.button("🔍 Search", type="primary"):
        if q.strip() == "":
            st.warning("⚠️ Please enter a movie name to search.")
        else:
            # fuzzy search: normalize by removing spaces and special chars
            q_normalized = q.lower().replace(" ", "").replace("-", "").replace(":", "")
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
    st.header("📊 Data Statistics & Visualizations")

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
    st.header("📊 Recommendation System Evaluation")

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
    st.header("🤖 AI Chatbot - Ask About Movies")
    st.markdown("Chat with AI to get personalized movie recommendations and information")

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

                            # Use already computed embeddings (no need to recompute!)
                            st.session_state["chat_embeddings"] = embeddings

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
                # Step 1: Retrieve relevant movies (RAG)
                relevant_movies, scores = retrieve_relevant_movies(
                    user_input, st.session_state["chat_embeddings"], df, model, top_k=10
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
