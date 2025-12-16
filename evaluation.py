import numpy as np
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MovieRecommenderEvaluator:
    """
    Class đánh giá hệ thống gợi ý phim
    Tái sử dụng data và hàm từ main.py
    """

    def __init__(self, df, recommend_func):
        """
        Args:
            df: DataFrame từ main.py
            recommend_func: Hàm recommend_by_title từ main.py
        """
        self.df = df
        self.recommend_func = recommend_func

    def precision_at_k(self, recommended, relevant, k):
        """Tính Precision@K"""
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(relevant))
        return hits / k if k > 0 else 0

    def recall_at_k(self, recommended, relevant, k):
        """Tính Recall@K"""
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(relevant))
        return hits / len(relevant) if len(relevant) > 0 else 0

    def ndcg_at_k(self, recommended, relevant, k):
        """Tính NDCG@K"""
        recommended_k = recommended[:k]
        dcg = sum([1 / np.log2(i + 2) for i, rec in enumerate(recommended_k) if rec in relevant])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
        return dcg / idcg if idcg > 0 else 0

    def evaluate_content_based(self, n_samples=100, k=10):
        """
        Evaluate Content-Based Filtering

        Ground Truth: Movies with same director OR at least 1 common genre
        """
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        rating_actual = []
        rating_pred = []

        # Random sampling
        test_movies = self.df.sample(min(n_samples, len(self.df)), random_state=42)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (index, movie) in enumerate(test_movies.iterrows()):
            progress = (idx + 1) / len(test_movies)
            progress_bar.progress(progress)
            status_text.text(f"Evaluating: {idx + 1}/{len(test_movies)} - {movie['title']}")

            # Ground truth: movies with same director or same genre
            same_director = self.df[
                (self.df["director"] == movie["director"])
                & (self.df["director"] != "")
                & (self.df["director"] != "Unknown")
            ]["title"].tolist()

            same_genres = self.df[
                self.df["genres_list"].apply(
                    lambda x: bool(set(x) & set(movie["genres_list"]))
                    if isinstance(x, list) and isinstance(movie["genres_list"], list)
                    else False
                )
            ]["title"].tolist()

            relevant = list(set(same_director + same_genres))
            if movie["title"] in relevant:
                relevant.remove(movie["title"])

            if len(relevant) == 0:
                continue

            # Get recommendations using existing function
            try:
                recommendations = self.recommend_func(movie["title"], topn=k)
                if recommendations.empty:
                    continue

                recommended = recommendations["title"].tolist()

                # Calculate metrics
                precision = self.precision_at_k(recommended, relevant, k)
                recall = self.recall_at_k(recommended, relevant, k)
                ndcg = self.ndcg_at_k(recommended, relevant, k)

                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)

                # Rating consistency
                actual_rating = movie["vote_average"]
                for _, rec in recommendations.iterrows():
                    rating_actual.append(actual_rating)
                    rating_pred.append(rec["vote_average"])

            except Exception:
                continue

        progress_bar.empty()
        status_text.empty()

        # Calculate RMSE & MAE
        rmse = np.sqrt(mean_squared_error(rating_actual, rating_pred)) if rating_actual else 0
        mae = mean_absolute_error(rating_actual, rating_pred) if rating_actual else 0

        return {
            "precision@k": np.mean(precision_scores) if precision_scores else 0,
            "recall@k": np.mean(recall_scores) if recall_scores else 0,
            "ndcg@k": np.mean(ndcg_scores) if ndcg_scores else 0,
            "rmse": rmse,
            "mae": mae,
            "n_evaluated": len(precision_scores),
        }

    def evaluate_diversity(self, n_samples=50, k=10):
        """Evaluate diversity"""
        all_recommendations = []
        genre_diversity_scores = []

        test_movies = self.df.sample(min(n_samples, len(self.df)), random_state=42)

        for _, movie in test_movies.iterrows():
            try:
                recs = self.recommend_func(movie["title"], topn=k)
                if not recs.empty:
                    rec_titles = recs["title"].tolist()
                    all_recommendations.extend(rec_titles)

                    # Count unique genres
                    all_genres = []
                    for title in rec_titles:
                        movie_genres = self.df[self.df["title"] == title]["genres_list"].values
                        if len(movie_genres) > 0 and isinstance(movie_genres[0], list):
                            all_genres.extend(movie_genres[0])
                    genre_diversity_scores.append(len(set(all_genres)))
            except:
                continue

        unique_recs = len(set(all_recommendations))
        total_recs = len(all_recommendations)

        return {
            "diversity": unique_recs / total_recs if total_recs > 0 else 0,
            "coverage": len(set(all_recommendations)) / len(self.df),
            "avg_genre_diversity": np.mean(genre_diversity_scores) if genre_diversity_scores else 0,
            "total_unique": unique_recs,
        }

    def run_full_evaluation(self, n_samples=100, k=10):
        """Run all evaluations"""
        st.subheader("Running evaluation...")

        # 1. Content-Based
        st.write("**Accuracy Metrics (Precision, Recall, NDCG, RMSE, MAE)**")
        content_results = self.evaluate_content_based(n_samples=n_samples, k=k)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Precision@K", f"{content_results['precision@k']:.4f}")
        col2.metric("Recall@K", f"{content_results['recall@k']:.4f}")
        col3.metric("NDCG@K", f"{content_results['ndcg@k']:.4f}")
        col4.metric("RMSE", f"{content_results['rmse']:.4f}")
        col5.metric("MAE", f"{content_results['mae']:.4f}")

        st.caption(f"Evaluated {content_results['n_evaluated']} movies")
        st.markdown("---")

        # 2. Diversity
        st.write("**Diversity Metrics**")
        diversity_results = self.evaluate_diversity(n_samples=n_samples, k=k)

        col1, col2, col3 = st.columns(3)
        col1.metric("Diversity", f"{diversity_results['diversity']:.4f}")
        col2.metric("Coverage", f"{diversity_results['coverage']:.4f}")
        col3.metric("Genre Diversity", f"{diversity_results['avg_genre_diversity']:.2f}")

        st.markdown("---")
        st.success("Evaluation completed!")

        return {"content": content_results, "diversity": diversity_results}


def display_evaluation_info():
    """Display metrics explanation"""
    st.markdown("""
    ### 📖 Metrics Explanation
    
    **Accuracy Metrics:**
    - **Precision@K**: Ratio of relevant movies in top-K (0-1, higher is better)
    - **Recall@K**: Ratio of relevant movies found (0-1, higher is better)
    - **NDCG@K**: Ranking quality (0-1, higher is better)
    - **RMSE/MAE**: Rating error between original and recommended movies (lower is better)
    
    **Diversity Metrics:**
    - **Diversity**: Ratio of unique movies in total recommendations (0-1)
    - **Coverage**: Ratio of movies recommended at least once (0-1)
    - **Genre Diversity**: Average number of genres in recommendation list
    
    **Ground Truth:** Movies with same director OR at least 1 common genre
    """)
