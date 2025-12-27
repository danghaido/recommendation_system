"""
ChromaDB Helper - Connection and retrieval functions
"""

import chromadb
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource(show_spinner=False)
def get_chromadb_client():
    """
    Get ChromaDB Cloud client with credentials from Streamlit secrets
    """
    try:
        # Try to load from Streamlit secrets first (for deployment)
        if hasattr(st, "secrets") and "chromadb" in st.secrets:
            api_key = st.secrets["chromadb"]["api_key"]
            tenant = st.secrets["chromadb"]["tenant"]
            database = st.secrets["chromadb"]["database"]
        else:
            # Fallback for local development (not recommended for production)
            raise ValueError("ChromaDB credentials not found in secrets")

        client = chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)

        print("Connected to ChromaDB Cloud")
        return client

    except Exception as e:
        st.error(f"Failed to connect to ChromaDB Cloud: {e}")
        st.info("Make sure to configure secrets.toml or Streamlit Cloud secrets")
        return None


@st.cache_resource(show_spinner=False)
def get_chromadb_collection(_client, collection_name="movie_embeddings"):
    """
    Get ChromaDB collection
    """
    try:
        if _client is None:
            return None

        collection = _client.get_collection(name=collection_name)
        print(f"Loaded collection: {collection_name} with {collection.count()} items")
        return collection

    except Exception as e:
        st.error(f"Failed to load collection: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_sentence_model():
    """Load SentenceTransformer model for query encoding"""
    return SentenceTransformer("all-MiniLM-L6-v2")


def search_similar_movies_by_text(collection, query_text, model, top_k=10, excluded_titles=None):
    """
    Search for similar movies using text query

    Args:
        collection: ChromaDB collection
        query_text: Query text to search
        model: SentenceTransformer model
        top_k: Number of results to return
        excluded_titles: List of titles to exclude from results

    Returns:
        List of movie titles (to be mapped to indices by caller)
    """
    try:
        if collection is None or model is None:
            return []

        # Encode query
        query_embedding = model.encode([query_text])[0].tolist()
        print("\n=== ChromaDB Text Search ===")
        print(f"Query: {query_text[:100]}...")
        print(f"Embedding shape: {len(query_embedding)}")

        # Search in ChromaDB (get more for filtering)
        n_results = min(top_k * 3, 300)
        print(f"Requesting {n_results} results from ChromaDB...")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        print("\nChromaDB Response:")
        print(f"  - IDs returned: {len(results.get('ids', [[]])[0]) if results and 'ids' in results else 0}")
        print(f"  - Metadatas: {len(results.get('metadatas', [[]])[0]) if results and 'metadatas' in results else 0}")
        print(f"  - Distances: {len(results.get('distances', [[]])[0]) if results and 'distances' in results else 0}")

        if not results or not results["ids"] or len(results["ids"]) == 0 or len(results["ids"][0]) == 0:
            print(f"ChromaDB returned empty results for query: {query_text}")
            return []

        # Print top 10 results for debugging
        print(f"\nTop {min(10, len(results['ids'][0]))} results:")
        for i in range(min(10, len(results["ids"][0]))):
            metadata = results["metadatas"][0][i] if i < len(results["metadatas"][0]) else {}
            distance = results["distances"][0][i] if i < len(results["distances"][0]) else None
            distance_str = f"{distance:.4f}" if isinstance(distance, (int, float)) else "N/A"
            print(
                f"  {i + 1}. Title: {metadata.get('title', 'N/A')[:40]}, Index: {metadata.get('index', 'N/A')}, Distance: {distance_str}"
            )

        # Extract titles from metadata (return titles instead of indices)
        titles = []
        for i, metadata in enumerate(results["metadatas"][0]):
            title = metadata.get("title", "")

            # Skip if in excluded list
            if excluded_titles and title in excluded_titles:
                continue

            if title:
                titles.append(title)

            if len(titles) >= top_k:
                break

        print(f"\nReturning {len(titles)} titles")
        print(f"Titles: {titles[:5]}..." if len(titles) > 5 else f"Titles: {titles}")
        return titles

    except Exception as e:
        print(f"Search error: {e}")
        import traceback

        traceback.print_exc()
        return []


def search_similar_movies_by_embedding(_collection, query_embedding, top_k=10, excluded_indices=None):
    """
    Search for similar movies using embedding vector

    Args:
        _collection: ChromaDB collection
        query_embedding: Numpy array or list of embedding
        top_k: Number of results to return
        excluded_indices: List of indices to exclude

    Returns:
        List of movie titles (to be mapped to indices by caller)
    """
    try:
        if _collection is None:
            return []

        # Convert to list if numpy array
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        print("\n=== ChromaDB Embedding Search ===")
        print(f"Embedding shape: {len(query_embedding)}")
        print(
            f"Excluded indices: {excluded_indices[:5]}..."
            if excluded_indices and len(excluded_indices) > 5
            else f"Excluded indices: {excluded_indices}"
        )

        n_results = min(top_k * 3, 300)
        print(f"Requesting {n_results} results from ChromaDB...")

        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        print("\nChromaDB Response:")
        print(f"  - IDs returned: {len(results.get('ids', [[]])[0]) if results and 'ids' in results else 0}")
        print(f"  - Metadatas: {len(results.get('metadatas', [[]])[0]) if results and 'metadatas' in results else 0}")
        print(f"  - Distances: {len(results.get('distances', [[]])[0]) if results and 'distances' in results else 0}")

        if not results or not results["ids"] or len(results["ids"]) == 0 or len(results["ids"][0]) == 0:
            print("ChromaDB returned empty results for embedding query")
            return []

        # Print top 10 results for debugging
        print(f"\nTop {min(10, len(results['ids'][0]))} results:")
        for i in range(min(10, len(results["ids"][0]))):
            metadata = results["metadatas"][0][i] if i < len(results["metadatas"][0]) else {}
            distance = results["distances"][0][i] if i < len(results["distances"][0]) else None
            distance_str = f"{distance:.4f}" if isinstance(distance, (int, float)) else "N/A"
            print(f"  {i + 1}. Title: {metadata.get('title', 'N/A')[:40]}, Distance: {distance_str}")

        # Extract titles from metadata (return titles instead of indices)
        titles = []
        for metadata in results["metadatas"][0]:
            title = metadata.get("title", "")

            if title:
                titles.append(title)

            if len(titles) >= top_k:
                break

        print(f"\nReturning {len(titles)} titles")
        print(f"Titles: {titles[:5]}..." if len(titles) > 5 else f"Titles: {titles}")
        return titles

    except Exception as e:
        print(f"Search error: {e}")
        import traceback

        traceback.print_exc()
        return []


def get_movie_embedding_by_index(collection, movie_index):
    """
    Get embedding for a specific movie by its index
    Uses ID format: movie_{index}

    Args:
        collection: ChromaDB collection
        movie_index: Index of the movie in dataframe

    Returns:
        Numpy array of embedding or None
    """
    try:
        if collection is None:
            return None

        print("\n=== Get Movie Embedding ===")
        print(f"Fetching embedding for index: {movie_index}")

        # Query by ID (format: movie_{index})
        movie_id = f"movie_{movie_index}"
        results = collection.get(ids=[movie_id], include=["embeddings", "metadatas"])

        print(f"Results: {len(results.get('embeddings', []))} embeddings found")
        if results.get("metadatas") and len(results["metadatas"]) > 0:
            print(f"Movie: {results['metadatas'][0].get('title', 'N/A')}")

        if results and results.get("embeddings") is not None and len(results["embeddings"]) > 0:
            embedding = np.array(results["embeddings"][0])
            print(f"Embedding shape: {embedding.shape}")
            return embedding

        print(f"No embedding found for index {movie_index}")
        return None

    except Exception as e:
        print(f"Error getting embedding for index {movie_index}: {e}")
        import traceback

        traceback.print_exc()
        return None
