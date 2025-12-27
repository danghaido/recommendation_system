"""
Setup ChromaDB Cloud - Embedding and Upload Script
Run this script ONCE to compute embeddings and upload to ChromaDB Cloud
"""

import ast

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ChromaDB Cloud Configuration
CHROMADB_API_KEY = "ck-8eqqT4dC6oD1LTYrSZd861JiD18aLYWD8E2QvUr2Rpoq"
CHROMADB_TENANT = "b2272094-71db-4499-8493-f2f113d76080"
CHROMADB_DATABASE = "testing"
COLLECTION_NAME = "movie_embeddings"

DATA_PATH = "csv/processed_data.csv"


def load_and_prepare_data():
    """Load and prepare movie data"""
    print("Loading movie data...")
    df = pd.read_csv(DATA_PATH)

    # Basic clean
    df = df.fillna({"overview": "", "genres": "", "cast": "", "director": "", "release_date": ""})

    # Parse genres_list
    if "genres_list" in df.columns:
        df["genres_list"] = df["genres_list"].apply(
            lambda x: ast.literal_eval(x)
            if isinstance(x, str) and x.startswith("[")
            else (x if isinstance(x, list) else [])
        )

    # Extract year
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype(str)

    # Create combined text for embedding
    df["combined"] = (
        "Title: "
        + df["title"].astype(str)
        + ". Year: "
        + df["year"].astype(str)
        + ". Director: "
        + df["director"].astype(str)
        + ". Director: "
        + df["director"].astype(str)
        + ". Genres: "
        + df["genres"].astype(str)
        + ". Cast: "
        + df["cast"].astype(str)
        + ". Rating: "
        + df["vote_average"].astype(str)
        + "/10. Overview: "
        + df["overview"].astype(str)
    )

    print(f"Loaded {len(df)} movies")
    return df


def compute_embeddings(df):
    """Compute embeddings using SentenceTransformer"""
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Computing embeddings for {len(df)} movies...")
    embeddings = model.encode(df["combined"].tolist(), show_progress_bar=True, batch_size=32)

    print(f"Embeddings computed! Shape: {embeddings.shape}")
    return embeddings


def upload_to_chromadb(df, embeddings):
    """Upload embeddings to ChromaDB Cloud"""
    print("Connecting to ChromaDB Cloud...")

    try:
        client = chromadb.CloudClient(api_key=CHROMADB_API_KEY, tenant=CHROMADB_TENANT, database=CHROMADB_DATABASE)
        print("Connected to ChromaDB Cloud")

        # Delete existing collection if exists
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except:
            pass

        # Create new collection
        collection = client.create_collection(
            name=COLLECTION_NAME, metadata={"description": "Movie embeddings for recommendation system"}
        )
        print(f"Created collection: {COLLECTION_NAME}")

        # Prepare data for upload
        ids = [f"movie_{i}" for i in range(len(df))]
        embeddings_list = embeddings.tolist()

        # Metadata for each movie
        metadatas = []
        for idx, row in df.iterrows():
            metadatas.append(
                {
                    "index": int(idx),  # Add index for ordering
                    "title": str(row["title"]),
                    "genres": str(row["genres"]),
                    "director": str(row["director"]),
                    "cast": str(row["cast"])[:500],
                    "vote_average": float(row["vote_average"]),
                    "year": str(row["year"]),
                    "overview": str(row["overview"])[:1000],
                }
            )

        documents = df["combined"].tolist()

        # Upload in batches
        batch_size = 100
        total_batches = (len(df) + batch_size - 1) // batch_size

        print(f"Uploading {len(df)} embeddings in {total_batches} batches...")

        for i in tqdm(range(0, len(df), batch_size), desc="Uploading"):
            end_idx = min(i + batch_size, len(df))

            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings_list[i:end_idx],
                metadatas=metadatas[i:end_idx],
                documents=documents[i:end_idx],
            )

        print(f"Successfully uploaded {len(df)} embeddings to ChromaDB Cloud!")

        # Verify upload
        count = collection.count()
        print(f"Collection now contains {count} items")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main execution function"""
    print("=" * 60)
    print("ChromaDB Cloud Setup - Movie Embeddings")
    print("=" * 60)

    # Step 1: Load data
    df = load_and_prepare_data()

    # Step 2: Compute embeddings
    embeddings = compute_embeddings(df)

    # Step 3: Upload to ChromaDB Cloud
    success = upload_to_chromadb(df, embeddings)

    if success:
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run 'streamlit run main.py' to use the app")
        print("2. The app will now load embeddings from ChromaDB Cloud")
        print("3. No more slow embedding computation on startup!")
    else:
        print("\nSetup failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
