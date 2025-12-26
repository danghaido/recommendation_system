# chatbot.py
# Helper functions for Movie Chatbot with RAG
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from litellm import completion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("Warning: litellm not installed. Install with: pip install litellm")

# You can change these to use different models
DEFAULT_MODEL_CONFIG = {
    "huggingface": {
        "model_name": "huggingface/Qwen/Qwen2.5-7B-Instruct",
        "base_url": None,
    },
    "openai": {
        "model_name": "gpt-3.5-turbo",
        "base_url": None,
    },
    "gemini": {
        "model_name": "gemini/gemini-pro",
        "base_url": None,
    },
    "custom": {
        "model_name": "openai/your-model-name",
        "base_url": "http://localhost:8000/v1",
    },
}


def retrieve_relevant_movies(query, embeddings, df, embedding_model, top_k=10):
    """
    Retrieve top-k most relevant movies based on query using RAG

    Args:
        query: User query string
        embeddings: Pre-computed embeddings for all movies
        df: Movie dataframe
        embedding_model: SentenceTransformer model
        top_k: Number of movies to retrieve

    Returns:
        relevant_movies: DataFrame of top-k relevant movies
        scores: Similarity scores
    """
    # Encode query
    query_embedding = embedding_model.encode([query])

    # Compute similarity
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Get relevant movies
    relevant_movies = df.iloc[top_indices]

    return relevant_movies, similarities[top_indices]


def create_context_from_movies(movies_df):
    """Create context string from retrieved movies"""
    context = "Here is the relevant information from the movie database:\n\n"

    for idx, row in movies_df.iterrows():
        # Extract year from release_date
        year = "Unknown"
        if "release_date" in row and pd.notna(row["release_date"]):
            try:
                year = str(pd.to_datetime(row["release_date"]).year)
            except:
                year = "Unknown"

        context += f"Movie: {row['title']}\n"
        context += f"Year: {year}\n"
        context += f"Genres: {row['genres']}\n"
        context += f"Director: {row['director']}\n"
        context += f"Cast: {row['cast']}\n"
        context += f"Rating: {row['vote_average']}/10\n"
        context += f"Overview: {row['overview']}\n"
        context += "-" * 80 + "\n\n"

    return context


def query_llm_model(
    api_key, prompt, model_name="huggingface/mistralai/Mistral-7B-Instruct-v0.2", base_url=None, max_tokens=500
):
    """
    Query LLM model using litellm (supports multiple providers)

    Args:
        api_key: API key for the provider
        prompt: The prompt to send to the model
        model_name: Model name in litellm format (e.g., "gpt-3.5-turbo", "gemini/gemini-pro", "huggingface/model-name")
        base_url: Custom base URL for self-hosted models (optional)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text or error message
    """
    if not LITELLM_AVAILABLE:
        return "Error: litellm is not installed. Please run: pip install litellm"

    try:
        # Prepare messages for chat completion
        messages = [{"role": "user", "content": prompt}]

        # Build completion arguments
        completion_args = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "api_key": api_key,
        }

        # Add base_url if provided (for self-hosted models)
        if base_url:
            completion_args["base_url"] = base_url

        # Call litellm
        response = completion(**completion_args)

        # Extract generated text
        if response and response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "No response generated"

    except Exception as e:
        return f"Error querying model: {str(e)}"


def create_prompt(user_query, context, model_type="huggingface"):
    """
    Create prompt for LLM with RAG context

    Args:
        user_query: User's question
        context: Movie context from RAG
        model_type: Type of model ("huggingface", "openai", "gemini", "custom")

    Returns:
        Formatted prompt based on model type
    """
    # Base instruction and context
    base_prompt = f"""You are a knowledgeable and friendly movie recommendation assistant. Provide helpful, conversational responses based on the movie database information.

Movie Database Context:
{context}

User Question: {user_query}

Instructions for your response:
1. Start with a brief, friendly introduction acknowledging the user's question
2. Explain what you found in the database relevant to their query
3. Present the movies in a conversational way:
   - For each movie, weave together the title, year, director, cast, rating, and a brief description
   - Use natural language instead of bullet points (e.g., "The first movie I'd recommend is...")
   - Explain WHY each movie fits their criteria
   - Highlight interesting facts or connections between movies if applicable
4. If no exact match exists, explain the closest alternatives and why they might interest the user
5. End with a helpful closing (e.g., asking if they'd like more information)
6. Never say "no movies found" - always find and present the most relevant options
7. Keep your tone warm, enthusiastic, and informative
8. Use paragraphs instead of lists for a more conversational feel

Answer:"""

    # Format based on model type
    if model_type == "huggingface":
        # Mistral/Llama format with [INST] tags
        prompt = f"<s>[INST] {base_prompt} [/INST]"
    elif model_type in ["openai", "gemini", "custom"]:
        # OpenAI/Gemini use chat format, so just return the instruction
        prompt = base_prompt
    else:
        # Default fallback
        prompt = base_prompt

    return prompt
