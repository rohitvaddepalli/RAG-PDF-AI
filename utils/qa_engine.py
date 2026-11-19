import os
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from utils.embeddings import embed_query
from utils.vector_store import VectorStore

DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
DEFAULT_TOP_K: int = 4
DEFAULT_SCORE_THRESHOLD: float = 0.25

INSUFFICIENT_CONTEXT_MESSAGE: str = (
    "I do not have enough information from the uploaded document to answer this question."
)

SYSTEM_INSTRUCTION: str = (
    "You are a factual, concise AI assistant answering questions about a PDF document.\n"
    "Strict Rules:\n"
    "1. Answer ONLY using the facts explicitly stated in the provided context.\n"
    "2. Do NOT extrapolate, speculate, or introduce external knowledge.\n"
    "3. If the provided context does not contain enough information to answer the question truthfully and completely, "
    f"respond with: '{INSUFFICIENT_CONTEXT_MESSAGE}'."
)


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not resolved_key or not resolved_key.strip():
        raise ValueError(
            "Gemini API key not found. Please set the GEMINI_API_KEY environment variable "
            "or pass an explicit api_key."
        )

    return genai.Client(api_key=resolved_key.strip())


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "Document")
        page = chunk.get("page_number", "Unknown")
        text = chunk.get("text", "").strip()
        context_blocks.append(f"[Block {idx} | Source: {source} | Page: {page}]\n{text}")

    context_str = "\n\n".join(context_blocks)

    return (
        f"Context information from the document:\n\n"
        f"{context_str}\n\n"
        f"User Question: {question.strip()}\n\n"
        f"Answer:"
    )


def answer_question(
    question: str,
    vector_store: VectorStore,
    client: Optional[Any] = None,
    api_key: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    model_name: str = DEFAULT_GEMINI_MODEL,
    embed_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Answers a user question based strictly on retrieved PDF context.

    Steps:
    1. Validates the question input.
    2. Generates the question embedding using embed_query().
    3. Retrieves top relevant chunks from VectorStore.
    4. If no relevant chunks are found, returns an insufficient-context response.
    5. Builds the context-aware prompt.
    6. Calls Gemini API using the official google-genai SDK.
    7. Returns a structured dictionary containing the answer and source chunks.

    Args:
        question: User question string.
        vector_store: Active VectorStore instance to search against.
        client: Optional pre-configured or mock Google GenAI client (for dependency injection).
        api_key: Optional explicit API key.
        top_k: Number of relevant chunks to retrieve (default: 4, within 3-5 range).
        score_threshold: Minimum cosine similarity score required for relevance (default: 0.25).
        model_name: Gemini model name (default: 'gemini-2.5-flash').
        embed_model: Optional mock/preloaded embedding model for offline testing.

    Returns:
        Dict[str, Any]:
            - 'answer': Generated answer string or insufficient-context message.
            - 'retrieved_chunks': List of chunk dictionaries retrieved from VectorStore.
            - 'context_used': Boolean indicating if retrieved context was used.
            - 'status': 'success', 'insufficient_context', or 'error'.
            - 'error_message': Error description if status == 'error', else None.
    """
    if not isinstance(question, str) or not question.strip():
        return {
            "answer": "Please provide a valid, non-empty question.",
            "retrieved_chunks": [],
            "context_used": False,
            "status": "error",
            "error_message": "Question must be a non-empty string.",
        }

    # Step 1: Generate query embedding
    try:
        query_vector = embed_query(question.strip(), model=embed_model)
    except Exception as e:
        return {
            "answer": f"Failed to generate embedding for question: {str(e)}",
            "retrieved_chunks": [],
            "context_used": False,
            "status": "error",
            "error_message": str(e),
        }

    # Step 2: Retrieve top 3-5 relevant chunks from VectorStore
    try:
        retrieved_chunks = vector_store.search(
            query_embedding=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )
    except Exception as e:
        return {
            "answer": f"Failed to search vector store: {str(e)}",
            "retrieved_chunks": [],
            "context_used": False,
            "status": "error",
            "error_message": str(e),
        }

    # Step 3: Check for empty or insufficient context
    if not retrieved_chunks:
        return {
            "answer": INSUFFICIENT_CONTEXT_MESSAGE,
            "retrieved_chunks": [],
            "context_used": False,
            "status": "insufficient_context",
            "error_message": None,
        }

    # Step 4: Build context-aware prompt
    prompt = build_prompt(question=question, chunks=retrieved_chunks)

    # Step 5: Initialize / resolve LLM client
    try:
        active_client = client if client is not None else get_gemini_client(api_key=api_key)
    except Exception as e:
        return {
            "answer": f"Authentication error: {str(e)}",
            "retrieved_chunks": retrieved_chunks,
            "context_used": True,
            "status": "error",
            "error_message": str(e),
        }

    # Step 6: Generate response using Google Gemini
    try:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
        )
        response = active_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        answer_text = response.text.strip() if response and response.text else INSUFFICIENT_CONTEXT_MESSAGE

        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks,
            "context_used": True,
            "status": "success",
            "error_message": None,
        }
    except Exception as e:
        return {
            "answer": f"An error occurred while calling the Gemini API: {str(e)}",
            "retrieved_chunks": retrieved_chunks,
            "context_used": True,
            "status": "error",
            "error_message": str(e),
        }


class QAEngine:
    """
    Object-oriented interface for the RAG Question-Answering engine.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ):
        self.client = client
        self.api_key = api_key
        self.model_name = model_name
        self.top_k = top_k
        self.score_threshold = score_threshold

    def ask(
        self,
        question: str,
        vector_store: VectorStore,
        embed_model: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Convenience method to ask a question against a VectorStore."""
        return answer_question(
            question=question,
            vector_store=vector_store,
            client=self.client,
            api_key=self.api_key,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            model_name=self.model_name,
            embed_model=embed_model,
        )
