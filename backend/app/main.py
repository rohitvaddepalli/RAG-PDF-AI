"""FastAPI application entrypoint for RAG PDF AI."""

from contextlib import asynccontextmanager
import re
import uuid
from typing import AsyncGenerator
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import AskRequest, AskResponse, HealthResponse, SourceReference, UploadResponse
from app.services.chunking_service import chunk_pages
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.pdf_service import PDFExtractionError, extract_pages_from_pdf_bytes
from app.services.retrieval_service import document_store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager to load resources once at startup."""
    print("Initializing embedding model...")
    embedding_service.load_model()
    print("Embedding model loaded successfully.")
    yield
    print("Shutting down RAG PDF AI backend.")


app = FastAPI(
    title="RAG PDF AI Backend",
    description="Production-grade API for querying PDFs via local embeddings and Google Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Return system health status and in-memory document metrics."""
    return HealthResponse(
        status="healthy",
        embedding_model_loaded=embedding_service.is_loaded,
        documents_in_memory=document_store.document_count(),
    )


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["RAG Operations"],
)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a PDF document, extract text page by page, chunk into 500-800 character pieces,
    embed using local SentenceTransformers, and store in memory.
    """
    filename = file.filename or "uploaded.pdf"
    content_type = file.content_type or ""
    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files (.pdf) are supported.",
        )

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF file is empty (0 bytes).",
        )

    if len(pdf_bytes) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size exceeds the maximum allowed limit of "
                f"{settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
            ),
        )

    try:
        pages = extract_pages_from_pdf_bytes(pdf_bytes)
    except PDFExtractionError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while parsing the PDF: {str(e)}",
        )

    chunks = chunk_pages(
        pages=pages,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid text chunks could be extracted from this document.",
        )

    try:
        chunk_texts = [c["text"] for c in chunks]
        embeddings_matrix = embedding_service.encode_texts(chunk_texts)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute embeddings: {str(e)}",
        )

    doc_id = uuid.uuid4().hex
    document_store.add_document(
        document_id=doc_id,
        chunks=chunks,
        embeddings=embeddings_matrix,
        page_count=len(pages),
        file_name=filename,
    )

    return UploadResponse(
        document_id=doc_id,
        page_count=len(pages),
        chunk_count=len(chunks),
        file_name=filename,
    )


_OVERVIEW_QUERY_RE = re.compile(
    r"\b(summar(y|ize|ise)|overview|takeaways?|main\s+points?|key\s+points?|"
    r"action\s+items?|recommendations?|primary\s+findings?|"
    r"what\s+is\s+(this|the)\s+(pdf|doc(ument)?|file)\s+about)\b",
    re.IGNORECASE,
)


def _is_overview_query(question: str) -> bool:
    """Return True for broad document-overview or summary questions."""
    return bool(_OVERVIEW_QUERY_RE.search(question))


@app.post("/api/ask", response_model=AskResponse, tags=["RAG Operations"])
async def ask_question(payload: AskRequest) -> AskResponse:
    """
    Search document chunks using NumPy cosine similarity, filter by threshold,
    and formulate a grounded answer using Google Gemini.
    """
    doc = document_store.get_document(payload.document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{payload.document_id}' not found in memory. Please upload a PDF first.",
        )

    cleaned_question = payload.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace only.",
        )

    try:
        query_vector = embedding_service.encode_query(cleaned_question)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode question into embedding vector: {str(e)}",
        )

    top_k = payload.top_k or settings.DEFAULT_TOP_K
    threshold = -1.0 if _is_overview_query(cleaned_question) else settings.RELEVANCE_THRESHOLD
    try:
        retrieved_chunks = document_store.search(
            document_id=payload.document_id,
            query_vector=query_vector,
            top_k=top_k,
            threshold=threshold,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(e)}",
        )

    if not retrieved_chunks:
        return AskResponse(
            answer="I couldn't find this in the document",
            sources=[],
        )

    try:
        answer_text = llm_service.generate_answer(
            question=cleaned_question,
            context_chunks=retrieved_chunks,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM synthesis failed: {str(e)}",
        )

    sources = [
        SourceReference(
            page=c["page"],
            snippet=c["snippet"],
            score=c["score"],
        )
        for c in retrieved_chunks
    ]

    return AskResponse(
        answer=answer_text,
        sources=sources,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
