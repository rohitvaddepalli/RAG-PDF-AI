"""Pydantic data models for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned upon successful PDF upload and embedding."""

    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    page_count: int = Field(..., description="Total number of pages processed")
    chunk_count: int = Field(..., description="Total number of semantic chunks generated")
    file_name: Optional[str] = Field(None, description="Original name of the uploaded PDF file")


class AskRequest(BaseModel):
    """Request payload for asking a question against an uploaded document."""

    document_id: str = Field(..., description="ID of the previously uploaded document")
    question: str = Field(..., min_length=1, description="Question text to answer from the PDF")
    top_k: Optional[int] = Field(4, ge=1, le=20, description="Maximum number of context chunks to retrieve")


class SourceReference(BaseModel):
    """Source reference citation from the document."""

    page: int = Field(..., description="1-indexed page number of the source chunk")
    snippet: str = Field(..., description="Verbatim textual excerpt from the chunk")
    score: float = Field(..., description="Cosine similarity relevance score")


class AskResponse(BaseModel):
    """Response returned for a question asked against a document."""

    answer: str = Field(..., description="Grounded answer synthesized by the LLM or fallback message")
    sources: List[SourceReference] = Field(default_factory=list, description="List of source citations used")


class HealthResponse(BaseModel):
    """Health status response."""

    status: str = Field("healthy", description="Application status")
    embedding_model_loaded: bool = Field(..., description="Whether embedding model is initialized in memory")
    documents_in_memory: int = Field(..., description="Number of documents currently indexed in memory")
