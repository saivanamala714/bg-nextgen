from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class ChatMessage(BaseModel):
    """Individual chat message in conversation history."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., min_length=1, description="User question")
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Session identifier")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Chat history")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")


class ChatResponse(BaseModel):
    """Response model for chat endpoint (used for non-streaming)."""
    answer: str = Field(..., description="AI assistant response")
    session_id: str = Field(..., description="Session identifier")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents used")
    metadata: Dict[str, Any] = Field(default={}, description="Response metadata")


class PDFChunk(BaseModel):
    """Model for PDF chunk data."""
    chunk_text: str = Field(..., description="Text content of the chunk")
    embedding: List[float] = Field(..., description="Vector embedding")
    metadata: Dict[str, Any] = Field(default={}, description="Chunk metadata (page numbers, etc.)")
    pdf_name: str = Field(..., description="Source PDF filename")


class IngestRequest(BaseModel):
    """Request model for ingesting precomputed embeddings."""
    chunks: List[PDFChunk] = Field(..., description="PDF chunks with embeddings")
    pdf_name: Optional[str] = Field(default="document.pdf", description="PDF filename")


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    success: bool = Field(..., description="Whether ingestion was successful")
    inserted_count: int = Field(..., description="Number of chunks inserted")
    message: str = Field(..., description="Status message")


class DocumentSearchResult(BaseModel):
    """Model for document search results."""
    chunk_text: str = Field(..., description="Text content")
    similarity: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    pdf_name: str = Field(..., description="Source PDF name")


class ChatLogEntry(BaseModel):
    """Model for chat log entries."""
    id: str
    session_id: str
    question: str
    answer: str
    context_used: str
    metadata: Dict[str, Any]
    created_at: datetime
    feedback: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(default="1.0.0", description="API version")
    services: Dict[str, str] = Field(default={}, description="External service status")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error description")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
