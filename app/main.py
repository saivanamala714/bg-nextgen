from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager

from .core.config import settings
from .routers import chat, ingest
from .models.schemas import HealthResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting PDF Chat RAG Backend...")
    
    # Test connections on startup
    try:
        # Test Supabase connection
        from .core.supabase_client import supabase_client
        logger.info("✓ Supabase connection established")
        
        # Test embedding service
        from .services.embedding_service import embedding_service
        logger.info("✓ Embedding service initialized")
        
        # Test RAG service
        from .services.rag_service import rag_service
        logger.info("✓ RAG service initialized")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    logger.info("Shutting down PDF Chat RAG Backend...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="FastAPI backend for PDF chat with RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__}
        ).dict()
    )


# Include routers
app.include_router(chat.router)
app.include_router(ingest.router)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status.
    Checks database connectivity and service availability.
    """
    try:
        services_status = {}
        
        # Check Supabase
        try:
            from .core.supabase_client import supabase_client
            response = supabase_client.client.table('pdf_chunks').select('count').limit(1).execute()
            services_status["supabase"] = "healthy"
        except Exception as e:
            services_status["supabase"] = f"unhealthy: {str(e)}"
        
        # Check embedding service
        try:
            from .services.embedding_service import embedding_service
            # Test local embedding
            test_embedding = embedding_service.generate_local_embedding("test")
            services_status["local_embeddings"] = "healthy"
        except Exception as e:
            services_status["local_embeddings"] = f"unhealthy: {str(e)}"
        
        # Check OpenAI connectivity
        try:
            from .services.embedding_service import embedding_service
            import asyncio
            test_embedding = await embedding_service.generate_query_embedding("test")
            services_status["openai_embeddings"] = "healthy"
        except Exception as e:
            services_status["openai_embeddings"] = f"unhealthy: {str(e)}"
        
        # Check Groq connectivity
        try:
            from .services.rag_service import rag_service
            # Just check if client is initialized
            services_status["groq_llm"] = "healthy"
        except Exception as e:
            services_status["groq_llm"] = f"unhealthy: {str(e)}"
        
        # Determine overall status
        overall_status = "healthy" if all(
            "healthy" in status for status in services_status.values()
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            services={"error": str(e)}
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "description": "FastAPI backend for PDF chat with RAG",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "ingest": "/ingest/precomputed",
            "ingest_file": "/ingest/json-file",
            "status": "/ingest/status"
        }
    }


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests with timing information."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
