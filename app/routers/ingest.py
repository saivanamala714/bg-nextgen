from fastapi import APIRouter, HTTPException, status, UploadFile, File
from typing import List, Dict, Any
import json
import logging
import os

from ..models.schemas import IngestRequest, IngestResponse, PDFChunk
from ..core.supabase_client import supabase_client
from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/precomputed", response_model=IngestResponse)
async def ingest_precomputed_embeddings(request: IngestRequest):
    """
    Ingest precomputed embeddings from local processing.
    This is the recommended approach for production deployments.
    """
    try:
        logger.info(f"Ingesting {len(request.chunks)} precomputed chunks")
        
        # Convert Pydantic models to dicts for database insertion
        chunks_data = []
        for chunk in request.chunks:
            chunk_dict = {
                'pdf_name': request.pdf_name,
                'chunk_text': chunk.chunk_text,
                'embedding': chunk.embedding,
                'metadata': chunk.metadata
            }
            chunks_data.append(chunk_dict)
        
        # Insert chunks into database
        inserted_count = await supabase_client.insert_chunks(chunks_data)
        
        logger.info(f"Successfully inserted {inserted_count} chunks")
        
        return IngestResponse(
            success=True,
            inserted_count=inserted_count,
            message=f"Successfully ingested {inserted_count} chunks from {request.pdf_name}"
        )
        
    except Exception as e:
        logger.error(f"Error ingesting precomputed embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest embeddings: {str(e)}"
        )


@router.post("/json-file", response_model=IngestResponse)
async def ingest_json_file(file: UploadFile = File(...)):
    """
    Ingest embeddings from a JSON file upload.
    JSON file should contain the same format as local_ingest/embeddings_to_upload.json
    """
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a JSON file"
            )
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Read and parse JSON file
        contents = await file.read()
        try:
            data = json.loads(contents)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON file: {str(e)}"
            )
        
        # Validate data structure
        if not isinstance(data, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON file must contain a list of chunk objects"
            )
        
        # Convert to PDFChunk objects for validation
        chunks = []
        for i, item in enumerate(data):
            try:
                chunk = PDFChunk(**item)
                chunks.append(chunk)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid chunk at index {i}: {str(e)}"
                )
        
        # Use the same logic as precomputed endpoint
        request = IngestRequest(chunks=chunks, pdf_name=file.filename.replace('.json', '.pdf'))
        return await ingest_precomputed_embeddings(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting JSON file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}"
        )


@router.get("/status")
async def get_ingestion_status():
    """
    Get current ingestion status and statistics.
    """
    try:
        # Get chunk count from database
        response = supabase_client.client.table('pdf_chunks').select('count', count='exact').execute()
        chunk_count = response.count if response.count else 0
        
        # Get unique PDF count
        pdf_response = supabase_client.client.table('pdf_chunks').select('pdf_name').execute()
        unique_pdfs = len(set(chunk['pdf_name'] for chunk in pdf_response.data)) if pdf_response.data else 0
        
        return {
            'total_chunks': chunk_count,
            'unique_documents': unique_pdfs,
            'embedding_dimension': settings.embedding_dimension,
            'status': 'ready' if chunk_count > 0 else 'no_data'
        }
        
    except Exception as e:
        logger.error(f"Error getting ingestion status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/clear")
async def clear_all_data():
    """
    Clear all ingested data from the database.
    WARNING: This is a destructive operation!
    """
    try:
        logger.warning("Clearing all ingested data from database")
        
        # Delete all chunks
        chunks_response = supabase_client.client.table('pdf_chunks').delete().execute()
        deleted_chunks = len(chunks_response.data) if chunks_response.data else 0
        
        # Optionally delete chat logs (comment out if you want to preserve them)
        # logs_response = supabase_client.client.table('chat_logs').delete().execute()
        # deleted_logs = len(logs_response.data) if logs_response.data else 0
        
        return {
            'deleted_chunks': deleted_chunks,
            'message': f'Successfully deleted {deleted_chunks} chunks'
        }
        
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
