from typing import List, Dict, Any
import openai
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import torch
from ..core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using local and OpenAI models."""
    
    def __init__(self):
        # Initialize local BGE-M3 model for PDF processing
        self.local_model = None
        self._load_local_model()
        
        # Initialize OpenAI client for query embeddings
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
    
    def _load_local_model(self) -> None:
        """Load the BGE-M3 model for local embeddings."""
        try:
            logger.info("Loading BGE-M3 model for local embeddings...")
            self.local_model = SentenceTransformer('BAAI/bge-m3')
            logger.info("BGE-M3 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            raise
    
    def generate_local_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using local BGE-M3 model.
        Used for PDF chunk embeddings during ingestion.
        """
        try:
            # Generate embedding
            embedding = self.local_model.encode(
                text,
                normalize_embeddings=True,  # Important for cosine similarity
                convert_to_numpy=True
            )
            
            # Convert to list and ensure correct dimension
            embedding_list = embedding.tolist()
            
            # Validate dimension
            if len(embedding_list) != settings.embedding_dimension:
                logger.warning(f"Embedding dimension mismatch: {len(embedding_list)} vs {settings.embedding_dimension}")
                # Pad or truncate if necessary
                if len(embedding_list) < settings.embedding_dimension:
                    embedding_list.extend([0.0] * (settings.embedding_dimension - len(embedding_list)))
                else:
                    embedding_list = embedding_list[:settings.embedding_dimension]
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error generating local embedding: {e}")
            raise
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding using local BGE-M3 model.
        Used for query embeddings during chat to match stored PDF embeddings.
        """
        try:
            # Use the same local model for consistency
            embedding = self.local_model.encode(
                query,
                normalize_embeddings=True,  # Important for cosine similarity
                convert_to_numpy=True
            )
            
            # Convert to list and ensure correct dimension
            embedding_list = embedding.tolist()
            
            # Validate dimension
            if len(embedding_list) != settings.embedding_dimension:
                logger.warning(f"Embedding dimension mismatch: {len(embedding_list)} vs {settings.embedding_dimension}")
                # Pad or truncate if necessary
                if len(embedding_list) < settings.embedding_dimension:
                    embedding_list.extend([0.0] * (settings.embedding_dimension - len(embedding_list)))
                else:
                    embedding_list = embedding_list[:settings.embedding_dimension]
            
            logger.info(f"Generated BGE-M3 query embedding with {len(embedding_list)} dimensions")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using local model.
        Optimized for batch processing during PDF ingestion.
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            
            # Process in batches to manage memory
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings for batch
                batch_embeddings = self.local_model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
                
                # Convert to lists and validate dimensions
                for embedding in batch_embeddings:
                    embedding_list = embedding.tolist()
                    
                    # Ensure correct dimension
                    if len(embedding_list) != settings.embedding_dimension:
                        if len(embedding_list) < settings.embedding_dimension:
                            embedding_list.extend([0.0] * (settings.embedding_dimension - len(embedding_list)))
                        else:
                            embedding_list = embedding_list[:settings.embedding_dimension]
                    
                    all_embeddings.append(embedding_list)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            logger.info(f"Generated {len(all_embeddings)} embeddings total")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0


# Global embedding service instance
embedding_service = EmbeddingService()
