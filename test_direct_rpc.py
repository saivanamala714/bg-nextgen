#!/usr/bin/env python3
"""
Test script to directly test the Supabase RPC call.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from app.core.supabase_client import supabase_client
from app.core.config import settings

async def test_direct_rpc():
    """Test the RPC call directly."""
    
    print("🧪 Testing Direct Supabase RPC Call...")
    
    # Generate embedding
    model = SentenceTransformer('BAAI/bge-m3')
    query_text = "Bhagavad"
    
    embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    embedding_list = embedding.tolist()
    
    print(f"Query: {query_text}")
    print(f"Embedding dimensions: {len(embedding_list)}")
    print(f"First 5 values: {embedding_list[:5]}")
    print(f"Similarity threshold: {settings.similarity_threshold}")
    
    # Test the search directly with low threshold
    try:
        # Temporarily override the threshold
        original_threshold = settings.similarity_threshold
        settings.similarity_threshold = 0.0  # Use 0.0 to find any matches
        
        print(f"Using similarity threshold: {settings.similarity_threshold}")
        
        results = await supabase_client.search_documents(
            query_embedding=embedding_list,
            query_text=query_text,
            limit=5
        )
        
        # Restore original threshold
        settings.similarity_threshold = original_threshold
        
        print(f"\n📊 Search Results:")
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Similarity: {result.get('similarity', 'N/A')}")
            print(f"  PDF: {result.get('pdf_name', 'N/A')}")
            print(f"  Text preview: {result.get('chunk_text', 'N/A')[:100]}...")
            print()
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    results = asyncio.run(test_direct_rpc())
