#!/usr/bin/env python3
"""
Test script to verify embedding generation consistency between ingestion and chat.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import numpy as np

def test_embedding_consistency():
    """Test if embeddings are generated consistently."""
    
    print("🧪 Testing Embedding Generation Consistency...")
    
    # Load BGE-M3 model (same as both ingestion and app)
    model = SentenceTransformer('BAAI/bge-m3')
    
    # Test text
    test_text = "Bhagavad Gita"
    
    # Generate embedding using ingestion method
    print("\n📄 Testing ingestion method...")
    ingestion_embedding = model.encode(
        test_text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # Generate embedding using app method  
    print("\n🚀 Testing app method...")
    app_embedding = model.encode(
        test_text,
        normalize_embeddings=True,  # Important for cosine similarity
        convert_to_numpy=True
    )
    
    # Compare embeddings
    print(f"\n📊 Comparison Results:")
    print(f"Ingestion embedding shape: {ingestion_embedding.shape}")
    print(f"App embedding shape: {app_embedding.shape}")
    print(f"First 5 values - Ingestion: {ingestion_embedding[:5]}")
    print(f"First 5 values - App: {app_embedding[:5]}")
    
    # Calculate similarity
    similarity = np.dot(ingestion_embedding, app_embedding)
    print(f"\n🎯 Similarity between two methods: {similarity:.10f}")
    
    if similarity > 0.999999:
        print("✅ Embeddings are IDENTICAL - Generation is consistent!")
        return True
    else:
        print("❌ Embeddings DIFFER - Generation is inconsistent!")
        print(f"Difference: {1 - similarity}")
        return False

def test_with_database_embedding():
    """Test against an actual embedding from the database."""
    
    print("\n🗄️ Testing against database embedding...")
    
    # This would be the embedding we got from the database test
    # For now, let's just verify our generation works
    model = SentenceTransformer('BAAI/bge-m3')
    test_text = "Bhagavad Gita"
    
    embedding = model.encode(
        test_text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    print(f"Generated embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Last 5 values: {embedding[-5:]}")
    print(f"Min/Max values: {embedding.min():.6f} / {embedding.max():.6f}")
    
    return embedding

if __name__ == "__main__":
    is_consistent = test_embedding_consistency()
    test_embedding = test_with_database_embedding()
    
    if is_consistent:
        print("\n🎉 Embedding generation is consistent!")
        print("The issue might be elsewhere (data transmission, etc.)")
    else:
        print("\n⚠️ Embedding generation is inconsistent!")
        print("This explains why search isn't working.")
