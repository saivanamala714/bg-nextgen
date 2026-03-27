#!/usr/bin/env python3
"""
Split large embeddings JSON file into smaller batches for upload.
"""

import json
import math
from pathlib import Path

def split_embeddings_file(input_file: str, output_dir: str, batch_size: int = 100):
    """Split embeddings JSON file into smaller batches."""
    
    # Load the large JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} chunks from {input_file}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Split into batches
    num_batches = math.ceil(len(data) / batch_size)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        
        output_file = Path(output_dir) / f"batch_{i+1:03d}_of_{num_batches:03d}.json"
        
        with open(output_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        print(f"Created batch {i+1}/{num_batches}: {output_file.name} ({len(batch_data)} chunks)")

if __name__ == "__main__":
    input_file = "embeddings_to_upload.json"
    output_dir = "batches"
    batch_size = 50  # Smaller batches to stay under 50MB limit
    
    split_embeddings_file(input_file, output_dir, batch_size)
    print(f"\n✅ Split complete! Upload files from '{output_dir}' directory.")
