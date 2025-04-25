"""
Embedding Store Module

This module handles the creation and management of text embeddings and vector database.
It uses Google's Gemini API for generating embeddings and FAISS for vector indexing.

Usage example:
    python embedding_store.py chunks.json --index-output faiss_index.bin
"""

import json
import numpy as np
import faiss
import os
import gc  # For garbage collection
import time
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import utility functions
from components.utils.utils import (
    print_step, print_success, print_error, print_warning, 
    print_info, validate_file_exists, print_progress_bar,
    limit_text_for_display
)

class EmbeddingStore:
    """
    Store and manage text embeddings using Google's Gemini API.
    
    This class handles:
    1. Generating embeddings for text chunks
    2. Creating and managing a FAISS vector database
    3. Saving and loading the database
    4. Searching for similar content based on vector similarity
    """
    
    def __init__(self):
        """
        Initialize embedding store with Google Gemini API configuration
        """
        # Load API key from environment variables
        print_info("Using Google Gemini API for embeddings")
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print_error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        # Configure Gemini API
        genai.configure(api_key=api_key)
        print_success("Gemini API configured successfully")
            
        # Set default dimension (will be updated after creating first embedding)
        self.dimension = 768  # Default for Gemini embedding model
        self.index = None
        self.chunks = []
        
        # Add caching for embeddings to improve performance
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_embedding(self, text, max_retries=3):
        """
        Get embedding for a text using Gemini API with retry mechanism and caching
        
        Args:
            text (str): Text to embed
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        # Check if input is already a numpy array (pre-computed embedding)
        if isinstance(text, np.ndarray):
            return text
            
        # Check if we have this text in our cache
        cache_key = hash(str(text))
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            if self.cache_hits % 20 == 0:  # Only log occasionally
                print_info(f"Embedding cache: {self.cache_hits} hits, {self.cache_misses} misses")
            return self.embedding_cache[cache_key]
            
        # Check text size in bytes to avoid API payload limits (36000 bytes)
        text_bytes = len(text.encode('utf-8'))
        max_bytes = 30000  # Set safe limit below the 36000 API limit
        
        # Trim texts to avoid API issues
        original_text = text
        if text_bytes > max_bytes:
            print_warning(f"Text too large ({text_bytes} bytes), truncating to fit API limit...")
            # Calculate approximate character count to fit under byte limit (with safety margin)
            char_limit = int(len(text) * (max_bytes / text_bytes) * 0.9)
            text = text[:char_limit]
            
            # Verify the truncated text is under the limit
            truncated_bytes = len(text.encode('utf-8'))
            if truncated_bytes > max_bytes:
                # Further truncate if still too large
                text = text[:int(char_limit * 0.9)]
                
            print_info(f"Truncated from {text_bytes} to {len(text.encode('utf-8'))} bytes")
        
        # Try with retries
        self.cache_misses += 1
        for retry in range(max_retries):
            try:
                # Get embedding using Gemini API
                result = genai.embed_content(
                    model="models/embedding-001",  # MUST use models/ prefix
                    content=text,
                    task_type="retrieval_query"
                )
                
                # Extract embedding from response (handle different response formats)
                embedding_vector = None
                if hasattr(result, 'embedding'):
                    embedding_vector = np.array(result.embedding, dtype=np.float32)
                elif isinstance(result, dict) and 'embedding' in result:
                    embedding_vector = np.array(result['embedding'], dtype=np.float32)
                elif hasattr(result, 'values'):
                    embedding_vector = np.array(result.values, dtype=np.float32)
                elif hasattr(result, 'embeddings') and result.embeddings:
                    embedding_vector = np.array(result.embeddings[0], dtype=np.float32)
                
                if embedding_vector is not None:
                    # Store in cache before returning
                    self.embedding_cache[cache_key] = embedding_vector
                    return embedding_vector
                else:
                    if retry < max_retries - 1:
                        print_warning(f"No embedding returned from API (attempt {retry+1}/{max_retries}). Retrying...")
                        time.sleep(1 * (retry + 1))  # Exponential backoff
                        continue
                    else:
                        print_error(f"No embedding returned from API after {max_retries} attempts")
                        return np.zeros(self.dimension, dtype=np.float32)
                    
            except Exception as e:
                if "payload size exceeds the limit" in str(e):
                    # Special handling for payload size errors
                    if retry < max_retries - 1:
                        print_warning(f"Payload size error (attempt {retry+1}/{max_retries}). Aggressive truncation...")
                        # More aggressive truncation for next attempt
                        char_limit = int(len(text) * 0.7)  # Take 70% of current length
                        text = text[:char_limit]
                        time.sleep(0.5)
                        continue
                    else:
                        # Last attempt with severe truncation
                        print_warning("Final attempt with severe truncation...")
                        # Severely truncate to just the beginning portion
                        text = original_text[:min(5000, len(original_text))]
                        try:
                            result = genai.embed_content(
                                model="models/embedding-001",
                                content=text,
                                task_type="retrieval_query"
                            )
                            
                            # Extract embedding
                            if hasattr(result, 'embedding'):
                                embedding_vector = np.array(result.embedding, dtype=np.float32)
                                self.embedding_cache[cache_key] = embedding_vector
                                print_success("Successfully embedded with severe truncation")
                                return embedding_vector
                        except Exception as final_e:
                            print_error(f"Final embedding attempt failed: {str(final_e)}")
                            return np.zeros(self.dimension, dtype=np.float32)
                elif retry < max_retries - 1:
                    print_warning(f"Embedding generation failed (attempt {retry+1}/{max_retries}): {str(e)}. Retrying...")
                    time.sleep(1 * (retry + 1))  # Exponential backoff
                    continue
                else:
                    print_error(f"Embedding generation failed after {max_retries} attempts: {str(e)}")
                    # Return zero vector as fallback to avoid breaking the pipeline
                    return np.zeros(self.dimension, dtype=np.float32)
        
        # This should never be reached but added as a safety
        return np.zeros(self.dimension, dtype=np.float32)
            
    def get_embeddings_batch(self, texts, batch_size=10, max_retries=3):
        """
        Get embeddings for multiple texts at once using batching for efficiency
        
        Args:
            texts (list): List of texts to embed
            batch_size (int): Size of each processing batch
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            list: List of embedding vectors
        """
        all_embeddings = []
        
        # Use ThreadPoolExecutor for parallel embedding generation
        def process_text(args):
            index, text = args
            
            # Check text size in bytes to avoid API payload limits (36000 bytes)
            text_bytes = len(text.encode('utf-8'))
            max_bytes = 30000  # Set safe limit below the 36000 API limit
            
            # Truncate if needed
            original_text = text
            if text_bytes > max_bytes:
                print_warning(f"Text at index {index} too large ({text_bytes} bytes), truncating...")
                # Calculate approximate character count to fit under byte limit
                char_limit = int(len(text) * (max_bytes / text_bytes) * 0.9)
                text = text[:char_limit]
                
                # Verify truncated content is under the limit
                truncated_bytes = len(text.encode('utf-8'))
                if truncated_bytes > max_bytes:
                    # Further truncate if still too large
                    text = text[:int(char_limit * 0.9)]
                
                print_info(f"Truncated text at index {index} from {text_bytes} to {len(text.encode('utf-8'))} bytes")
            
            # Apply retry logic
            for retry in range(max_retries):
                try:
                    # Get embedding
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_query"
                    )
                    
                    # Extract embedding using multiple possible response formats
                    if hasattr(result, 'embedding'):
                        return index, np.array(result.embedding, dtype=np.float32)
                    elif isinstance(result, dict) and 'embedding' in result:
                        return index, np.array(result['embedding'], dtype=np.float32)
                    elif hasattr(result, 'values'):
                        return index, np.array(result.values, dtype=np.float32)
                    elif hasattr(result, 'embeddings') and result.embeddings:
                        return index, np.array(result.embeddings[0], dtype=np.float32)
                    else:
                        if retry < max_retries - 1:
                            print_warning(f"No embedding returned for text at index {index} (attempt {retry+1}/{max_retries}). Retrying...")
                            time.sleep(1 * (retry + 1))  # Exponential backoff
                            continue
                        else:
                            print_warning(f"No embedding returned for text at index {index} after {max_retries} attempts.")
                            return index, np.zeros(self.dimension, dtype=np.float32)
                except Exception as e:
                    if "payload size exceeds the limit" in str(e):
                        # Special handling for payload size errors
                        if retry < max_retries - 1:
                            print_warning(f"Payload size error for text at index {index} (attempt {retry+1}/{max_retries}). Aggressive truncation...")
                            # More aggressive truncation for next attempt
                            char_limit = int(len(text) * 0.7)  # Take 70% of current length
                            text = text[:char_limit]
                            time.sleep(0.5)
                            continue
                        else:
                            # Last attempt with severe truncation
                            print_warning(f"Final attempt with severe truncation for text at index {index}...")
                            # Severely truncate to just the beginning portion
                            text = original_text[:min(5000, len(original_text))]
                            try:
                                result = genai.embed_content(
                                    model="models/embedding-001",
                                    content=text,
                                    task_type="retrieval_query"
                                )
                                
                                # Extract embedding
                                if hasattr(result, 'embedding'):
                                    embedding_vector = np.array(result.embedding, dtype=np.float32)
                                    print_success(f"Successfully embedded text at index {index} with severe truncation")
                                    return index, embedding_vector
                            except Exception as final_e:
                                print_error(f"Final embedding attempt failed for text at index {index}: {str(final_e)}")
                                return index, np.zeros(self.dimension, dtype=np.float32)
                    elif retry < max_retries - 1:
                        print_warning(f"Failed to embed text at index {index} (attempt {retry+1}/{max_retries}): {str(e)}. Retrying...")
                        time.sleep(1 * (retry + 1))  # Exponential backoff
                        continue
                    else:
                        print_warning(f"Failed to embed text at index {index} after {max_retries} attempts: {str(e)}")
                        return index, np.zeros(self.dimension, dtype=np.float32)
            
            # If we get here, all retries failed
            return index, np.zeros(self.dimension, dtype=np.float32)
        
        # Process in batches, each batch using parallel threads
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [None] * len(batch)
            
            # Create a list of (index, text) pairs
            tasks = [(i+j, text) for j, text in enumerate(batch)]
            
            # Use ThreadPoolExecutor to process in parallel
            with ThreadPoolExecutor(max_workers=min(len(batch), 5)) as executor:
                futures = [executor.submit(process_text, args) for args in tasks]
                
                for future in as_completed(futures):
                    try:
                        idx, embedding = future.result()
                        # Calculate the local index within the batch
                        local_idx = idx - i
                        batch_embeddings[local_idx] = embedding
                    except Exception as e:
                        print_error(f"Error processing embedding: {str(e)}")
            
            all_embeddings.extend(batch_embeddings)
            
            # Update progress on batch completion
            print_progress_bar(
                min(i + batch_size, len(texts)),
                len(texts),
                prefix='Creating embeddings:',
                suffix=f'Chunk {min(i + batch_size, len(texts))}/{len(texts)}'
            )
            
        return all_embeddings
            
    def create_embeddings(self, chunks_data):
        """
        Create embeddings for all chunks and build FAISS index
        
        Args:
            chunks_data (list): List of chunk dictionaries
            
        Returns:
            list: List of embedding vectors
        """
        print_info(f"Creating embeddings for {len(chunks_data)} chunks")
        
        # Store chunks data for later reference
        self.chunks = chunks_data
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks_data]
        
        # Generate embeddings with batched processing
        print_info("Using batch processing for embeddings")
        embeddings = self.get_embeddings_batch(texts)
        
        # Alternatively, fall back to individual processing if batching fails
        if not embeddings:
            print_warning("Batch embedding failed, falling back to individual processing")
            embeddings = []
            for i, text in enumerate(texts):
                # Update progress
                print_progress_bar(
                    i+1, 
                    len(texts), 
                    prefix='Creating embeddings:', 
                    suffix=f'Chunk {i+1}/{len(texts)}'
                )
                
                # Generate embedding
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
                
                # Free memory periodically
                if i % 100 == 0:
                    gc.collect()
            
        # Update dimension based on actual embedding size
        if embeddings and len(embeddings[0]) > 0:
            self.dimension = len(embeddings[0])
            print_info(f"Embedding dimension: {self.dimension}")
        else:
            print_error("No valid embeddings generated")
            raise ValueError("Failed to generate embeddings")
        
        # Create FAISS index
        print_info("Building FAISS index...")
        try:
            # Create a flat L2 index (exact search)
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Convert embeddings to correct format
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add ID mapping to track which embedding corresponds to which chunk
            self.index = faiss.IndexIDMap(self.index)
            self.index.add_with_ids(embeddings_array, np.array(range(len(embeddings))))
            
            print_success(f"FAISS index built with {len(embeddings)} vectors")
        except Exception as e:
            print_error(f"Failed to build FAISS index: {str(e)}")
            raise
        
        return embeddings
    
    def save(self, index_path='faiss_index.bin', chunks_path='processed_chunks.json'):
        """
        Save the FAISS index and chunks data to disk
        
        Args:
            index_path (str): Path to save the FAISS index
            chunks_path (str): Path to save the processed chunks
        """
        if self.index is None:
            print_error("No index to save. Call create_embeddings first.")
            raise ValueError("No index to save. Call create_embeddings first.")
            
        # Save FAISS index
        print_info(f"Saving FAISS index to {index_path}")
        try:
            faiss.write_index(self.index, index_path)
            print_success(f"FAISS index saved to {index_path}")
        except Exception as e:
            print_error(f"Failed to save FAISS index: {str(e)}")
            raise
        
        # Save chunks data
        print_info(f"Saving processed chunks to {chunks_path}")
        try:
            with open(chunks_path, 'w') as f:
                json.dump(self.chunks, f, indent=2)
            print_success(f"Processed chunks saved to {chunks_path}")
        except Exception as e:
            print_error(f"Failed to save chunks data: {str(e)}")
            raise
    
    def load(self, index_path='faiss_index.bin', chunks_path='processed_chunks.json'):
        """
        Load FAISS index and chunks data from files
        
        Args:
            index_path (str): Path to the FAISS index file
            chunks_path (str): Path to the processed chunks file
        """
        # Validate files exist
        validate_file_exists(index_path)
        validate_file_exists(chunks_path)
        
        # Load FAISS index
        print_info(f"Loading FAISS index from {index_path}")
        try:
            self.index = faiss.read_index(index_path)
            print_success(f"FAISS index loaded with {self.index.ntotal} vectors")
        except Exception as e:
            print_error(f"Failed to load FAISS index: {str(e)}")
            raise
        
        # Load chunks data
        print_info(f"Loading processed chunks from {chunks_path}")
        try:
            with open(chunks_path, 'r') as f:
                self.chunks = json.load(f)
            print_success(f"Loaded {len(self.chunks)} chunks")
        except Exception as e:
            print_error(f"Failed to load chunks data: {str(e)}")
            raise
            
    def search(self, query, top_k=5):
        """
        Search for similar chunks to a query
        
        Args:
            query (str or numpy.ndarray): Search query text or embedding vector
            top_k (int): Number of top matches to return
            
        Returns:
            list: List of dictionaries with chunks and metadata
        """
        if not self.index:
            print_error("No index loaded - call load() first")
            return []
        
        start_time = time.time()
        
        try:
            # Check if query is already an embedding vector
            if isinstance(query, np.ndarray):
                query_embedding = query.reshape(1, -1).astype('float32')
            else:
                # Check if query is a string before trying to display it
                if isinstance(query, str):
                    # If query is a string, print it and get its embedding
                    print_info(f"Searching for: {limit_text_for_display(query)}")
                    query_embedding = self.get_embedding(query)
                elif callable(query):
                    # Handle case where query is a function/method
                    print_warning("Query is a function/method object, not a string or embedding")
                    return []
                else:
                    # Handle other types
                    print_warning(f"Unexpected query type: {type(query)}")
                    try:
                        query_str = str(query)
                        print_info(f"Converted query to string: {limit_text_for_display(query_str)}")
                        query_embedding = self.get_embedding(query_str)
                    except:
                        print_error("Cannot process query with unexpected type")
                        return []
            
            # Debug the embedding type and shape
            print_info(f"Query embedding type: {type(query_embedding)}")
            
            # Handle different embedding formats
            if query_embedding is None:
                print_error("Failed to get query embedding")
                return []
            
            # Convert to numpy array if it's not already
            if not isinstance(query_embedding, np.ndarray):
                try:
                    # Try converting directly
                    query_embedding = np.array(query_embedding, dtype=np.float32)
                except Exception as convert_err:
                    print_error(f"Error converting embedding to numpy array: {convert_err}")
                    return []
            
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Print embedding shape for debugging
            print_info(f"Query embedding shape: {query_embedding.shape}")
            
            # Search the index
            D, I = self.index.search(query_embedding, top_k)
            
            # Get the original chunks for the indices
            results = []
            for i, idx in enumerate(I[0]):
                if idx < len(self.chunks):
                    results.append({
                        "score": float(D[0][i]),
                        "index": int(idx),
                        "chunk": self.chunks[idx]
                    })
            
            end_time = time.time()
            processing_time = end_time - start_time
            print_info(f"Found {len(results)} matches in {processing_time:.2f} seconds")
            
            return results
        except Exception as e:
            print_error(f"Search failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def test_embedding_api(self):
        """
        Validate that the embedding API is working correctly
        
        Returns:
            bool: True if successful, False if failed
        """
        print_info("Testing embedding API with a simple example...")
        test_text = "This is a test to verify that the embedding API is working correctly."
        
        try:
            # Try with the standard configuration
            result = genai.embed_content(
                model="models/embedding-001",
                content=test_text,
                task_type="retrieval_query"
            )
            
            # Check if we can extract an embedding
            if hasattr(result, 'embedding') or isinstance(result, dict) and 'embedding' in result:
                print_success("Embedding API test succeeded")
                return True
            else:
                print_warning("Could not extract embedding from API response")
                return False
        
        except Exception as e:
            print_error(f"Embedding API test failed: {str(e)}")
            return False

def build_vector_store(chunks_json_path, output_index='faiss_index.bin', 
                      output_chunks='processed_chunks.json', memory_efficient=False):
    """
    Build vector store from chunks JSON file
    
    Args:
        chunks_json_path (str): Path to chunks JSON file
        output_index (str): Path to save the FAISS index
        output_chunks (str): Path to save the processed chunks
        memory_efficient (bool): Enable memory efficient processing mode
        
    Returns:
        EmbeddingStore: Configured embedding store
    """
    print_step(2, "Building Vector Store")
    
    # Validate input file
    validate_file_exists(chunks_json_path)
    
    # Create embedding store and test API
    store = EmbeddingStore()
    
    # Test the embedding API first
    print_info("Testing the embedding API before proceeding")
    api_working = store.test_embedding_api()
    if not api_working:
        print_error("Embedding API test failed. Please check your API key and network connection.")
        raise RuntimeError("Embedding API test failed")
    
    # Load chunks
    print_info(f"Loading chunks from {chunks_json_path}")
    try:
        with open(chunks_json_path, 'r') as f:
            chunks = json.load(f)
        print_success(f"Loaded {len(chunks)} chunks")
    except Exception as e:
        print_error(f"Failed to load chunks: {str(e)}")
        raise
        
    # Create embeddings and index
    store.create_embeddings(chunks)
    
    # Save to disk
    store.save(output_index, output_chunks)
    print_success(f"Vector store built with {len(chunks)} chunks")
    
    return store

if __name__ == "__main__":
    """Run as standalone script to build vector store"""
    import argparse
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Build vector store from chunks")
    parser.add_argument("chunks_path", help="Path to the chunks JSON file")
    parser.add_argument("--index-output", default="faiss_index.bin", help="Output path for FAISS index")
    parser.add_argument("--chunks-output", default="processed_chunks.json", help="Output path for processed chunks")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Build the vector store
    build_vector_store(args.chunks_path, args.index_output, args.chunks_output) 