# RAG System Technical Documentation

## Introduction

This document provides a detailed technical overview of the Retrieval-Augmented Generation (RAG) system implemented in this repository. It explains the concepts, architecture, implementation details, and code structure to help developers understand and modify the system.

## Table of Contents

1. [RAG Concept Overview](#rag-concept-overview)
2. [System Architecture](#system-architecture)
3. [Component Deep-Dives](#component-deep-dives)
   - [PDF Processing](#pdf-processing)
   - [Embedding Generation](#embedding-generation)
   - [Vector Storage](#vector-storage)
   - [Query Processing](#query-processing)
4. [Code Structure](#code-structure)
5. [Key Methods Explained](#key-methods-explained)
6. [Implementation Choices](#implementation-choices)
7. [Performance Considerations](#performance-considerations)
8. [Extending the System](#extending-the-system)

## RAG Concept Overview

Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generation-based AI approaches:

1. **Retrieval Component**: Finds relevant information from a knowledge base (in our case, a PDF document).
2. **Generation Component**: Uses the retrieved information to generate accurate, contextually relevant answers.

RAG offers several advantages over pure generation approaches:
- Reduces hallucinations by grounding answers in retrieved text
- Provides source attribution to validate responses
- Updates knowledge without retraining the underlying model
- Works with domain-specific content not in the model's training data

## System Architecture

The system follows a three-stage pipeline architecture:

1. **Document Processing**
   - Input: PDF document
   - Process: Extract text, split into chunks, extract metadata
   - Output: JSON file with chunked text and metadata

2. **Vector Database Building**
   - Input: Chunked text from Step 1
   - Process: Generate embeddings, build FAISS index
   - Output: FAISS index file and processed chunks JSON

3. **Query Processing**
   - Input: User query and vector database from Step 2
   - Process: Retrieve relevant chunks, construct prompt, generate answer
   - Output: Generated answer with sources

## Component Deep-Dives

### PDF Processing

The PDF processor (`pdf_processor.py`) extracts text from PDF documents and divides it into manageable chunks for embedding.

**Key features:**
- Uses PyPDF2 for text extraction
- Implements overlapping chunks to preserve context
- Detects sections and page numbers for source attribution
- Handles formatting issues common in PDFs

**Chunking strategy:**
```python
# Pseudocode for chunking algorithm
for page in pdf:
    text = extract_text(page)
    chunks = split_text(text, chunk_size, overlap)
    for chunk in chunks:
        add_metadata(chunk, page_number, detect_section(chunk))
    
    store_chunks(chunks)
```

### Embedding Generation

The embedding store (`embedding_store.py`) converts text chunks into vector representations using Google's Gemini embedding API.

**Key features:**
- Batch processing for API efficiency
- Retry mechanism with exponential backoff
- Memory-efficient processing options
- Multiple response format handling

**Embedding API interaction:**
```python
def get_embedding(text):
    # Call Google's Gemini API
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    
    # Extract the embedding vector
    return np.array(result.embedding, dtype=np.float32)
```

### Vector Storage

The FAISS vector database enables efficient similarity search among text chunks.

**Key features:**
- Uses IndexFlatL2 for exact similarity search
- IndexIDMap for tracking chunk mappings
- Persistence capabilities for saving/loading
- Memory-mapped operations for large datasets

**Index creation:**
```python
# Create FAISS index for similarity search
index = faiss.IndexFlatL2(dimension)  # L2 distance
index = faiss.IndexIDMap(index)       # Add ID mapping
index.add_with_ids(embeddings, ids)   # Add vectors
```

### Query Processing

The RAG processor (`rag_processor.py`) ties everything together to process user queries.

**Key features:**
- Context retrieval based on semantic similarity
- Dynamic prompt construction
- Token management for optimal context usage
- Source attribution through metadata extraction

**RAG process flow:**
```python
def process_query(query):
    # 1. Retrieve relevant context
    query_embedding = get_embedding(query)
    contexts = search_similar(query_embedding)
    
    # 2. Build prompt with query and contexts
    prompt = construct_prompt(query, contexts)
    
    # 3. Generate answer
    answer = llm_generate(prompt)
    
    # 4. Extract and format metadata
    sections, pages = extract_metadata(contexts)
    
    return {
        "answer": answer,
        "sections": sections,
        "pages": pages,
        "context": contexts
    }
```

## Code Structure

### File Organization

- `main.py`: Entry point and pipeline orchestration
- `pdf_processor.py`: PDF text extraction and chunking
- `embedding_store.py`: Embedding generation and vector database
- `rag_processor.py`: Query processing and answer generation
- `utils.py`: Utility functions for all components
- `app.py`: Web interface implementation
- `run.py`: Command-line interface

### Class Hierarchy

- `PDFProcessor`: Handles document processing
- `EmbeddingStore`: Manages embeddings and vector search
- `RAGProcessor`: Processes queries and generates answers

## Key Methods Explained

### EmbeddingStore.get_embedding()

```python
def get_embedding(self, text, max_retries=3):
    """
    Get embedding for a text using Gemini API with retry mechanism
    
    Args:
        text (str): Text to embed
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        numpy.ndarray: Embedding vector
    """
```

This method:
1. Trims text to avoid API limits
2. Makes API call to Gemini with proper parameters
3. Implements retries with exponential backoff
4. Handles multiple response formats
5. Returns a normalized embedding vector

### EmbeddingStore.search()

```python
def search(self, query, top_k=5):
    """
    Search for most similar chunks to the query
    
    Args:
        query (str): Query text to search for
        top_k (int): Number of results to return
        
    Returns:
        list: List of dictionaries with chunk data and distance scores
    """
```

This method:
1. Generates an embedding for the query
2. Performs a FAISS search for similar vectors
3. Maps the results back to the original chunks
4. Returns chunks with similarity scores

### RAGProcessor.construct_prompt()

```python
def construct_prompt(self, query, contexts):
    """Constructs a prompt with the query and context chunks"""
```

This method:
1. Builds a template prompt with instructions
2. Adds the user query
3. Intelligently adds context chunks up to a token limit
4. Ensures the prompt is within model context window

## Implementation Choices

### Why FAISS for Vector Search?

FAISS was chosen because:
- High performance for large datasets
- Flexibility in index types
- Memory-mapping capabilities
- Wide industry adoption

### Batch Processing vs. Individual Processing

The system supports both batched and individual embedding generation:
- Batch processing: More efficient API usage, reduced latency
- Individual processing: Better error isolation, lower memory usage
- Fallback mechanism: Uses individual processing if batch fails

### Token Management

Context length is managed by:
- Estimating token count for each chunk
- Dynamically adding chunks until a limit is reached
- Prioritizing chunks by similarity score
- Truncating when necessary

## Performance Considerations

### Memory Optimization

- Garbage collection during processing
- Streaming for large files
- Memory-efficient mode option
- Index type selection based on dataset size

### Throughput Optimization

- Batch processing for API calls
- Exponential backoff for rate limits
- Intermediate result saving
- Progress tracking for long operations

## Extending the System

### Adding New Embedding Models

To use a different embedding model:
1. Update the `get_embedding()` method in `embedding_store.py`
2. Adjust the dimension parameter if needed
3. Modify the response handling for the new API

### Supporting Different Document Types

To add support for new document types:
1. Create a new processor similar to `pdf_processor.py`
2. Implement text extraction for the new format
3. Maintain the same chunk format with metadata

### Customizing the RAG Prompt

To modify how answers are generated:
1. Edit the `construct_prompt()` method in `rag_processor.py`
2. Update the prompt template
3. Adjust token limit calculations if needed

### Using Different Vector Search Algorithms

To change the vector search approach:
1. Modify the index creation in `embedding_store.py`
2. Replace `IndexFlatL2` with another FAISS index type
3. Adjust search parameters as needed

## Conclusion

This RAG implementation provides a flexible, efficient way to answer questions based on document content. The modular design allows for easy customization and extension to meet different use cases and requirements. 