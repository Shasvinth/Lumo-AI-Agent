"""
Components package for TealStor Multilingual Textbook Q&A system.

This package contains the core components for the TealStor system:
- RAG Processor: Handles query processing and answer generation
- Embedding Store: Manages text embeddings and vector search
- PDF Processor: Extracts text and metadata from PDFs
- Utils: Utility functions shared across components
"""

from components.processors.rag_processor import RAGProcessor
from components.processors.embedding_store import EmbeddingStore
from components.processors.pdf_processor import process_pdf
from components.utils.utils import *

__all__ = ['RAGProcessor', 'EmbeddingStore', 'process_pdf'] 