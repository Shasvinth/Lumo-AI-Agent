"""
RAG Processor Module

This module handles the RAG (Retrieval-Augmented Generation) process:
1. Retrieving relevant contexts for a query
2. Constructing prompts with the retrieved contexts
3. Generating answers using the Gemini API
4. Formatting results with metadata

Usage example:
    python rag_processor.py queries.json --output queries_output.csv
"""

import os
import json
import pandas as pd
import time  # For rate limiting
import gc  # For garbage collection
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np

from components.processors.embedding_store import EmbeddingStore
from components.utils.utils import (
    print_step, print_success, print_error, print_warning, 
    print_info, validate_file_exists, print_progress_bar,
    limit_text_for_display, format_metadata
)

class RAGProcessor:
    """
    Process queries using RAG (Retrieval-Augmented Generation).
    
    This class handles the core RAG workflow:
    1. Load chunks and embeddings from a vector store
    2. For each query, retrieve relevant contexts
    3. Construct a prompt combining the query and contexts
    4. Generate an answer using Gemini API
    5. Format the results with metadata
    """
    
    def __init__(self, index_path, chunks_path):
        """Initialize the RAG processor with a vector store and chunks."""
        print_step(1, "Initializing RAG Processor")
        
        # Set up Google Gemini API access
        google_api_key = os.getenv("GEMINI_API_KEY")
        
        if not google_api_key:
            print_error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=google_api_key)
        
        # Set up Gemini models - use 1.5-flash for faster processing
        try:
            # Use Gemini 1.5 Flash for faster processing with good quality
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash", 
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            print_success("Initialized Gemini-1.5-Flash model for response generation")
            
            # Keep the embedding model the same
            self.embedding_model = genai.GenerativeModel('embedding-001')
            print_success("Initialized embedding model")
        except Exception as e:
            print_error(f"Error initializing Gemini models: {e}")
            raise
        
        # Set up vector store for embeddings
        try:
            self.vector_store = EmbeddingStore()
            self.vector_store.load(index_path=index_path, chunks_path=chunks_path)
            print_success("Vector store loaded successfully")
        except Exception as e:
            print_error(f"Error loading vector store: {str(e)}")
            raise ValueError(f"Failed to load vector store: {str(e)}")
        
        # Create caches for performance
        self.embedding_cache = {}  # Cache for embeddings
        self.query_cache = {}      # Cache for query results
        self.cache_hits = 0
        self.cache_misses = 0
        
        print_success(f"RAG Processor initialized successfully")
            
    def detect_language(self, text):
        """
        Detect the language of the input text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Language code (en, ta, si)
        """
        try:
            lang = detect(text)
            # Map detected language to our supported languages
            lang_map = {
                'en': 'en',  # English
                'ta': 'ta',  # Tamil
                'si': 'si'   # Sinhala
            }
            return lang_map.get(lang, 'en')  # Default to English if not supported
        except LangDetectException:
            return 'en'  # Default to English if detection fails
            
    def translate_text(self, text, target_lang='en'):
        """
        Translate text to target language
        
        Args:
            text (str): Text to translate
            target_lang (str): Target language code
            
        Returns:
            str: Translated text
        """
        if target_lang == 'en':
            return text
            
        try:
            # Map language codes to Google Translate codes
            lang_map = {
                'ta': 'ta',  # Tamil
                'si': 'si'   # Sinhala
            }
            google_lang = lang_map.get(target_lang, 'en')
            
            # Translate using Google Translate
            translator = GoogleTranslator(source='auto', target=google_lang)
            translation = translator.translate(text)
            return translation
        except Exception as e:
            print_warning(f"Translation failed: {str(e)}")
            return text
            
    def generate_embeddings(self, content):
        """Generate embeddings for a string using Google's Gemini embedding model with caching."""
        if not isinstance(content, str):
            print_error(f"Content must be a string, got {type(content)}")
            return None
            
        # Create a cache key by hashing the content
        cache_key = hash(content)
        
        # Check if we have this embedding cached
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            cached_embedding = self.embedding_cache[cache_key]
            
            # Extra validation to ensure we don't return a callable from cache
            if callable(cached_embedding):
                print_error("Cached embedding is callable, which is invalid")
                del self.embedding_cache[cache_key]  # Remove invalid entry
                # Fall through to regenerate the embedding
            else:
                if self.cache_hits % 10 == 0:  # Only log occasionally to reduce output
                    print_info(f"Embedding cache hit ({self.cache_hits} hits, {self.cache_misses} misses)")
                return cached_embedding
        
        # If not in cache, generate the embedding
        try:
            self.cache_misses += 1
            
            # Check if content exceeds the size limit (approx. 30000 bytes to be safe)
            content_bytes = len(content.encode('utf-8'))
            max_bytes = 30000  # Set a safe limit below the 36000 API limit
            
            if content_bytes > max_bytes:
                print_warning(f"Content size ({content_bytes} bytes) exceeds embedding limit. Truncating...")
                
                # Truncate by characters to stay under the limit
                # Use a sliding window approach for truncation
                char_limit = int(len(content) * (max_bytes / content_bytes) * 0.9)  # 90% safety factor
                
                # Take first portion which typically contains the most relevant info
                truncated_content = content[:char_limit]
                
                # Verify truncated content is under the limit
                if len(truncated_content.encode('utf-8')) > max_bytes:
                    # Further truncate if still too large
                    truncated_content = truncated_content[:int(char_limit * 0.9)]
                
                print_info(f"Truncated content to {len(truncated_content)} chars ({len(truncated_content.encode('utf-8'))} bytes)")
                content = truncated_content
            
            # Use the correct embedding API call - genai.embed_content instead of model.embed_content
            result = genai.embed_content(
                model="models/embedding-001",
                content=content,
                task_type="retrieval_query"
            )
            
            # Extract and properly format embedding values
            embedding_values = None
            
            # Debug the result structure
            print_info(f"Embedding result type: {type(result)}")
            
            # Handle the result based on its type
            if isinstance(result, dict):
                print_info("Result is a dictionary, using dictionary access")
                if 'embedding' in result:
                    embedding_values = result['embedding']
                    print_info("Using result['embedding']")
                elif 'values' in result:
                    embedding_values = result['values']
                    print_info("Using result['values']")
            # For object with attributes
            elif hasattr(result, 'embedding'):
                print_info("Using result.embedding attribute")
                embedding_values = result.embedding
                # Check if embedding is itself an object with values
                if hasattr(embedding_values, 'values') and not callable(embedding_values.values):
                    print_info("Using result.embedding.values attribute")
                    embedding_values = embedding_values.values
            elif hasattr(result, 'values') and not callable(result.values):
                print_info("Using result.values attribute")
                embedding_values = result.values
            
            # Final validation - ensure we don't have a callable
            if embedding_values is None:
                print_error("Could not extract embedding values from result")
                return None
            if callable(embedding_values):
                print_error("Extracted embedding is a callable, not valid embedding data")
                return None
                
            # Convert embedding to numpy array if it's not already
            if not isinstance(embedding_values, np.ndarray):
                try:
                    print_info(f"Converting embedding from {type(embedding_values)} to numpy array")
                    embedding_values = np.array(embedding_values, dtype=np.float32)
                except Exception as e:
                    print_error(f"Failed to convert embedding to numpy array: {str(e)}")
                    return None
            
            # Store in cache before returning
            self.embedding_cache[cache_key] = embedding_values
            
            # Log cache stats occasionally
            if self.cache_misses % 10 == 0:
                print_info(f"Embedding cache miss ({self.cache_hits} hits, {self.cache_misses} misses)")
                
            return embedding_values
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            print_warning(f"Embedding generation failed: {str(e)}")
            
            # If the error is about payload size, attempt more aggressive truncation
            if "payload size exceeds the limit" in str(e):
                try:
                    print_info("Attempting more aggressive truncation for large content...")
                    # Get approximately 40% of the content (from the beginning, which typically has the most relevant info)
                    severe_truncation = content[:min(len(content) // 2, 10000)]
                    
                    # Use the correct embedding API call with severely truncated content
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=severe_truncation,
                        task_type="retrieval_query"
                    )
                    
                    # Extract embedding values using the same approach
                    embedding_values = None
                    
                    # Debug the result structure
                    print_info(f"Truncated embedding result type: {type(result)}")
                    
                    # Handle the result based on its type
                    if isinstance(result, dict):
                        print_info("Truncated result is a dictionary, using dictionary access")
                        if 'embedding' in result:
                            embedding_values = result['embedding']
                            print_info("Using truncated result['embedding']")
                        elif 'values' in result:
                            embedding_values = result['values']
                            print_info("Using truncated result['values']")
                    # For object with attributes
                    elif hasattr(result, 'embedding'):
                        print_info("Using truncated result.embedding attribute")
                        embedding_values = result.embedding
                        # Check if embedding is itself an object with values
                        if hasattr(embedding_values, 'values') and not callable(embedding_values.values):
                            print_info("Using truncated result.embedding.values attribute")
                            embedding_values = embedding_values.values
                    elif hasattr(result, 'values') and not callable(result.values):
                        print_info("Using truncated result.values attribute")
                        embedding_values = result.values
                    
                    # Final validation - ensure we don't have a callable
                    if embedding_values is None:
                        print_error("Could not extract embedding values from truncated result")
                        return None
                    if callable(embedding_values):
                        print_error("Extracted truncated embedding is a callable, not valid embedding data")
                        return None
                    
                    # Convert embedding to numpy array if it's not already
                    if not isinstance(embedding_values, np.ndarray):
                        try:
                            print_info(f"Converting truncated embedding from {type(embedding_values)} to numpy array")
                            embedding_values = np.array(embedding_values, dtype=np.float32)
                        except Exception as e:
                            print_error(f"Failed to convert truncated embedding to numpy array: {str(e)}")
                            return None
                    
                    # We don't cache this result since it's a fallback
                    print_success("Successfully generated embedding with severe truncation")
                    return embedding_values
                except Exception as inner_e:
                    print_error(f"Even with severe truncation, embedding failed: {str(inner_e)}")
            
            return None

    def _format_contexts(self, contexts):
        """Format contexts with their titles, sources, and page references."""
        formatted_contexts = ""
        for idx, context in enumerate(contexts, 1):
            context_text = context.get("text", "")
            context_title = context.get("title", "")
            context_source = context.get("source", "")
            page_num = context.get("page_num", "")
            section = context.get("section", "")
            descriptive_title = context.get("descriptive_title", "")
            
            source_info = []
            if descriptive_title:
                source_info.append(f"Title: {descriptive_title}")
            elif context_title:
                source_info.append(f"Title: {context_title}")
            if context_source:
                source_info.append(f"Source: {context_source}")
            if page_num:
                source_info.append(f"Page: {page_num}")
            if section:
                source_info.append(f"Section: {section}")
            
            source_str = ", ".join(source_info)
            if source_str:
                formatted_contexts += f"CONTEXT {idx} ({source_str}):\n{context_text}\n\n"
            else:
                formatted_contexts += f"CONTEXT {idx}:\n{context_text}\n\n"
            
        return formatted_contexts

    def determine_query_type(self, query):
        """
        Determine the type of query based on its content.
        This is a simple implementation that can be expanded.
        
        Args:
            query (str): Query text
            
        Returns:
            str: Query type (default, definition, comparison, etc.)
        """
        query = query.lower()
        
        # Default to regular query type
        query_type = "regular"
        
        # Check for definition queries
        if any(pattern in query for pattern in [
            "what is", "what are", "define", "definition of", "meaning of", 
            "explain", "describe", "tell me about"
        ]):
            query_type = "definition"
            
        # Check for comparison queries
        elif any(pattern in query for pattern in [
            "compare", "difference between", "similarities between", 
            "versus", "vs", "similarities and differences"
        ]):
            query_type = "comparison"
            
        # Check for list/enumeration queries
        elif any(pattern in query for pattern in [
            "list", "enumerate", "what are the", "steps in", "stages of", 
            "types of", "kinds of", "examples of"
        ]):
            query_type = "list"
        
        return query_type

    def construct_prompt(self, query, contexts, query_id):
        """
        Constructs a prompt for an LLM based on the retrieved contexts.
        
        Args:
            query (str): The user's query
            contexts (str): Formatted context string with all relevant text chunks
            query_id (str): Unique ID for the query
            
        Returns:
            str: A formatted prompt with instructions and context
        """
        # Determine query type to optimize prompt
        query_type = self.determine_query_type(query)
        
        # Log query type for debugging
        print_info(f"Query type detected: {query_type}")
        
        # Base template instructions are the same regardless of query type
        base_instructions = """
You are an AI assistant for textbooks. Your job is to:
1. Accurately answer questions based ONLY on the information in the provided texts
2. If the information is not in the texts, explain what specific information you would need to answer properly, and suggest a more specific query the user could try
3. Do NOT repeat "I don't have information" for each source - just say it once
4. If web search results are provided but don't contain information relevant to the query, suggest related topics that might be more fruitful
5. Never make up information or use your own knowledge
6. Extract and provide specific page numbers and section references where available
7. When answering questions, provide proper citations by mentioning specific sections or pages from your contexts
8. Be concise but complete in your answers
"""
        
        # Use a streamlined prompt format for faster processing
        prompt = f"""
{base_instructions}

QUERY: {query}

{contexts}

INSTRUCTIONS:
- Answer the query accurately and completely based only on the provided contexts
- Format your answer to be readable and well-structured
- If the information to answer the query is not provided in the contexts, suggest what specific information would help and recommend a better query
- Provide page numbers and section references when possible
- Query ID: {query_id}

YOUR ANSWER:
"""
        
        return prompt
    
    def extract_metadata(self, contexts):
        """
        Extract section titles and page numbers from contexts
        
        Args:
            contexts (list): List of context chunks
            
        Returns:
            tuple: (list of sections, list of page numbers)
        """
        sections = set()
        pages = set()
        
        for context in contexts:
            chunk = context["chunk"]
            if "section" in chunk:
                # Convert to string to avoid type errors during join
                sections.add(str(chunk["section"]))
            elif "source" in chunk:
                # Convert to string to avoid type errors during join
                sections.add(str(chunk["source"]))
            
            # Check if pages key exists before trying to access it
            if "pages" in chunk:
                for page in chunk["pages"]:
                    # Convert to string to avoid type errors during join
                    pages.add(str(page))
            elif "source_url" in chunk:
                # For web sources, use the source URL as a "page"
                pages.add(f"Source: {chunk.get('source_url', 'Web')}")
                
        return list(sections), sorted(list(pages))
    
    def process_query(self, query_id, query_text, top_k=5, language=None):
        """
        Process a query and generate a response using RAG.
        
        Args:
            query_id (str): Unique ID for the query
            query_text (str): The query text
            top_k (int): Number of top contexts to retrieve
            language (str): Target language for response
        
        Returns:
            dict: Response containing Answer, Pages, Sections, Language
        """
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Check and handle source language
        language_info = {}
        original_language = None
        
        start_detect = time.time()
        if language and language != 'en':
            # Only detect if we need to translate non-English
            language_info = self.detect_language(query_text)
            original_language = language_info.get('language_code')
            if original_language and original_language != 'en':
                # Only translate if detected language isn't English
                translated_query = self.translate_text(query_text, 'en')
                if translated_query:
                    print_info(f"Translated query from {original_language} to English: {translated_query}")
                    query_text = translated_query
        detect_time = time.time() - start_detect
        print_info(f"Language detection took {detect_time:.2f}s")
        
        # Generate response using RAG
        try:
            # Create cache key for this query to avoid re-processing identical queries
            cache_key = f"{hash(query_text)}_{top_k}"
            if cache_key in self.query_cache:
                print_info("Using cached query result")
                cached_result = self.query_cache[cache_key]
                
                # If language is different from cached, only translate the answer
                if language and language != cached_result.get('language', 'en'):
                    print_info(f"Translating cached result to {language}")
                    cached_result['answer'] = self.translate_text(cached_result['answer'], language)
                    cached_result['language'] = language
                
                # Update timing info for cached result
                process_time = time.time() - start_time
                print_success(f"Query processed from cache in {process_time:.2f}s")
                return cached_result
                
            # Generate embedding for the query
            embedding_start = time.time()
            query_embedding = self.generate_embeddings(query_text)
            if query_embedding is None:
                return {"answer": "Failed to generate query embedding.", "context": "", "pages": [], "sections": [], "language": language or "en"}
            embedding_time = time.time() - embedding_start
            print_info(f"Query embedding took {embedding_time:.2f}s")
            
            # Retrieve similar contexts
            retrieval_start = time.time()
            try:
                # Verify query_embedding is a valid type before searching
                if query_embedding is None:
                    print_error("Query embedding is None, cannot search")
                    return {"answer": "Error: Could not generate a valid embedding for your query.", "context": "", "pages": [], "sections": [], "language": language or "en"}
                
                if callable(query_embedding):
                    print_error("Query embedding is a callable object, not a valid embedding")
                    return {"answer": "Error: Query embedding has an invalid format.", "context": "", "pages": [], "sections": [], "language": language or "en"}
                
                # Now search with the validated embedding
                contexts = self.vector_store.search(query_embedding, top_k=top_k)
                if contexts is None:
                    return {"answer": "Error retrieving contexts.", "context": "", "pages": [], "sections": [], "language": language or "en"}
            except Exception as search_error:
                print_error(f"Error during context search: {str(search_error)}")
                import traceback
                traceback.print_exc()
                return {"answer": f"Error searching for relevant information: {str(search_error)}", "context": "", "pages": [], "sections": [], "language": language or "en"}
                
            retrieval_time = time.time() - retrieval_start
            print_info(f"Context retrieval took {retrieval_time:.2f}s ({len(contexts)} contexts)")
            
            if not contexts:
                return {"answer": "I couldn't find specific information about this topic in the available sources. Try rephrasing your question to be more specific, or check if your query relates to content in the selected sources. Consider focusing on key terms or concepts that might appear in the available materials.", "context": "", "pages": [], "sections": [], "language": language or "en"}
            
            # Extract metadata for response
            metadata_start = time.time()
            pages, sections = self.extract_metadata(contexts)
            metadata_time = time.time() - metadata_start
            print_info(f"Metadata extraction took {metadata_time:.2f}s")
            
            # Format contexts for prompt
            format_start = time.time()
            formatted_contexts = self._format_contexts(contexts)
            format_time = time.time() - format_start
            print_info(f"Context formatting took {format_time:.2f}s")
            
            # Check if contexts actually contain meaningful content
            has_meaningful_content = False
            for ctx in contexts:
                if 'chunk' in ctx and 'text' in ctx['chunk'] and len(ctx['chunk']['text'].strip()) > 50:
                    has_meaningful_content = True
                    break
                    
            if not has_meaningful_content:
                print_warning(f"Retrieved contexts contain little or no meaningful content")
                return {"answer": "While I found some references that might relate to your query, they don't contain enough specific information to provide a proper answer. Try being more specific in your question, or check if this topic is covered in the available materials. You might want to try a different query focusing on related concepts.", "context": formatted_contexts, "pages": pages, "sections": sections, "language": language or "en"}
            
            # Choose prompt template and construct prompt
            prompt_start = time.time()
            prompt = self.construct_prompt(query_text, formatted_contexts, query_id)
            prompt_time = time.time() - prompt_start
            print_info(f"Prompt construction took {prompt_time:.2f}s")
            
            # Generate response with Gemini
            generation_start = time.time()
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            generation_time = time.time() - generation_start
            print_info(f"Answer generation took {generation_time:.2f}s")
            
            # Translate response if needed
            translation_time = 0
            if language and language != 'en':
                translation_start = time.time()
                translated_answer = self.translate_text(answer, language)
                if translated_answer:
                    answer = translated_answer
                translation_time = time.time() - translation_start
                print_info(f"Answer translation took {translation_time:.2f}s")
            
            # Construct final result with lowercase keys
            result = {
                "answer": answer,
                "context": formatted_contexts,
                "pages": pages,  # Store as list
                "sections": sections,  # Store as list
                "language": language or "en"
            }
            
            # Cache result for future identical queries
            self.query_cache[cache_key] = result.copy()
            
            # Track performance
            total_time = time.time() - start_time
            print_success(f"Query processed in {total_time:.2f}s")
            print_info(f"Time breakdown: Detection={detect_time:.2f}s, Embedding={embedding_time:.2f}s, " +
                      f"Retrieval={retrieval_time:.2f}s, Metadata={metadata_time:.2f}s, " +
                      f"Formatting={format_time:.2f}s, Prompt={prompt_time:.2f}s, " +
                      f"Generation={generation_time:.2f}s, Translation={translation_time:.2f}s")
            
            # Return the same result object
            return result
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            print_error(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error processing query: {str(e)}",
                "context": "",
                "pages": [],
                "sections": [],
                "language": language or "en"
            }
    
    def process_queries_file(self, queries_path, output_csv="queries_output.csv", top_k=5):
        """
        Process all queries from a JSON file
        
        Args:
            queries_path (str): Path to queries JSON file
            output_csv (str): Path to output CSV file
            top_k (int): Number of contexts to retrieve per query
            
        Returns:
            list: List of query results
        """
        print_step(3, "Processing Queries")
        
        # Validate files
        validate_file_exists(queries_path)
        
        # Load queries
        print_info(f"Loading queries from {queries_path}")
        try:
            with open(queries_path, 'r') as f:
                queries = json.load(f)
            print_success(f"Loaded {len(queries)} queries")
        except Exception as e:
            print_error(f"Failed to load queries: {str(e)}")
            raise
            
        # Process each query
        results = []
        for i, query in enumerate(queries):
            # Update progress
            print_progress_bar(
                i+1, 
                len(queries), 
                prefix='Processing queries:', 
                suffix=f'Query {i+1}/{len(queries)}'
            )
            
            # Process query
            result = self.process_query(query["ID"], query["query"], top_k=top_k)
            results.append(result)
            
            # Save intermediate results every 5 queries in case of crash
            if (i + 1) % 5 == 0 or i == len(queries) - 1:
                intermediate_df = pd.DataFrame(results)
                intermediate_df.to_csv(f"{output_csv}.tmp", index=False)
                print_info(f"Saved intermediate results ({i+1}/{len(queries)} queries)")
                
            # Small delay to avoid rate limits
            if i < len(queries) - 1:
                time.sleep(0.5)
            
        # Convert to DataFrame and save final results
        print_info(f"Saving final results to {output_csv}")
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            
            # Remove temporary file if it exists
            if os.path.exists(f"{output_csv}.tmp"):
                os.remove(f"{output_csv}.tmp")
                
            print_success(f"Processed {len(results)} queries, saved to {output_csv}")
        except Exception as e:
            print_error(f"Failed to save results: {str(e)}")
            # Try to recover from temp file
            if os.path.exists(f"{output_csv}.tmp"):
                print_warning(f"Attempting to recover from temporary file {output_csv}.tmp")
                os.rename(f"{output_csv}.tmp", output_csv)
            raise
            
        return results

if __name__ == "__main__":
    """Run as standalone script to process queries"""
    import argparse
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Process queries using RAG")
    parser.add_argument("queries_path", help="Path to the queries JSON file")
    parser.add_argument("--index", default="faiss_index.bin", help="Path to FAISS index")
    parser.add_argument("--chunks", default="processed_chunks.json", help="Path to processed chunks")
    parser.add_argument("--output", default="queries_output.csv", help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of contexts to retrieve per query")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create and run processor
    processor = RAGProcessor(
        args.index, 
        args.chunks
    )
    
    processor.process_queries_file(args.queries_path, args.output, args.top_k)