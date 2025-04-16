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

from embedding_store import EmbeddingStore
from utils import (
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
    
    def __init__(self, index_path='faiss_index.bin', chunks_path='processed_chunks.json'):
        """
        Initialize RAG processor
        
        Args:
            index_path (str): Path to FAISS index
            chunks_path (str): Path to processed chunks JSON
        """
        # Validate files
        validate_file_exists(index_path)
        validate_file_exists(chunks_path)
        
        # Load API key
        print_info("Loading Gemini API key")
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print_error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Set up Gemini LLM with safety settings and rate limiting
        print_info("Configuring Gemini model")
        try:
            genai.configure(api_key=api_key)
            
            # Configure model with safety settings
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more factual responses
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            self.gemini_model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            print_success("Gemini model configured successfully")
        except Exception as e:
            print_error(f"Failed to configure Gemini model: {str(e)}")
            raise
        
        # Load embedding store
        print_info("Loading embedding store")
        try:
            self.embedding_store = EmbeddingStore()
            self.embedding_store.load(index_path, chunks_path)
            print_success("Embedding store loaded successfully")
        except Exception as e:
            print_error(f"Failed to load embedding store: {str(e)}")
            raise
        
    def construct_prompt(self, query, contexts, max_tokens=8000):
        """
        Construct RAG prompt with query and contexts
        
        Args:
            query (str): User query
            contexts (list): List of context chunks
            max_tokens (int): Maximum tokens in the prompt
            
        Returns:
            str: Formatted prompt for the LLM
        """
        print_info(f"Constructing prompt for query: {limit_text_for_display(query)}")
        
        # Extract context texts with metadata tracking
        context_texts = []
        total_length = 0
        
        # Add contexts until we approach max tokens
        for context in contexts:
            context_text = context["chunk"]["text"]
            # Rough estimate: 1 token ~= 4 characters
            est_tokens = len(context_text) / 4
            
            if total_length + est_tokens > max_tokens:
                print_warning(f"Context limit reached ({total_length} est. tokens)")
                break
                
            context_texts.append(context_text)
            total_length += est_tokens
            
        combined_context = "\n\n---\n\n".join(context_texts)
        
        # Construct prompt template
        prompt = f"""Using the following textbook content, answer the question below. 
Your answer should be comprehensive but focused on the question.
Provide specific historical details, events, and explanations based on the textbook content.
If the answer cannot be found in the provided context, say so clearly.

Context:
{combined_context}

Question:
{query}

Answer:"""
        
        print_info(f"Prompt constructed with {len(context_texts)} context chunks (~{total_length} tokens)")
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
            sections.add(chunk["section"])
            for page in chunk["pages"]:
                pages.add(page)
                
        return list(sections), sorted(list(pages))
    
    def process_query(self, query_id, query_text, top_k=5):
        """
        Process a single query
        
        Args:
            query_id (str): Query ID
            query_text (str): Query text
            top_k (int): Number of contexts to retrieve
        
        Returns:
            dict: Dict with query results
        """
        print_info(f"Processing query {query_id}: {limit_text_for_display(query_text)}")
        
        try:
            # Retrieve relevant context
            contexts = self.embedding_store.search(query_text, top_k=top_k)
            
            # Extract metadata
            sections, pages = self.extract_metadata(contexts)
            sections_str, pages_str = format_metadata(sections, pages)
            
            # Build prompt
            prompt = self.construct_prompt(query_text, contexts)
            
            # Generate answer with retry mechanism
            max_retries = 3
            answer = None
            
            for retry in range(max_retries):
                try:
                    # Add delay for rate limiting
                    if retry > 0:
                        time.sleep(2 * retry)  # Exponential backoff
                        print_warning(f"Retrying request (attempt {retry + 1}/{max_retries})")
                        
                    # Generate content
                    response = self.gemini_model.generate_content(prompt)
                    answer = response.text
                    break  # Success - exit retry loop
                except Exception as e:
                    if retry == max_retries - 1:
                        print_error(f"Failed to generate answer after {max_retries} attempts: {str(e)}")
                        # Return a fallback response
                        answer = "Sorry, I couldn't generate an answer due to an error."
                    else:
                        print_warning(f"Generation error (will retry): {str(e)}")
                        # Continue to next retry
            
            # Combine context texts
            combined_context = "\n\n".join([context["chunk"]["text"] for context in contexts])
            
            result = {
                "ID": query_id,
                "Context": combined_context,
                "Answer": answer,
                "Sections": sections_str,
                "Pages": pages_str
            }
            
            print_success(f"Query {query_id} processed successfully")
            
            # Free memory
            gc.collect()
            
            return result
            
        except Exception as e:
            print_error(f"Error processing query {query_id}: {str(e)}")
            # Return a minimal result with error information
            return {
                "ID": query_id,
                "Context": "",
                "Answer": f"Error processing query: {str(e)}",
                "Sections": "",
                "Pages": ""
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