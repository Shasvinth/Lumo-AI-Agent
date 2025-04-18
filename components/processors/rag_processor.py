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
            
    def construct_prompt(self, query, contexts, max_tokens=8000, target_lang='en', use_markdown=False):
        """
        Construct RAG prompt with query and contexts
        
        Args:
            query (str): User query
            contexts (list): List of context chunks
            max_tokens (int): Maximum tokens in the prompt
            target_lang (str): Target language for the response
            use_markdown (bool): Whether to format the output with Markdown
            
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
        
        # Markdown formatting instructions
        markdown_instructions = ""
        if use_markdown:
            markdown_instructions = """
Format your answer using Markdown syntax:
- Use # for main headings and ## for subheadings
- Use **bold** for emphasis on important terms
- Use bullet points (- ) for lists of items
- Use numbered lists (1. 2. 3.) for sequential information
- Use > for quotes or important definitions
- Use code blocks for any mathematical formulas or special notation
"""
        
        # Construct prompt template based on target language
        if target_lang == 'en':
            prompt = f"""Using the following textbook content, answer the question below. 
Your answer should be comprehensive but focused on the question.
Provide specific historical details, events, and explanations based on the textbook content.
If the answer cannot be found in the provided context, say so clearly.
{markdown_instructions}

Context:
{combined_context}

Question:
{query}

Answer:"""
        elif target_lang == 'ta':
            prompt = f"""பின்வரும் பாடப்புத்தக உள்ளடக்கத்தைப் பயன்படுத்தி, கீழே உள்ள கேள்விக்கு பதிலளிக்கவும்.
உங்கள் பதில் விரிவானதாக இருக்க வேண்டும், ஆனால் கேள்வியில் கவனம் செலுத்த வேண்டும்.
பாடப்புத்தக உள்ளடக்கத்தின் அடிப்படையில் குறிப்பிட்ட வரலாற்று விவரங்கள், நிகழ்வுகள் மற்றும் விளக்கங்களை வழங்கவும்.
பதிலை வழங்கப்பட்ட சூழலில் கண்டறிய முடியவில்லை என்றால், அதை தெளிவாக கூறவும்.
{markdown_instructions}

சூழல்:
{combined_context}

கேள்வி:
{query}

பதில்:"""
        else:  # Sinhala
            prompt = f"""පහත පෙළ පොතේ අන්තර්ගතය භාවිතා කරමින්, පහත ප්‍රශ්නයට පිළිතුරු සපයන්න.
ඔබේ පිළිතුර සවිස්තරාත්මක විය යුතුය, නමුත් ප්‍රශ්නය කෙරෙහි අවධානය යොමු කළ යුතුය.
පෙළ පොතේ අන්තර්ගතය මත පදනම්ව නිශ්චිත ඉතිහාසික විස්තර, සිදුවීම් සහ පැහැදිලි කිරීම් සපයන්න.
පිළිතුර සපයන ලද සන්දර්භයේ සොයාගත නොහැකි නම්, එය පැහැදිලිව පවසන්න.
{markdown_instructions}

සන්දර්භය:
{combined_context}

ප්‍රශ්නය:
{query}

පිළිතුර:"""
        
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
    
    def process_query(self, query_id, query_text, top_k=5, use_markdown=False):
        """
        Process a single query
        
        Args:
            query_id (str): Query ID
            query_text (str): Query text
            top_k (int): Number of contexts to retrieve
            use_markdown (bool): Whether to format the output with Markdown
        
        Returns:
            dict: Dict with query results
        """
        print_info(f"Processing query {query_id}: {limit_text_for_display(query_text)}")
        
        try:
            # Detect query language
            query_lang = self.detect_language(query_text)
            print_info(f"Detected query language: {query_lang}")
            
            # Translate query to English for embedding search
            if query_lang != 'en':
                query_text_en = self.translate_text(query_text, 'en')
                print_info("Translated query to English for search")
            else:
                query_text_en = query_text
            
            # Retrieve relevant context
            contexts = self.embedding_store.search(query_text_en, top_k=top_k)
            
            # Extract metadata
            sections, pages = self.extract_metadata(contexts)
            sections_str, pages_str = format_metadata(sections, pages)
            
            # Build prompt in detected language
            prompt = self.construct_prompt(query_text, contexts, target_lang=query_lang, use_markdown=use_markdown)
            
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
                        # Return a fallback response in the detected language
                        if query_lang == 'en':
                            answer = "Sorry, I couldn't generate an answer due to an error."
                        elif query_lang == 'ta':
                            answer = "மன்னிக்கவும், பிழை காரணமாக பதிலை உருவாக்க முடியவில்லை."
                        else:  # Sinhala
                            answer = "සමාවෙන්න, දෝෂයක් හේතුවෙන් පිළිතුරක් ජනනය කිරීමට නොහැකි විය."
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
                "Pages": pages_str,
                "Language": query_lang
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
                "Pages": "",
                "Language": ""
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