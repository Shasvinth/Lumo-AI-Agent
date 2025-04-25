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
    
    def __init__(self, index_path=None, chunks_path=None, model_name="gemini-1.5-flash"):
        self.genai = genai
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 2048,
            "stop_sequences": [],
        }
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
        ]
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        self.embedding_model = genai.GenerativeModel("embedding-001")
        logging.info(f"RAG Processor initialized with model: {model_name}")
        
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
            print_success("Gemini model configured successfully")
        except Exception as e:
            print_error(f"Failed to configure Gemini model: {str(e)}")
            raise
        
        # Load embedding store
        print_info("Loading embedding store")
        try:
            self.embedding_store = EmbeddingStore()
            if index_path and chunks_path:
                self.embedding_store.load(index_path=index_path, chunks_path=chunks_path)
                print_success(f"Embedding store loaded from {index_path} and {chunks_path}")
            else:
                self.embedding_store.load(index_path='faiss_index.bin', chunks_path='processed_chunks.json')
                print_success("Embedding store loaded from default paths")
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
            
    def generate_embeddings(self, content):
        """Generate embeddings for a string using Google's Gemini embedding model."""
        try:
            result = self.embedding_model.embed_content(
                content=[genai.types.Content(parts=[genai.types.Part(text=content)])], task_type="RETRIEVAL_QUERY")
            return result.embedding.values
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return None

    def construct_prompt(self, query, contexts, language="en", is_web_content=False):
        """
        Construct a prompt for the RAG model based on the query, contexts, and language.
        Enhanced to give more weight to web content when specified.
        """
        if language == "en":
            if is_web_content:
                # Enhanced prompt for web content
                system_prompt = (
                    "You are an expert assistant trained to provide comprehensive and detailed answers based on web content. "
                    "The user has searched for information online, and I need you to carefully analyze the web search results provided. "
                    "You must generate a highly detailed, factual response that thoroughly addresses the query using ONLY the information in these search results. "
                    "Include specific names, dates, events, and statistics when present in the sources. "
                    "If the information in the search results is incomplete but relevant, make note of this. "
                    "DO NOT make up or infer information not present in the search results. "
                    "If the web search results do not contain information relevant to answering the query, state this clearly. "
                    "Structure your answer in a coherent and logical manner, with paragraphs for different aspects of the answer. "
                    "Cite specific web sources when possible by referencing their URLs or titles."
                )
            else:
                # Regular prompt for textbook content
                system_prompt = (
                    "You are an expert assistant helping users understand complex topics. "
                    "For the following question, I'll provide relevant context from educational materials. "
                    "Use ONLY the information in these contexts to construct your answer. "
                    "If the information needed to answer the question is not in the contexts provided, "
                    "simply state that you don't have enough information to answer accurately. "
                    "DO NOT make up or infer information not present in the provided contexts."
                )
        elif language == "ta":
            if is_web_content:
                # Enhanced Tamil prompt for web content
                system_prompt = (
                    "நீங்கள் இணைய உள்ளடக்கத்தின் அடிப்படையில் விரிவான மற்றும் விரிவான பதில்களை வழங்க பயிற்சி பெற்ற ஒரு நிபுணர் உதவியாளர். "
                    "பயனர் ஆன்லைனில் தகவலைத் தேடியுள்ளார், மேலும் வழங்கப்பட்ட தேடல் முடிவுகளை கவனமாக பகுப்பாய்வு செய்ய வேண்டும். "
                    "இந்த தேடல் முடிவுகளில் உள்ள தகவல்களை மட்டும் பயன்படுத்தி வினவலை முழுமையாக நிவர்த்தி செய்யும் மிகவும் விரிவான, உண்மையான பதிலை நீங்கள் உருவாக்க வேண்டும். "
                    "ஆதாரங்களில் இருக்கும் போது குறிப்பிட்ட பெயர்கள், தேதிகள், நிகழ்வுகள் மற்றும் புள்ளிவிவரங்களைச் சேர்க்கவும். "
                    "தேடல் முடிவுகளில் உள்ள தகவல் முழுமையற்றதாக இருந்தாலும் தொடர்புடையதாக இருந்தால், இதைக் குறிப்பிடவும். "
                    "தேடல் முடிவுகளில் இல்லாத தகவல்களை உருவாக்கவோ அல்லது ஊகிக்கவோ வேண்டாம். "
                    "வலைத் தேடல் முடிவுகளில் வினவலுக்குப் பதிலளிப்பதற்கான தொடர்புடைய தகவல்கள் இல்லை என்றால், இதைத் தெளிவாகக் கூறுங்கள். "
                    "உங்கள் பதிலை ஒத்திசைவான மற்றும் தர்க்கரீதியான முறையில் கட்டமைக்கவும், பதிலின் வெவ்வேறு அம்சங்களுக்கான பத்திகளுடன். "
                    "சாத்தியமான போது அவற்றின் URL கள் அல்லது தலைப்புகளைக் குறிப்பிடுவதன் மூலம் குறிப்பிட்ட இணைய ஆதாரங்களை மேற்கோள் காட்டவும்."
                )
            else:
                # Regular Tamil prompt for textbook content
                system_prompt = (
                    "நீங்கள் சிக்கலான தலைப்புகளைப் புரிந்துகொள்ள பயனர்களுக்கு உதவும் நிபுணர் உதவியாளர். "
                    "பின்வரும் கேள்விக்கு, நான் கல்வி உபகரணங்களிலிருந்து தொடர்புடைய சூழலை வழங்குவேன். "
                    "உங்கள் பதிலை உருவாக்க இந்த சூழல்களில் உள்ள தகவல்களை மட்டுமே பயன்படுத்தவும். "
                    "கேள்விக்கு பதிலளிக்க தேவையான தகவல்கள் வழங்கப்பட்ட சூழல்களில் இல்லை என்றால், "
                    "துல்லியமாக பதிலளிக்க போதுமான தகவல்கள் இல்லை என்பதைக் கூறவும். "
                    "வழங்கப்பட்ட சூழல்களில் இல்லாத தகவல்களை உருவாக்கவோ அல்லது ஊகிக்கவோ வேண்டாம்."
                )
        elif language == "si":
            if is_web_content:
                # Enhanced Sinhala prompt for web content
                system_prompt = (
                    "ඔබ වෙබ් අන්තර්ගතය මත පදනම්ව සවිස්තරාත්මක හා විස්තරාත්මක පිළිතුරු සැපයීමට පුහුණු කරන ලද විශේෂඥ සහායකයෙකි. "
                    "පරිශීලකයා මාර්ගගතව තොරතුරු සොයා ඇති අතර, මම ඔබට සපයා ඇති සෙවුම් ප්‍රතිඵල ප්‍රවේශමෙන් විශ්ලේෂණය කිරීමට අවශ්‍යයි. "
                    "ඔබ මෙම සෙවුම් ප්‍රතිඵලවල ඇති තොරතුරු පමණක් භාවිතා කරමින් විමසුමට සම්පූර්ණයෙන් ආමන්ත්‍රණය කරන ඉතා විස්තරාත්මක, කරුණු සහිත ප්‍රතිචාරයක් ජනනය කළ යුතුය. "
                    "මූලාශ්‍රවල ඇති විට නිශ්චිත නම්, දින, සිදුවීම් සහ සංඛ්‍යාලේඛන ඇතුළත් කරන්න. "
                    "සෙවුම් ප්‍රතිඵලවල ඇති තොරතුරු අසම්පූර්ණ නමුත් අදාළ නම්, මෙය සටහන් කරන්න. "
                    "සෙවුම් ප්‍රතිඵලවල නොමැති තොරතුරු හදා ගැනීම හෝ අනුමාන කිරීම නොකරන්න. "
                    "වෙබ් සෙවුම් ප්‍රතිඵලවල විමසුමට පිළිතුරු දීමට අදාළ තොරතුරු අඩංගු නොවේ නම්, මෙය පැහැදිලිව ප්‍රකාශ කරන්න. "
                    "ඔබේ පිළිතුර සුසංගත හා තාර්කික ආකාරයකින්, පිළිතුරේ විවිධ අංශ සඳහා ඡේද සමඟ ව්‍යුහගත කරන්න. "
                    "හැකි විට ඔවුන්ගේ URLs හෝ මාතෘකා යොමු කිරීමෙන් නිශ්චිත වෙබ් ප්‍රභවයන් උපුටා දක්වන්න."
                )
            else:
                # Regular Sinhala prompt for textbook content
                system_prompt = (
                    "ඔබ සංකීර්ණ මාතෘකා තේරුම් ගැනීමට පරිශීලකයින්ට උදව් කරන විශේෂඥ සහායකයෙකි. "
                    "පහත ප්‍රශ්නය සඳහා, මම අධ්‍යාපනික ද්‍රව්‍ය වලින් අදාළ සන්දර්භය ලබා දෙන්නෙමි. "
                    "ඔබේ පිළිතුර තැනීමට මෙම සන්දර්භයන්හි ඇති තොරතුරු පමණක් භාවිතා කරන්න. "
                    "ප්‍රශ්නයට පිළිතුරු දීමට අවශ්‍ය තොරතුරු සපයා ඇති සන්දර්භයන්හි නොමැති නම්, "
                    "නිවැරදිව පිළිතුරු දීමට ප්‍රමාණවත් තොරතුරු ඔබට නොමැති බව පවසන්න. "
                    "සපයා ඇති සන්දර්භයන්හි නොමැති තොරතුරු හදා ගැනීම හෝ අනුමාන කිරීම නොකරන්න."
                )
        else:
            if is_web_content:
                # Default enhanced prompt for web content (English)
                system_prompt = (
                    "You are an expert assistant trained to provide comprehensive and detailed answers based on web content. "
                    "The user has searched for information online, and I need you to carefully analyze the web search results provided. "
                    "You must generate a highly detailed, factual response that thoroughly addresses the query using ONLY the information in these search results. "
                    "Include specific names, dates, events, and statistics when present in the sources. "
                    "If the information in the search results is incomplete but relevant, make note of this. "
                    "DO NOT make up or infer information not present in the search results. "
                    "If the web search results do not contain information relevant to answering the query, state this clearly. "
                    "Structure your answer in a coherent and logical manner, with paragraphs for different aspects of the answer. "
                    "Cite specific web sources when possible by referencing their URLs or titles."
                )
            else:
                # Default regular prompt for textbook content (English)
                system_prompt = (
                    "You are an expert assistant helping users understand complex topics. "
                    "For the following question, I'll provide relevant context from educational materials. "
                    "Use ONLY the information in these contexts to construct your answer. "
                    "If the information needed to answer the question is not in the contexts provided, "
                    "simply state that you don't have enough information to answer accurately. "
                    "DO NOT make up or infer information not present in the provided contexts."
                )

        # Format the contexts with titles/sources when available
        formatted_contexts = ""
        for idx, context in enumerate(contexts, 1):
            context_text = context.get("text", "")
            context_title = context.get("title", "")
            context_source = context.get("source", "")
            
            if context_title:
                formatted_contexts += f"CONTEXT {idx} (Source: {context_title}):\n{context_text}\n\n"
            elif context_source:
                formatted_contexts += f"CONTEXT {idx} (Source: {context_source}):\n{context_text}\n\n"
            else:
                formatted_contexts += f"CONTEXT {idx}:\n{context_text}\n\n"

        # Assemble the full prompt
        full_prompt = (
            f"{system_prompt}\n\n"
            f"USER QUERY: {query}\n\n"
            f"CONTEXTS:\n{formatted_contexts}\n"
            f"Based on the contexts provided, please answer the user's query. If the contexts are not relevant to the query, please state so clearly."
        )

        return full_prompt

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
                sections.add(chunk["section"])
            elif "source" in chunk:
                sections.add(chunk["source"])
            
            # Check if pages key exists before trying to access it
            if "pages" in chunk:
                for page in chunk["pages"]:
                    pages.add(page)
            elif "source_url" in chunk:
                # For web sources, use the source URL as a "page"
                pages.add(f"Source: {chunk.get('source_url', 'Web')}")
                
        return list(sections), sorted(list(pages))
    
    def process_query(self, query_id, query_text, top_k=5, language="en", source_type=None):
        """
        Process a query and return a response with contexts.
        
        Args:
            query_id (str): Unique identifier for the query
            query_text (str): The question text
            top_k (int): Number of contexts to retrieve
            language (str): Language code (en, ta, si)
            source_type (str): Type of source ('web' or None for textbooks)
            
        Returns:
            dict: Result with answer, sections, pages, etc.
        """
        try:
            # Detect language if not specified
            if not language or language == "auto":
                language = self.detect_language(query_text)
                
            print_info(f"Processing query ID: {query_id}")
            print_info(f"Query: {limit_text_for_display(query_text)}")
            print_info(f"Language: {language}, Source type: {source_type or 'textbook'}")
            
            # Generate embedding for the query
            query_embedding = self.embedding_store.get_embedding(query_text)
            
            # Search for relevant contexts
            contexts = self.embedding_store.search(query_embedding, top_k=top_k)
            print_info(f"Retrieved {len(contexts)} contexts")
            
            # Determine if this is web content
            is_web_content = source_type == "web" if source_type else False
            
            # Extract metadata (sections and pages)
            sections, pages = self.extract_metadata(contexts)
            sections_str = ", ".join(sections)
            pages_str = ", ".join(map(str, pages))
            
            # Format contexts for the prompt
            formatted_contexts = []
            for i, context in enumerate(contexts):
                chunk = context["chunk"]
                formatted_contexts.append({
                    "text": chunk["text"],
                    "title": chunk.get("section", ""),
                    "source": chunk.get("source_url", "")
                })
            
            # Construct the prompt with context awareness
            prompt = self.construct_prompt(query_text, formatted_contexts, language, is_web_content)
            
            # Generate the response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Format the context for return (debugging purposes)
            context_str = ""
            for i, c in enumerate(contexts):
                chunk_text = c['chunk']['text']
                # Include section and page information if available
                section_info = f"Section: {c['chunk'].get('section', 'Unknown')}"
                page_info = f"Pages: {', '.join(map(str, c['chunk'].get('pages', ['Unknown'])))}"
                context_str += f"Context {i+1}:\n{section_info}\n{page_info}\n\n{chunk_text}\n\n{'='*50}\n\n"
            
            # Return the result with all metadata
            result = {
                "Query_ID": query_id,
                "Answer": answer,
                "Context": context_str,
                "Sections": sections_str,
                "Pages": pages_str,
                "Language": language
            }
            
            return result
                
        except Exception as e:
            print_error(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return error result
            return {
                "Query_ID": query_id,
                "Answer": f"Error processing query: {str(e)}",
                "Context": "",
                "Sections": "",
                "Pages": "",
                "Language": language
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