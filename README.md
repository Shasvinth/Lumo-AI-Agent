# RAG Chatbot System with Google Gemini

This repository contains a complete implementation of a Retrieval-Augmented Generation (RAG) system for question answering using Google's Gemini API. The system processes PDF documents, creates embeddings, and generates accurate answers to user queries based on the document content.

## System Overview

The RAG system consists of three main components:

1. **PDF Processing**: Extracts text from PDF documents and splits it into manageable chunks.
2. **Vector Store Building**: Creates embeddings for text chunks and builds a vector database for similarity search.
3. **Query Processing**: Retrieves relevant context for a query, generates answers using Gemini, and provides source information.

## Architecture

![RAG Architecture](https://i.imgur.com/ePtLqEY.png)

The system follows these steps:
1. Extract text from the PDF and split into chunks with metadata
2. Generate embeddings for each chunk using Google's Gemini API
3. Store embeddings in a FAISS vector index for efficient retrieval
4. When a query is received, find the most relevant chunks
5. Construct a prompt with the query and relevant context
6. Generate an answer using Gemini LLM
7. Return the answer with source information (sections and page numbers)

## Files Structure

- `main.py`: Main entry point that orchestrates the complete RAG pipeline
- `pdf_processor.py`: Handles PDF text extraction and chunking
- `embedding_store.py`: Manages text embeddings and vector database using Gemini API
- `rag_processor.py`: Processes queries, retrieves context, and generates answers
- `utils.py`: Contains utility functions for progress tracking and error handling
- `app.py`: Web application for interacting with the RAG system
- `run.py`: Command-line interface for the system

## Setup and Usage

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### Usage

#### Complete Pipeline

Run the complete RAG pipeline (PDF processing, vector store building, and query processing):

```
python main.py --pdf your_textbook.pdf --queries queries.json --output-csv results.csv
```

#### Individual Steps

1. Process a PDF document:
   ```
   python pdf_processor.py your_textbook.pdf
   ```

2. Build the vector store:
   ```
   python embedding_store.py chunks.json
   ```

3. Process queries:
   ```
   python rag_processor.py queries.json
   ```

#### Web Interface

Run the web application:

```
python app.py
```

Then open your browser at `http://localhost:5000`

## Technical Details

### PDF Processing

The PDF processing module extracts text from PDF files and splits it into overlapping chunks. It maintains metadata for each chunk, including:
- Page number
- Section title
- Chunk text

This information is used to provide source attribution in the generated answers.

### Embedding Generation

The system uses Google's Gemini embedding model (`models/embedding-001`) to create vector representations of text chunks. Features include:
- Batch processing to optimize API calls
- Retry mechanism with exponential backoff for reliability
- Error handling with fallback options

### Vector Search

FAISS (Facebook AI Similarity Search) is used for efficient vector similarity search:
- Uses L2 distance for similarity calculation
- Exact search with IndexFlatL2 for accuracy
- ID mapping to track which embedding corresponds to which chunk

### RAG Implementation

The RAG processor:
1. Retrieves the top-k most similar chunks for a query
2. Extracts metadata (sections and page numbers)
3. Constructs a prompt that includes the query and context chunks
4. Generates an answer using Gemini
5. Returns the answer with source information

## Advanced Features

- **Memory Management**: Garbage collection and memory-efficient mode for processing large documents
- **Rate Limiting**: Implements delays and backoff strategies to comply with API limits
- **Error Recovery**: Retry mechanisms for API failures and intermediate result saving
- **Progress Tracking**: Visual progress indicators for long-running operations

## Customization

### Embedding Parameters

You can customize the embedding parameters in `embedding_store.py`:
- Change the model in `genai.embed_content` (default: "models/embedding-001")
- Adjust the task_type parameter (default: "retrieval_query")
- Modify the batch size for embedding generation (default: 10)

### RAG Parameters

Customize the RAG behavior in `rag_processor.py`:
- Adjust the number of context chunks retrieved (top_k parameter)
- Modify the prompt template for different response styles
- Change the Gemini model parameters for answer generation

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure your Gemini API key is correct and has the necessary permissions.
2. **Embedding Failures**: If embeddings fail to generate, check the API response using the logging information.
3. **Out of Memory**: For large documents, use the `--memory-efficient` flag to reduce memory usage.

### Debug Tools

- Run `python test_embedding_api.py` to test Gemini API connectivity
- Check the logs for detailed error information
- Set environment variables to control logging levels

## License

[License information here]

## Acknowledgments

- Google Generative AI for the Gemini models
- Facebook Research for FAISS

## References

- [Google Generative AI Documentation](https://ai.google.dev/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [RAG Paper](https://arxiv.org/abs/2005.11401) - Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" 