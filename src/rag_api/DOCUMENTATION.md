# RAG API Documentation

## Overview

The RAG API (`src/rag_api`) is a FastAPI-based service that powers the Chatbot-MiNI application. It implements Retrieval-Augmented Generation (RAG) to provide intelligent responses to user queries about the Faculty of Mathematics and Information Science (MiNI) at Warsaw University of Technology. The API integrates vector database retrieval, large language model generation, and multilingual support.

## Architecture

The API consists of the following main components:

- **API Layer** (`api.py`): FastAPI application with endpoints for chat interactions and feedback collection
- **Core Logic** (`main.py`): LLM querying functionality and CLI interface
- **Data Models** (`models.py`): Pydantic models for request/response structures
- **Modules**:
  - `retrieval.py`: Hybrid vector search using Qdrant with dense and sparse embeddings
  - `prompt_builder.py`: Construction of LLM prompts with context and conversation history
  - `translator.py`: Multilingual text translation using LLM
  - `logs.py`: Centralized logging configuration

## Key Features

- **Retrieval-Augmented Generation**: Combines vector database search with LLM generation for accurate, context-aware responses
- **Multilingual Support**: Automatic translation between Polish, English, and Ukrainian
- **Conversation History**: Maintains context across multiple interactions
- **Feedback Collection**: Captures user ratings and selections for model improvement
- **Hybrid Search**: Uses both dense (semantic) and sparse (keyword) embeddings for optimal retrieval
- **A/B Testing Support**: Configurable model variants and parameters

## API Endpoints

### POST /chat

Handles user queries and returns AI-generated responses.

**Request Body:**
```json
{
  "query": "string",
  "language": "pl",
  "conversation_history": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ],
  "mode": "string (optional)",
  "variant": "string (optional)",
  "modelConfig": {
    "model": "string",
    "temperature": 0.0,
    "max_tokens": 500,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": ["url1", "url2"],
  "conversation_history": [...]
}
```

### POST /feedback

Collects user feedback on responses for analytics and model improvement.

**Request Body:**
```json
{
  "message_id": 123,
  "rating": 5,
  "selected": true,
  "query": "string",
  "response_text": "string",
  "variant_config": {...}
}
```

## Data Flow

1. **Query Processing**: User query is translated to Polish if needed
2. **Retrieval**: Hybrid search retrieves top-k relevant text chunks from Qdrant vector database
3. **Prompt Building**: Context chunks, conversation history, and static FAQ are combined into LLM prompt
4. **Generation**: OpenRouter API queries configured LLM model
5. **Translation**: Response is translated back to user's language if necessary
6. **Response**: Formatted answer with sources and updated conversation history returned

## Configuration

The API uses the following environment variables:

- `OPENROUTER_API_KEY`: Required for LLM API access
- `QDRANT_DIR`: Path to Qdrant vector database (defaults to data directory)
- `LOG_LEVEL`: Logging verbosity (defaults to INFO)

## Dependencies

- **FastAPI**: Web framework
- **Qdrant**: Vector database client
- **OpenAI**: LLM API client (configured for OpenRouter)
- **FastEmbed**: Embedding generation
- **Pydantic**: Data validation
- **python-dotenv**: Environment variable loading

## Testing

Unit tests are located in the `tests/` directory:
- `api_test.py`: API endpoint tests
- `main_test.py`: Core functionality tests

## CLI Usage

The `main.py` script provides a command-line interface for testing:

```bash
python -m src.rag_api.main
```

Commands:
- Enter query: Get response
- `clear`: Reset conversation history
- `q`: Quit

## Error Handling

- Empty queries return 400 Bad Request
- Retrieval failures return apologetic message in appropriate language
- LLM API errors return generic error message
- Translation failures fall back to original text

## Security

- CORS middleware allows all origins (configured for development)
- Input validation via Pydantic models
- Thread-safe feedback writing with file locks</content>
<parameter name="filePath">/home/tymon/coding/Chatbot-MiNI/src/rag_api/DOCUMENTATION.md
