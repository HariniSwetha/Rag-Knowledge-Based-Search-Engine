# KNOWLEDGE BASED SEARCH ENGINE

A RAG (Retrieval-Augmented Generation) powered search engine for querying documents using Claude AI.

## FEATURES

- **Document Ingestion**: Upload PDF and TXT files
- **Smart Search**: Keyword-based retrieval with relevance ranking
- **AI-Powered Answers**: Claude generates contextual answers
- **Web Interface**: Clean, modern frontend
- **Document Management**: View, upload, and delete documents
- **RESTful API**: Easy integration with other systems

## ARCHITECTURE

- **Backend**: Flask + Python
- **LLM**: Anthropic Claude API
- **Frontend**: HTML/CSS/JavaScript (Vanilla)
- **Storage**: JSON-based knowledge base

## SETUP

### Prerequisites

- Python 3.8+
- Anthropic API key

### INSTALLATION

1. Clone the repository
2. Create virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
   ```

5. Run the server:

   ```bash
   python app.py
   ```

6. Open browser to `http://localhost:5000`

## API ENDPOINTS

- `POST /api/ingest` - Upload documents
- `POST /api/query` - Query knowledge base
- `GET /api/documents` - List documents
- `DELETE /api/documents/<doc_id>` - Delete document
- `GET /api/search` - Simple search

## USAGE

1. Upload documents via the web interface
2. Ask questions in natural language
3. Claude generates answers from relevant document chunks
4. View sources for each answer

## FILE STRUCTURE

```
├── app.py                 # Flask backend
├── rag_engine.py         # RAG implementation
├── index.html            # Frontend
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── README.md             # Documentation
```
