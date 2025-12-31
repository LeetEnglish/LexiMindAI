# LexiAI - AI Microservice

Python FastAPI microservice for AI operations using Hugging Face models.

## Features
- Document parsing and content extraction
- Writing evaluation (grammar, vocabulary, coherence)
- Speaking evaluation (pronunciation, fluency)
- Chat completion for tutoring

## Setup
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Docker
```bash
docker-compose up -d
```

## API Endpoints
- `POST /api/v1/document/parse` - Parse document
- `POST /api/v1/scoring/writing` - Score writing
- `POST /api/v1/scoring/speaking` - Score speaking
- `POST /api/v1/chat/complete` - Chat completion
