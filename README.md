# SEC 10-K Analyzer

A conversational AI tool that analyzes SEC 10-K filings using Claude Sonnet 4.6 via Amazon Bedrock. Upload any public company's annual report, ask plain-English questions, extract key financial metrics, and compare two companies side by side — all grounded strictly in the source documents.

## What It Does

- Loads any 10-K PDF and processes the entire document
- Chunks the document and generates semantic embeddings for intelligent retrieval
- For each question, finds the most relevant sections using vector similarity search
- Sends only the relevant context to Claude — keeping costs low and answers accurate
- Returns answers grounded strictly in the document — Claude says so clearly when the answer isn't there
- Extracts key financial metrics as structured JSON that can feed downstream systems
- Compares two companies side by side using documents from both filings
- Generates document-aware suggested questions tailored to what's actually in the filing(s)
- Maintains chat history so follow-up questions have full conversation context

## Why This Matters

Financial services firms spend enormous time manually reviewing 10-K filings for competitive intelligence, credit analysis, risk assessment, and due diligence. This tool demonstrates how Claude can compress hours of analyst work into seconds, while staying grounded in the source document rather than hallucinating answers.

The architecture reflects how this would actually be deployed in a FS enterprise — not a generic chatbot, but a retrieval-augmented system with persistent embeddings, structured output, honest uncertainty handling, and auditability.

## Tech Stack

- **Model**: Claude Sonnet 4.6 via Amazon Bedrock (cross-region inference profile)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) for semantic chunk retrieval
- **PDF parsing**: pypdf
- **Vector similarity**: numpy dot product similarity search
- **Embedding persistence**: pickle + numpy (.npy) — compute once, reuse forever
- **UI**: Streamlit
- **AWS SDK**: boto3
- **Language**: Python 3

## Architecture

```
PDF → text extraction → chunking (500 words, 50 word overlap)
                              ↓
                    sentence-transformers embeddings
                              ↓
                    saved to disk (per-file cache)

At query time:
Question → embedding → cosine similarity → top 5 chunks
                                                  ↓
                              chunks + question → Claude Sonnet 4.6
                                                  ↓
                                         grounded answer

Comparison mode:
Question → top 5 chunks from Doc A + top 5 chunks from Doc B
                                                  ↓
                    "Here's what Company A says, here's what Company B says"
                                                  ↓
                                    structured comparison answer
```

This is a RAG (Retrieval Augmented Generation) pipeline. Embeddings are computed once per document and cached to disk — subsequent queries load instantly.

## Project Structure

```
SEC-Analyzer/
├── app.py                  # Streamlit UI — full featured application
├── rag_analyzer.py         # CLI RAG pipeline — interactive question loop
├── extract_metrics.py      # Standalone metric extraction — outputs JSON
├── analyze_10k.py          # Basic multi-question analyzer (early version)
├── claude_test.py          # Connectivity test — verifies Bedrock + Claude
├── venv/                   # Python virtual environment (gitignored)
└── README.md               # This file
```

## Setup

### Prerequisites

- Python 3
- AWS account with Amazon Bedrock access
- Claude Sonnet 4.6 enabled in Bedrock (us-east-1)
- AWS credentials configured locally (`aws configure`)

### Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install boto3 pypdf sentence-transformers numpy streamlit
```

## Usage

### Activate virtual environment first

```bash
source venv/bin/activate
```

### Run the Streamlit app

```bash
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none streamlit run app.py
```

Opens at `http://localhost:8501`

### Single document mode

Upload one 10-K PDF. The app will:
- Process and cache embeddings on first load
- Generate 6 document-aware suggested questions
- Let you ask any question via chat interface with full conversation history
- Extract key financial metrics into a dashboard
- Export metrics as JSON

### Comparison mode

Upload two 10-K PDFs. The app will:
- Process both documents independently
- Generate 6 suggested comparison questions based on both filings
- Answer questions by retrieving relevant chunks from both documents
- Structure responses to address both companies explicitly
- Let you extract metrics from either document individually

### CLI interactive analyzer

```bash
python3 rag_analyzer.py
```

### Standalone metric extraction

```bash
python3 extract_metrics.py
```

### Test Bedrock connection

```bash
python3 claude_test.py
```

## Features

### Chat History
Every question and answer is stored in session state. Follow-up questions include the full conversation as context — Claude can reference previous answers and self-correct when given new information.

### Document-Aware Suggested Questions
Claude samples chunks from across the document(s) and generates 6 tailored questions based on what it finds. In comparison mode, questions are framed as comparisons between the two companies.

### Structured Metric Extraction
Extracts 10 key financial metrics using per-metric dedicated retrieval:

| Metric | Notes |
|---|---|
| Company Name | Full legal entity name |
| Fiscal Year | |
| Total Net Revenues | |
| Net Earnings | |
| Earnings Per Share | Diluted and basic where available |
| Return on Equity | |
| CET1 Capital Ratio | Standardized and Advanced frameworks |
| Total Assets | |
| Total Loans Outstanding | |
| Number of Employees | |

> ⚠️ Metrics extracted via RAG — semantic search may not always retrieve the correct table or entity. Always verify against the source document before use in analysis or reporting.

### Embedding Cache
Each uploaded file gets its own cache keyed to the filename:
- `filename_chunks.pkl` — document chunks
- `filename_embeddings.npy` — embedding vectors

First load processes the document. Every subsequent load is instant.

## Key Design Decisions

**Per-metric dedicated retrieval** — each financial metric uses its own targeted search query. Significantly improves accuracy over a single combined query, especially for metrics buried deep in financial tables.

**Honest null handling** — when a metric isn't found, the system returns N/A rather than hallucinating. In FS demos this is a feature: it shows the system is trustworthy and auditable.

**Per-file embedding cache** — different 10-Ks get independent caches. Switching between documents doesn't require reprocessing.

**Chat history in comparison mode** — conversation context is passed to Claude on every turn, enabling follow-up questions and self-correction across both documents.

**Cross-region inference profile** — uses `us.anthropic.claude-sonnet-4-6` rather than a direct model ID, required for newer Claude models on Bedrock.

## AWS Configuration

```
Model ID: us.anthropic.claude-sonnet-4-6
Region: us-east-1
Required permission: bedrock:InvokeModel
```

## Production Considerations

This is a prototype. A production FS deployment would add:

- **pgvector or Pinecone** instead of pickle/numpy — banks already run PostgreSQL, avoiding new infrastructure dependencies
- **pdfplumber** alongside pypdf for better table extraction — some financial metrics live in structured tables that semantic search doesn't reliably surface
- **Authentication** — SSO before accessing any filing data
- **Audit logging** — required for model risk management (SR 11-7 compliance)
- **PII handling** — strip or redact sensitive data before sending to any external API
- **Rate limiting** — Bedrock has per-account token limits; production needs queuing
- **Entity disambiguation** — parent company vs subsidiary CET1 ratios require explicit entity filtering, not just semantic search

### AWS Credentials
This project uses your local AWS credentials via boto3. 
Never hardcode credentials in the code. Configure them with:
`aws configure`

## Roadmap

- [x] Basic PDF loading and Claude Q&A
- [x] Full document RAG pipeline with semantic retrieval
- [x] Persistent embeddings (compute once, reuse forever)
- [x] Multiple questions in single run
- [x] Interactive CLI
- [x] Structured JSON metric extraction
- [x] Per-metric dedicated retrieval for accuracy
- [x] Streamlit UI
- [x] JSON download of extracted metrics
- [x] Arbitrary PDF upload — any 10-K, not just Goldman
- [x] Per-file embedding cache
- [x] Chat history with full conversation context
- [x] Document-aware suggested questions
- [x] Comparative analysis — two 10-Ks side by side
- [ ] Source citations — show which chunk an answer came from
- [ ] Table-aware extraction via pdfplumber
- [ ] Export analysis as PDF/Word report
- [ ] Year-over-year comparison (same company, two years)
- [ ] Multi-document comparison dashboard (3+ companies)