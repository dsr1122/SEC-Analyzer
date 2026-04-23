# SEC 10-K Analyzer

A conversational AI tool that analyzes SEC 10-K filings using Claude Sonnet 4.6 via Amazon Bedrock. Ask plain-English questions about any public company's annual report and get grounded, document-based answers — or extract key financial metrics automatically into a structured dashboard.

## What It Does

- Loads a 10-K PDF and processes the entire document (not just the first few pages)
- Chunks the document and generates semantic embeddings for intelligent retrieval
- For each question, finds the most relevant sections using vector similarity search
- Sends only the relevant context to Claude — keeping costs low and answers accurate
- Returns answers grounded strictly in the document — Claude says so clearly when the answer isn't there
- Extracts key financial metrics as structured JSON that can feed downstream systems

## Why This Matters

Financial services firms spend enormous time manually reviewing 10-K filings for competitive intelligence, credit analysis, risk assessment, and due diligence. This tool demonstrates how Claude can compress hours of analyst work into seconds, while staying grounded in the source document rather than hallucinating answers.

The architecture reflects how this would actually be deployed in a FS enterprise — not a generic chatbot, but a retrieval-augmented system with persistent embeddings, structured output, and honest uncertainty handling.

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
                    saved to disk (chunks.pkl + embeddings.npy)

At query time:
Question → embedding → cosine similarity → top 5 chunks
                                                  ↓
                              chunks + question → Claude Sonnet 4.6
                                                  ↓
                                         grounded answer
```

This is a RAG (Retrieval Augmented Generation) pipeline. Embeddings are computed once per document and cached — subsequent queries load instantly from disk.

## Project Structure

```
SEC-Analyzer/
├── app.py                  # Streamlit UI — interactive Q&A + metrics dashboard
├── rag_analyzer.py         # CLI RAG pipeline — interactive question loop
├── extract_metrics.py      # Structured metric extraction — outputs JSON
├── analyze_10k.py          # Basic multi-question analyzer (early version)
├── claude_test.py          # Connectivity test — verifies Bedrock + Claude
├── chunks.pkl              # Cached document chunks (gitignored)
├── embeddings.npy          # Cached embeddings (gitignored)
├── GS_2025_10K.pdf         # Goldman Sachs 2025 10-K (gitignored)
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

### Add a 10-K PDF

Download any public company 10-K as a PDF and place it in the project folder. Goldman Sachs publishes theirs at:

```
https://www.goldmansachs.com/investor-relations/financials/annual-reports/2023-annual-report/assets/pdfs/2023-form-10-k.pdf
```

Update the filename reference in `rag_analyzer.py` and `extract_metrics.py` if using a different file.

## Usage

### Activate virtual environment first

```bash
source venv/bin/activate
```

### Run the Streamlit app (recommended)

```bash
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none streamlit run app.py
```

Opens at `http://localhost:8501` with two tabs:
- **Ask a Question** — type any question about the 10-K, get a grounded answer
- **Key Metrics** — extract financial metrics into a dashboard with JSON download

### Run the CLI interactive analyzer

```bash
python3 rag_analyzer.py
```

Type questions at the prompt. First run processes and caches the document. Subsequent runs load instantly.

### Run structured metric extraction only

```bash
python3 extract_metrics.py
```

Extracts 8 key financial metrics and prints a formatted table. Also saves `metrics_output.json`.

### Test your Bedrock connection

```bash
python3 claude_test.py
```

## Metrics Extracted

| Metric | Notes |
|---|---|
| Total Net Revenues | |
| Net Earnings | |
| Earnings Per Share | Diluted and basic |
| Return on Equity | |
| CET1 Capital Ratio | Standardized and Advanced frameworks |
| Total Assets | |
| Total Loans Outstanding | |
| Number of Employees | |

## Key Design Decisions

**Per-metric dedicated retrieval** — each financial metric uses its own targeted search query rather than a single combined query. This improves accuracy significantly, especially for metrics buried deep in the document.

**Honest null handling** — when a metric isn't found in the retrieved chunks, the system returns null rather than hallucinating a number. In demos this is a feature: it shows the system is trustworthy.

**Embedding persistence** — embeddings are computed once and saved to disk. A 200+ page 10-K produces ~857 chunks. Recomputing on every run would add 30-60 seconds of latency. Saved embeddings load in under a second.

**Cross-region inference profile** — uses `us.anthropic.claude-sonnet-4-6` (Bedrock cross-region inference) rather than a direct model ID, which is required for newer Claude models.

## AWS Configuration

```
Model ID: us.anthropic.claude-sonnet-4-6
Region: us-east-1
Required permission: bedrock:InvokeModel
```

## Production Considerations

This is a prototype. A production FS deployment would add:

- **pgvector or Pinecone** instead of pickle/numpy for vector storage — banks already run PostgreSQL, avoiding new infrastructure dependencies
- **pdfplumber** alongside pypdf for better table extraction — some financial metrics live in structured tables that semantic search doesn't reliably surface
- **Authentication** — API key or SSO before accessing any filing data
- **Audit logging** — required for model risk management (SR 11-7 compliance)
- **PII handling** — strip or redact sensitive data before sending to any external API
- **Rate limiting** — Bedrock has per-account token limits; a production system needs queuing

## Roadmap

- [x] Basic PDF loading and Claude Q&A
- [x] Full document RAG pipeline with semantic retrieval
- [x] Persistent embeddings (compute once, reuse forever)
- [x] Multiple questions in single run
- [x] Interactive CLI
- [x] Structured JSON metric extraction
- [x] Per-metric dedicated retrieval for accuracy
- [x] Streamlit UI with Q&A and metrics dashboard
- [x] JSON download of extracted metrics
- [ ] Arbitrary PDF upload via UI (any 10-K, not just Goldman)
- [ ] Multi-document comparison (GS vs JPMorgan side by side)
- [ ] Table-aware extraction for metrics buried in financial tables