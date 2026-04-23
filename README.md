# SEC 10-K Analyzer

A conversational AI tool that analyzes SEC 10-K filings using Claude Sonnet 4.6 via Amazon Bedrock. Ask plain-English questions about any public company's annual report and get grounded, document-based answers.

## What It Does

- Loads a 10-K PDF from your local filesystem
- Extracts and processes the document text
- Sends relevant content to Claude Sonnet 4.6 along with your question
- Returns answers grounded strictly in the document — Claude will tell you if the answer isn't there

## Why This Matters

Financial services firms spend enormous time manually reviewing 10-K filings for competitive intelligence, credit analysis, risk assessment, and due diligence. This tool demonstrates how Claude can compress hours of analyst work into seconds, while staying grounded in the source document rather than hallucinating answers.

## Tech Stack

- **Model**: Claude Sonnet 4.6 via Amazon Bedrock (cross-region inference profile)
- **PDF parsing**: pypdf
- **AWS SDK**: boto3
- **Language**: Python 3

## Project Structure

```
SEC-Analyzer/
├── claude_test.py      # Simple connectivity test — verifies Bedrock + Claude are working
├── analyze_10k.py      # Core analyzer — loads a PDF and answers a question about it
├── venv/               # Python virtual environment
└── README.md           # This file
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
pip3 install boto3 pypdf
```

### Add a 10-K PDF

Download any public company 10-K as a PDF and place it in the project folder. For example, Goldman Sachs publishes theirs at:

```
https://www.goldmansachs.com/investor-relations/financials/annual-reports/2023-annual-report/assets/pdfs/2023-form-10-k.pdf
```

Rename it or update the filename reference in `analyze_10k.py`.

## Usage

### Test your connection first

```bash
python3 claude_test.py
```

You should get a one-sentence description of a 10-K filing back from Claude.

### Run the analyzer

```bash
python3 analyze_10k.py
```

This will extract text from the first 10 pages of the PDF and ask Claude:

> *"What are the primary business segments of this company?"*

## Example Output

```
Extracted 42142 characters from the 10-K

Question: What are the primary business segments of this company?

Answer:
Based on the document, Goldman Sachs manages and reports its activities 
in three business segments:

1. Global Banking & Markets
2. Asset & Wealth Management  
3. Platform Solutions
```

## AWS Configuration

This project uses the Amazon Bedrock cross-region inference profile for Claude Sonnet 4.6:

```
us.anthropic.claude-sonnet-4-6
```

Make sure your AWS credentials have the following permission:

```
bedrock:InvokeModel
```

## Roadmap

- [ ] Multiple questions in a single run
- [ ] Full document chunking (beyond first 10 pages)
- [ ] Interactive CLI — type any question at the prompt
- [ ] Structured JSON output for financial metrics extraction
- [ ] Streamlit UI for demos