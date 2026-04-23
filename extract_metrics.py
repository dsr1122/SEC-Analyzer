import boto3
import json
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def find_relevant_chunks(question, chunks, chunk_embeddings, top_k=5):
    question_embedding = embedder.encode([question])
    similarities = np.dot(chunk_embeddings, question_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def extract_single_metric(client, chunks, chunk_embeddings, metric_name, query, format_instruction):
    """Dedicated extraction for a single metric with focused retrieval"""
    relevant = find_relevant_chunks(query, chunks, chunk_embeddings, top_k=8)
    context = "\n\n".join(relevant)
    
    prompt = f"""You are a financial data extraction assistant. Find ONLY the following metric in the document excerpt below.

Metric: {metric_name}
Format: {format_instruction}

Return ONLY a valid JSON object with one key: "value". 
If not found, return {{"value": null}}.
Do not include any explanation or text outside the JSON.

DOCUMENT EXCERPT:
{context}

Return only the JSON object:"""

    response = client.invoke_model(
        modelId="us.anthropic.claude-sonnet-4-6",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    )
    
    result = json.loads(response["body"].read())
    raw = result["content"][0]["text"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    return parsed.get("value")

def extract_metrics(client, chunks, chunk_embeddings):
    
    metric_definitions = [
        {
            "key": "company_name",
            "query": "Goldman Sachs company name",
            "format": "Full legal company name"
        },
        {
            "key": "fiscal_year",
            "query": "fiscal year annual report December 2025",
            "format": "Year as string e.g. '2025'"
        },
        {
            "key": "total_net_revenues",
            "query": "total net revenues 58 billion 2025",
            "format": "Dollar amount e.g. '$58.3 billion'"
        },
        {
            "key": "net_earnings",
            "query": "net earnings net income 2025 annual",
            "format": "Dollar amount e.g. '$17.2 billion'"
        },
        {
            "key": "earnings_per_share",
            "query": "diluted earnings per share EPS 2025",
            "format": "Dollar amount, specify diluted or basic"
        },
        {
            "key": "return_on_equity",
            "query": "return on equity ROE 2025 annual",
            "format": "Percentage e.g. '15.0%'"
        },
        {
            "key": "cet1_capital_ratio",
            "query": "CET1 capital ratio standardized advanced 14.3 15.1 December 2025",
            "format": "Return as 'Standardized: X%, Advanced: Y%'"
        },
        {
            "key": "total_assets",
            "query": "total assets 1.7 trillion balance sheet December 2025",
            "format": "Dollar amount e.g. '$1.78 trillion'"
        },
        {
            "key": "total_loans_outstanding",
            "query": "loans 212 billion held for investment credit portfolio",
            "format": "Dollar amount in billions or millions as stated"
        },
        {
            "key": "number_of_employees",
            "query": "employees 47 thousand human capital workforce December 2025",
            "format": "Number with commas e.g. '47,400'"
        },
    ]
    
    metrics = {}
    
    for metric in metric_definitions:
        relevant = find_relevant_chunks(
            metric["query"], chunks, chunk_embeddings, top_k=5
        )
        context = "\n\n".join(relevant)
        
        prompt = f"""You are a financial data extraction assistant.

Extract this specific metric from the document excerpt below.
Metric: {metric["key"]}
Format: {metric["format"]}

Return ONLY a valid JSON object with one key: "value".
If not found, return {{"value": null}}.
No explanation or text outside the JSON.

DOCUMENT EXCERPT:
{context}

Return only the JSON object:"""

        response = client.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-6",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 128,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        
        result = json.loads(response["body"].read())
        raw = result["content"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        metrics[metric["key"]] = parsed.get("value")
        print(f"  ✓ {metric['key']}: {metrics[metric['key']]}")
    
    return metrics

def print_metrics(metrics):
    print("\n" + "=" * 60)
    print(f"  {metrics.get('company_name', 'Company')} — {metrics.get('fiscal_year', 'FY')} Key Metrics")
    print("=" * 60)
    
    fields = [
        ("Total Net Revenues",       "total_net_revenues"),
        ("Net Earnings",             "net_earnings"),
        ("Earnings Per Share",       "earnings_per_share"),
        ("Return on Equity",         "return_on_equity"),
        ("CET1 Capital Ratio",       "cet1_capital_ratio"),
        ("Total Assets",             "total_assets"),
        ("Total Loans Outstanding",  "total_loans_outstanding"),
        ("Number of Employees",      "number_of_employees"),
    ]
    
    for label, key in fields:
        value = metrics.get(key)
        display = value if value else "Not found in document"
        print(f"  {label:<25} {display}")
    
    print("=" * 60)

# Main
client = boto3.client("bedrock-runtime", region_name="us-east-1")

CHUNKS_FILE = "chunks.pkl"
EMBEDDINGS_FILE = "embeddings.npy"

if os.path.exists(CHUNKS_FILE) and os.path.exists(EMBEDDINGS_FILE):
    print("Loading saved chunks and embeddings...")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    chunk_embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Loaded {len(chunks)} chunks\n")
else:
    print("No saved embeddings found. Please run rag_analyzer.py first.")
    exit(1)

print("Extracting financial metrics...")
metrics = extract_metrics(client, chunks, chunk_embeddings)

# Print formatted table
print_metrics(metrics)

# Also save raw JSON
output_file = "metrics_output.json"
with open(output_file, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nRaw JSON saved to {output_file}")