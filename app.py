import streamlit as st
import boto3
import json
import numpy as np
import pickle
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(
    page_title="SEC 10-K Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_bedrock():
    return boto3.client("bedrock-runtime", region_name="us-east-1")

def extract_chunks(pdf_path, chunk_size=500, overlap=50):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def find_relevant_chunks(question, chunks, chunk_embeddings, embedder, top_k=5):
    question_embedding = embedder.encode([question])
    similarities = np.dot(chunk_embeddings, question_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def ask_claude(client, context, question):
    prompt = f"""You are a financial analyst assistant. Use only the document excerpt below to answer the question.
If the answer isn't in the document, say so clearly.

DOCUMENT EXCERPT:
{context}

QUESTION:
{question}"""

    response = client.invoke_model(
        modelId="us.anthropic.claude-sonnet-4-6",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

def extract_all_metrics(client, chunks, chunk_embeddings, embedder):
    metric_definitions = [
        {"key": "company_name", "query": "Goldman Sachs company name", "format": "Full legal company name"},
        {"key": "fiscal_year", "query": "fiscal year annual report December 2025", "format": "Year as string e.g. '2025'"},
        {"key": "total_net_revenues", "query": "total net revenues 58 billion 2025", "format": "Dollar amount e.g. '$58.3 billion'"},
        {"key": "net_earnings", "query": "net earnings net income 2025 annual", "format": "Dollar amount e.g. '$17.2 billion'"},
        {"key": "earnings_per_share", "query": "diluted earnings per share EPS 2025", "format": "Dollar amount, specify diluted or basic"},
        {"key": "return_on_equity", "query": "return on equity ROE 2025 annual", "format": "Percentage e.g. '15.0%'"},
        {"key": "cet1_capital_ratio", "query": "CET1 capital ratio standardized advanced 14.3 15.1 December 2025", "format": "Return as 'Standardized: X%, Advanced: Y%'"},
        {"key": "total_assets", "query": "total assets 1.7 trillion balance sheet December 2025", "format": "Dollar amount e.g. '$1.78 trillion'"},
        {"key": "total_loans_outstanding", "query": "loans 212 billion held for investment credit portfolio", "format": "Dollar amount in billions or millions as stated"},
        {"key": "number_of_employees", "query": "employees 47 thousand human capital workforce December 2025", "format": "Number with commas e.g. '47,400'"},
    ]

    metrics = {}
    for metric in metric_definitions:
        relevant = find_relevant_chunks(metric["query"], chunks, chunk_embeddings, embedder, top_k=5)
        context = "\n\n".join(relevant)

        prompt = f"""Extract this specific metric from the document excerpt.
Metric: {metric["key"]}
Format: {metric["format"]}

Return ONLY a valid JSON object with one key: "value".
If not found, return {{"value": null}}.

DOCUMENT EXCERPT:
{context}

Return only the JSON object:"""

        response = client.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-6",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        result = json.loads(response["body"].read())
        raw = result["content"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        metrics[metric["key"]] = parsed.get("value")

    return metrics

# Load or build embeddings
@st.cache_resource
def load_chunks_and_embeddings():
    embedder = load_embedder()
    CHUNKS_FILE = "chunks.pkl"
    EMBEDDINGS_FILE = "embeddings.npy"

    if os.path.exists(CHUNKS_FILE) and os.path.exists(EMBEDDINGS_FILE):
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        chunk_embeddings = np.load(EMBEDDINGS_FILE)
    else:
        chunks = extract_chunks("GS_2025_10K.pdf")
        chunk_embeddings = embedder.encode(chunks)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)
        np.save(EMBEDDINGS_FILE, chunk_embeddings)

    return chunks, chunk_embeddings

# UI
st.title("📊 SEC 10-K Analyzer")
st.caption("Powered by Claude Sonnet 4.6 via Amazon Bedrock")

embedder = load_embedder()
client = load_bedrock()
chunks, chunk_embeddings = load_chunks_and_embeddings()

tab1, tab2 = st.tabs(["💬 Ask a Question", "📈 Key Metrics"])

# Tab 1 - Interactive Q&A
with tab1:
    st.subheader("Ask anything about the Goldman Sachs 2025 10-K")
    
    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the primary business segments?"
    )
    
    if st.button("Analyze", type="primary"):
        if question:
            with st.spinner("Analyzing..."):
                relevant_chunks = find_relevant_chunks(
                    question, chunks, chunk_embeddings, embedder
                )
                context = "\n\n".join(relevant_chunks)
                answer = ask_claude(client, context, question)
            st.markdown(answer)
        else:
            st.warning("Please enter a question.")

# Tab 2 - Metrics Dashboard
with tab2:
    st.subheader("Goldman Sachs 2025 — Key Financial Metrics")
    
    if st.button("Extract Metrics", type="primary"):
        with st.spinner("Extracting metrics from 10-K..."):
            metrics = extract_all_metrics(
                client, chunks, chunk_embeddings, embedder
            )
        
        # Display as metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Net Revenues", metrics.get("total_net_revenues") or "N/A")
            st.metric("Net Earnings", metrics.get("net_earnings") or "N/A")
            st.metric("Earnings Per Share", metrics.get("earnings_per_share") or "N/A")
        
        with col2:
            st.metric("Return on Equity", metrics.get("return_on_equity") or "N/A")
            st.metric("CET1 Capital Ratio", metrics.get("cet1_capital_ratio") or "N/A")
            st.metric("Total Assets", metrics.get("total_assets") or "N/A")
        
        with col3:
            st.metric("Total Loans Outstanding", metrics.get("total_loans_outstanding") or "N/A")
            st.metric("Employees", metrics.get("number_of_employees") or "N/A")
        
        # Download button for JSON
        st.divider()
        st.download_button(
            label="⬇️ Download as JSON",
            data=json.dumps(metrics, indent=2),
            file_name="gs_2025_metrics.json",
            mime="application/json"
        )