import streamlit as st
import boto3
import json
import numpy as np
import pickle
import os
import tempfile
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
        {"key": "company_name", "query": "company name legal entity", "format": "Full legal company name"},
        {"key": "fiscal_year", "query": "fiscal year annual report December", "format": "Year as string e.g. '2025'"},
        {"key": "total_net_revenues", "query": "total net revenues net revenue annual", "format": "Dollar amount e.g. '$58.3 billion'"},
        {"key": "net_earnings", "query": "net earnings net income annual", "format": "Dollar amount e.g. '$17.2 billion'"},
        {"key": "earnings_per_share", "query": "diluted earnings per share EPS", "format": "Dollar amount, specify diluted or basic"},
        {"key": "return_on_equity", "query": "return on equity ROE annual", "format": "Percentage e.g. '15.0%'"},
        {"key": "cet1_capital_ratio", "query": "CET1 capital ratio standardized advanced", "format": "Return as 'Standardized: X%, Advanced: Y%' if both available, otherwise single value"},
        {"key": "total_assets", "query": "total assets balance sheet", "format": "Dollar amount e.g. '$1.78 trillion'"},
        {"key": "total_loans_outstanding", "query": "total loans outstanding loan portfolio held for investment", "format": "Dollar amount in billions or millions as stated"},
        {"key": "number_of_employees", "query": "number of employees headcount workforce", "format": "Number with commas e.g. '47,400'"},
    ]

    metrics = {}
    progress = st.progress(0, text="Extracting metrics...")

    for i, metric in enumerate(metric_definitions):
        progress.progress(
            int((i / len(metric_definitions)) * 100),
            text=f"Extracting {metric['key'].replace('_', ' ')}..."
        )

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

    progress.progress(100, text="Done!")
    return metrics

def get_cache_paths(filename):
    """Generate cache file paths keyed to the uploaded filename"""
    safe_name = filename.replace(" ", "_").replace(".pdf", "")
    return f"{safe_name}_chunks.pkl", f"{safe_name}_embeddings.npy"

def load_or_build_embeddings(pdf_path, filename, embedder):
    """Load cached embeddings or build from scratch"""
    chunks_file, embeddings_file = get_cache_paths(filename)

    if os.path.exists(chunks_file) and os.path.exists(embeddings_file):
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        chunk_embeddings = np.load(embeddings_file)
        return chunks, chunk_embeddings, False  # False = loaded from cache

    # Build from scratch
    chunks = extract_chunks(pdf_path)

    with st.spinner(f"Generating embeddings for {len(chunks)} chunks — this takes about 30 seconds..."):
        chunk_embeddings = embedder.encode(chunks, show_progress_bar=False)

    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    np.save(embeddings_file, chunk_embeddings)

    return chunks, chunk_embeddings, True  # True = freshly built

# Load shared resources
embedder = load_embedder()
client = load_bedrock()

# UI
st.title("📊 SEC 10-K Analyzer")
st.caption("Powered by Claude Sonnet 4.6 via Amazon Bedrock")

# File upload
st.subheader("Upload a 10-K Filing")
uploaded_file = st.file_uploader(
    "Upload a 10-K PDF",
    type="pdf",
    help="Upload any SEC 10-K filing as a PDF"
)

if uploaded_file is None:
    st.info("Upload a 10-K PDF above to get started.")
    st.stop()

# Save uploaded file to disk temporarily
tmp_path = f"uploaded_{uploaded_file.name}"
with open(tmp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# Load or build embeddings
chunks, chunk_embeddings, was_built = load_or_build_embeddings(
    tmp_path, uploaded_file.name, embedder
)

if was_built:
    st.success(f"✅ Processed {len(chunks)} chunks from {uploaded_file.name}")
else:
    st.success(f"✅ Loaded {uploaded_file.name} from cache ({len(chunks)} chunks)")

st.divider()

# Tabs
tab1, tab2 = st.tabs(["💬 Ask a Question", "📈 Key Metrics"])

# Tab 1 - Q&A
with tab1:
    st.subheader(f"Ask anything about this 10-K")

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

# Tab 2 - Metrics
with tab2:
    st.subheader("Key Financial Metrics")

    if st.button("Extract Metrics", type="primary"):
        metrics = extract_all_metrics(client, chunks, chunk_embeddings, embedder)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Company", metrics.get("company_name") or "N/A")
            st.metric("Fiscal Year", metrics.get("fiscal_year") or "N/A")
            st.metric("Total Net Revenues", metrics.get("total_net_revenues") or "N/A")
            st.metric("Net Earnings", metrics.get("net_earnings") or "N/A")

        with col2:
            st.metric("Earnings Per Share", metrics.get("earnings_per_share") or "N/A")
            st.metric("Return on Equity", metrics.get("return_on_equity") or "N/A")
            st.metric("CET1 Capital Ratio", metrics.get("cet1_capital_ratio") or "N/A")

        with col3:
            st.metric("Total Assets", metrics.get("total_assets") or "N/A")
            st.metric("Total Loans Outstanding", metrics.get("total_loans_outstanding") or "N/A")
            st.metric("Employees", metrics.get("number_of_employees") or "N/A")

        st.divider()
        st.download_button(
            label="⬇️ Download as JSON",
            data=json.dumps(metrics, indent=2),
            file_name=f"{uploaded_file.name.replace('.pdf', '')}_metrics.json",
            mime="application/json"
        )