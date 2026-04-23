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

def get_cache_paths(filename):
    safe_name = filename.replace(" ", "_").replace(".pdf", "")
    return f"{safe_name}_chunks.pkl", f"{safe_name}_embeddings.npy"

def load_or_build_embeddings(pdf_path, filename, embedder):
    chunks_file, embeddings_file = get_cache_paths(filename)
    if os.path.exists(chunks_file) and os.path.exists(embeddings_file):
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        chunk_embeddings = np.load(embeddings_file)
        return chunks, chunk_embeddings, False
    chunks = extract_chunks(pdf_path)
    with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
        chunk_embeddings = embedder.encode(chunks, show_progress_bar=False)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    np.save(embeddings_file, chunk_embeddings)
    return chunks, chunk_embeddings, True

def ask_claude(client, prompt, chat_history=None, system=None):
    """Core Claude call with optional chat history"""
    if system is None:
        system = (
            "You are a financial analyst assistant. Use only the document excerpts "
            "provided to answer questions. If the answer isn't in the document, say so clearly."
        )
    messages = []
    if chat_history:
        for turn in chat_history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": prompt})

    response = client.invoke_model(
        modelId="us.anthropic.claude-sonnet-4-6",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": system,
            "messages": messages
        })
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

def build_single_prompt(context, question):
    return f"DOCUMENT EXCERPT:\n{context}\n\nQUESTION:\n{question}"

def build_comparison_prompt(context_a, name_a, context_b, name_b, question):
    return f"""You are comparing two companies based on their 10-K filings.
Use only the excerpts below. Be specific about which company you are referring to at all times.
If the answer for one company isn't in the excerpts, say so clearly.

--- {name_a} EXCERPT ---
{context_a}

--- {name_b} EXCERPT ---
{context_b}

QUESTION:
{question}

Provide a structured comparison addressing both companies."""

def generate_suggested_questions(client, chunks_a, embeddings_a, name_a,
                                  chunks_b=None, embeddings_b=None, name_b=None,
                                  embedder=None):
    """Generate document-aware suggested questions for one or two documents"""
    def sample_chunks(chunks):
        total = len(chunks)
        indices = [0, total//4, total//2, 3*total//4, total-1]
        return "\n\n".join([chunks[i] for i in indices if i < total])

    context_a = sample_chunks(chunks_a)

    if chunks_b is not None:
        context_b = sample_chunks(chunks_b)
        prompt = f"""Based on these excerpts from two 10-K filings ({name_a} and {name_b}), 
generate exactly 6 insightful comparison questions a financial analyst would want to ask.
Focus on differences in performance, risk, strategy, and competitive position.
Keep each question under 15 words.

{name_a} EXCERPT:
{context_a}

{name_b} EXCERPT:
{context_b}

Return ONLY a JSON array of 6 short question strings:"""
    else:
        prompt = f"""Based on this 10-K excerpt, generate exactly 6 insightful questions 
a financial analyst would want to ask. Focus on business performance, risk, strategy, 
and competitive position. Keep each question under 15 words.

DOCUMENT EXCERPT:
{context_a}

Return ONLY a JSON array of 6 short question strings:"""

    response = client.invoke_model(
        modelId="us.anthropic.claude-sonnet-4-6",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    result = json.loads(response["body"].read())
    raw = result["content"][0]["text"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)[:6]
    except json.JSONDecodeError:
        if chunks_b is not None:
            return [
                f"How do {name_a} and {name_b} compare on capital strength?",
                f"Which company has stronger revenue growth?",
                f"How do their risk profiles differ?",
                f"Compare their strategic priorities.",
                f"Which company is better positioned competitively?",
                f"How do their workforce sizes and strategies compare?",
            ]
        return [
            "What are the primary business segments?",
            "What are the most significant risk factors?",
            "What was total revenue for the most recent fiscal year?",
            "How does the company describe its competitive position?",
            "What is the company's capital allocation strategy?",
            "What are the key strategic priorities going forward?",
        ]

def extract_all_metrics(client, chunks, chunk_embeddings, embedder):
    metric_definitions = [
        {"key": "company_name", "query": "company name legal entity", "format": "Full legal company name"},
        {"key": "fiscal_year", "query": "fiscal year annual report December", "format": "Year as string e.g. '2025'"},
        {"key": "total_net_revenues", "query": "total net revenues net revenue annual", "format": "Dollar amount e.g. '$58.3 billion'"},
        {"key": "net_earnings", "query": "net earnings net income annual", "format": "Dollar amount e.g. '$17.2 billion'"},
        {"key": "earnings_per_share", "query": "diluted earnings per share EPS", "format": "Dollar amount only e.g. '$51.32 diluted'"},
        {"key": "return_on_equity", "query": "return on equity ROE annual", "format": "Percentage e.g. '15.0%'"},
        {"key": "cet1_capital_ratio", "query": "CET1 capital ratio standardized advanced", "format": "e.g. '14.3% (Standardized)'"},
        {"key": "total_assets", "query": "total assets balance sheet", "format": "Dollar amount e.g. '$1.78 trillion'"},
        {"key": "total_loans_outstanding", "query": "total loans outstanding loan portfolio held for investment", "format": "Dollar amount in billions or millions"},
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

def run_question(question, client, embedder,
                 chunks_a, embeddings_a, name_a,
                 chunks_b=None, embeddings_b=None, name_b=None):
    """Retrieve relevant chunks and ask Claude, single or comparison mode"""
    if chunks_b is not None:
        relevant_a = find_relevant_chunks(question, chunks_a, embeddings_a, embedder)
        relevant_b = find_relevant_chunks(question, chunks_b, embeddings_b, embedder)
        context_a = "\n\n".join(relevant_a)
        context_b = "\n\n".join(relevant_b)
        prompt = build_comparison_prompt(context_a, name_a, context_b, name_b, question)
    else:
        relevant = find_relevant_chunks(question, chunks_a, embeddings_a, embedder)
        context = "\n\n".join(relevant)
        prompt = build_single_prompt(context, question)

    answer = ask_claude(client, prompt, chat_history=st.session_state.chat_history)
    st.session_state.chat_history.append({"question": question, "answer": answer})

# ── Load resources ──────────────────────────────────────────
embedder = load_embedder()
client = load_bedrock()

# ── Session state ────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "current_files" not in st.session_state:
    st.session_state.current_files = None

# ── UI ───────────────────────────────────────────────────────
st.title("📊 SEC 10-K Analyzer")
st.caption("Powered by Claude Sonnet 4.6 via Amazon Bedrock")

# File upload — two columns
st.subheader("Upload 10-K Filing(s)")
col_a, col_b = st.columns(2)

with col_a:
    file_a = st.file_uploader("Primary filing", type="pdf", key="file_a")

with col_b:
    file_b = st.file_uploader(
        "Second filing (optional — enables comparison mode)",
        type="pdf",
        key="file_b"
    )

if file_a is None:
    st.info("Upload at least one 10-K PDF to get started.")
    st.stop()

# ── Process uploaded files ───────────────────────────────────
def save_and_load(uploaded_file):
    tmp_path = f"uploaded_{uploaded_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    chunks, embeddings, was_built = load_or_build_embeddings(
        tmp_path, uploaded_file.name, embedder
    )
    return chunks, embeddings, was_built

chunks_a, embeddings_a, built_a = save_and_load(file_a)
name_a = file_a.name.replace(".pdf", "")

chunks_b, embeddings_b, name_b = None, None, None
if file_b is not None:
    chunks_b, embeddings_b, built_b = save_and_load(file_b)
    name_b = file_b.name.replace(".pdf", "")

# Status messages
if built_a:
    st.success(f"✅ Processed {len(chunks_a)} chunks from {file_a.name}")
else:
    st.success(f"✅ Loaded {file_a.name} from cache ({len(chunks_a)} chunks)")

if file_b is not None:
    if built_b:
        st.success(f"✅ Processed {len(chunks_b)} chunks from {file_b.name}")
    else:
        st.success(f"✅ Loaded {file_b.name} from cache ({len(chunks_b)} chunks)")

# ── Detect file change and reset state ──────────────────────
current_files = (file_a.name, file_b.name if file_b else None)
if st.session_state.current_files != current_files:
    st.session_state.chat_history = []
    st.session_state.suggested_questions = []
    st.session_state.current_files = current_files

# ── Generate suggested questions ────────────────────────────
if not st.session_state.suggested_questions:
    with st.spinner("Generating suggested questions..."):
        st.session_state.suggested_questions = generate_suggested_questions(
            client,
            chunks_a, embeddings_a, name_a,
            chunks_b, embeddings_b, name_b,
            embedder
        )

# Mode label
if file_b:
    st.info(f"🔀 Comparison mode: **{name_a}** vs **{name_b}**")
else:
    st.info(f"📄 Single document mode: **{name_a}**")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Ask a Question", "📈 Key Metrics"])

with tab1:
    st.subheader("Suggested Questions")
    sq_cols = st.columns(2)
    for i, q in enumerate(st.session_state.suggested_questions):
        if sq_cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
            with st.spinner("Analyzing..."):
                run_question(
                    q, client, embedder,
                    chunks_a, embeddings_a, name_a,
                    chunks_b, embeddings_b, name_b
                )

    st.divider()
    st.subheader("Ask Your Own Question")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. How do the two companies compare on capital strength?"
            if file_b else "e.g. What are the primary business segments?"
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted and question:
        with st.spinner("Analyzing..."):
            run_question(
                question, client, embedder,
                chunks_a, embeddings_a, name_a,
                chunks_b, embeddings_b, name_b
            )

    # Chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("Conversation")
        for turn in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.markdown(turn["answer"])

        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

with tab2:
    if file_b:
        st.info("Metrics dashboard shows one company at a time. Select which filing to extract from.")
        selected = st.radio(
            "Extract metrics from:",
            [name_a, name_b],
            horizontal=True
        )
        active_chunks = chunks_a if selected == name_a else chunks_b
        active_embeddings = embeddings_a if selected == name_a else embeddings_b
    else:
        active_chunks = chunks_a
        active_embeddings = embeddings_a

    if st.button("Extract Metrics", type="primary"):
        metrics = extract_all_metrics(client, active_chunks, active_embeddings, embedder)

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

        st.caption(
            "⚠️ Metrics extracted via RAG — semantic search may not always retrieve "
            "the correct table or entity. Always verify against the source document "
            "before use in analysis or reporting."
        )

        st.divider()
        active_name = name_a if (file_b is None or selected == name_a) else name_b
        st.download_button(
            label="⬇️ Download as JSON",
            data=json.dumps(metrics, indent=2),
            file_name=f"{active_name}_metrics.json",
            mime="application/json"
        )