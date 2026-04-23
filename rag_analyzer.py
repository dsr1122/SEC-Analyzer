import boto3
import json
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_chunks(pdf_path, chunk_size=500, overlap=50):
    """Extract full document text and split into overlapping chunks"""
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

def find_relevant_chunks(question, chunks, chunk_embeddings, top_k=5):
    """Find the most semantically relevant chunks for a question"""
    question_embedding = embedder.encode([question])
    
    # Calculate similarity between question and all chunks
    similarities = np.dot(chunk_embeddings, question_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [chunks[i] for i in top_indices]

def ask_claude(client, context, question):
    """Send relevant context and question to Claude"""
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

# Questions to ask
QUESTIONS = [
    "What are the primary business segments of this company?",
    "What are the most significant risks the company has identified?",
    "What was the company's total revenue for the most recent fiscal year?",
    "How does the company describe its competitive position in the market?",
]

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
    print("Extracting and chunking document...")
    chunks = extract_chunks("GS_2025_10K.pdf")
    print(f"Created {len(chunks)} chunks\n")
    
    print("Generating embeddings...")
    chunk_embeddings = embedder.encode(chunks)
    
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    np.save(EMBEDDINGS_FILE, chunk_embeddings)
    print("Saved chunks and embeddings to disk\n")

# Interactive CLI
print("=" * 60)
print("Goldman Sachs 10-K Analyzer")
print("Ask any question about the filing. Type 'quit' to exit.")
print("=" * 60)

while True:
    print()
    question = input("Your question: ").strip()
    
    if not question:
        continue
    
    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    
    relevant_chunks = find_relevant_chunks(question, chunks, chunk_embeddings)
    context = "\n\n".join(relevant_chunks)
    
    print("\nAnalyzing...\n")
    answer = ask_claude(client, context, question)
    print(f"Answer: {answer}")
    print("\n" + "-" * 60)