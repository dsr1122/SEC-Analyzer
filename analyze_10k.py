import boto3
import json
from pypdf import PdfReader

# Load and extract text from the 10-K
def extract_text(pdf_path, max_pages=10):
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        text += page.extract_text() + "\n"
    return text

# Ask Claude a question about the document
def ask_claude(client, context, question):
    prompt = f"""You are a financial analyst assistant. Use only the document excerpt below to answer the question. 
If the answer isn't in the document, say so.

DOCUMENT:
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
text = extract_text("GS_2025_10K.pdf")
print(f"Extracted {len(text)} characters from the 10-K\n")
print("=" * 60)

for question in QUESTIONS:
    print(f"\nQUESTION: {question}\n")
    answer = ask_claude(client, text, question)
    print(f"ANSWER: {answer}")
    print("=" * 60)