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
def ask_claude(context, question):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    
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

# Main
text = extract_text("GS_2025_10K.pdf")
print(f"Extracted {len(text)} characters from the 10-K\n")

question = "What are the primary business segments of this company?"
print(f"Question: {question}\n")

answer = ask_claude(text, question)
print(f"Answer: {answer}")