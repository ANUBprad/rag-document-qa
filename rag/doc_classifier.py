from rag.llm import client

def classify_document(docs):

    text = "\n".join([doc.page_content for doc in docs[:3]])

    prompt = f"""
Classify this document:

resume
medical
legal
research
general

Document:
{text[:1500]}

Return one word only.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    doc_type = response.choices[0].message.content.strip().lower()

    allowed = ["resume", "medical", "legal", "research", "general"]

    if doc_type not in allowed:
        return "general"

    return doc_type