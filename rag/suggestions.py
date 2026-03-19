from rag.llm import client

def generate_suggestions(context, query):

    prompt = f"""
Generate 3 short follow-up questions.

Context:
{context[:1500]}

User Question:
{query}

Return only questions.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    text = response.choices[0].message.content.strip()

    questions = [
        q.strip("- ").strip()
        for q in text.split("\n")
        if q.strip() and "page_content" not in q.lower()
    ]

    return questions[:3]