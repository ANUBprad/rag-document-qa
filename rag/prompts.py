from langchain.prompts import PromptTemplate

def get_prompt(mode, context, question, doc_type):

    if mode == "Ask Questions":
        return f"""
Answer based on the context.

Context:
{context}

Question:
{question}
"""

    elif mode == "Summarize":
        return f"""
Summarize the document clearly.

Context:
{context}
"""

    elif mode == "Analyze":
        return f"""
Analyze this {doc_type} document.

Provide:
- Key points
- Strengths
- Weaknesses
- Suggestions

Context:
{context}
"""

    elif mode == "Extract Insights":
        return f"""
Extract structured insights:

- Key topics
- Important entities
- Trends

Context:
{context}
"""

    elif mode == "Compare Documents":
        return f"""
Compare multiple documents and highlight differences.

Context:
{context}
"""

    return question