from rag.llm import client
from rag.prompts import get_prompt


def detect_intent(query):

    prompt = f"""
Classify intent:

summarize
question
insights

Query:
{query}

Return one word.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    intent = response.choices[0].message.content.strip().lower()

    if "summarize" in intent:
        return "summarize"
    elif "insight" in intent:
        return "insights"
    else:
        return "question"


def agent_handler(query, retriever, doc_type):

    intent = detect_intent(query)

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    if intent == "summarize":
        mode = "Summarize"
    elif intent == "insights":
        mode = "Extract Insights"
    else:
        mode = "Ask Questions"

    prompt = get_prompt(mode, context, query, doc_type)

    return prompt, docs, intent