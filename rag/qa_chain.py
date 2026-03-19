from rag.llm import client
from rag.agent import agent_handler
from rag.suggestions import generate_suggestions


def create_qa_chain(retriever):

    def ask(query, doc_type):

        prompt, docs, intent = agent_handler(query, retriever, doc_type)

        context = "\n\n".join([d.page_content for d in docs])

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()

        suggestions = generate_suggestions(context, query)

        return answer, suggestions, intent

    return ask