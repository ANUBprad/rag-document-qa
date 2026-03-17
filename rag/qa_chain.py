from groq import Groq
from dotenv import load_dotenv
import os
from rag.prompts import get_prompt
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def create_qa_chain(retriever):
    def ask(query, mode, doc_type):

        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = get_prompt(mode, context, query, doc_type)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content, docs
    return ask