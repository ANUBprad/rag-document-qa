from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def classify_document(text):

    prompt = f"""
Classify the following document into one of these categories:

- resume
- research_paper
- legal
- medical
- financial
- academic_notes
- business_report
- general

Document:
{text[:2000]}

Only return the category name.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()