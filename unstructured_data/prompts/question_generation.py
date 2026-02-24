q_prompt = """
Generate 5–8 high-quality questions from the following chunk.
Only use information present in the text.
Return the output strictly as JSON matching the schema:

{schema}

Chunk:
{chunk}
"""

a_prompt = """
Answer the following question using only the text below. If the answer is not in the text, return "NOT_ANSWERABLE_FROM_TEXT".

Text:
{chunk}

Question:
{question}
"""