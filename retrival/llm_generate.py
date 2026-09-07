from groq import Groq
from dotenv import load_dotenv
from retrival.dense_search import fetch_full_text_dimensions
load_dotenv()

client=Groq()

def generate_answer(question: str, retrieved: list, history: list[dict]) -> dict:
    context_parts = []
    for r in retrieved:
        item_id = r["item_id"]
        full_text, dims = fetch_full_text_dimensions(item_id)

        dims_str = ""
        if any(dims.values()):
            dims_str = (f"\nDimensions: {dims.get('height_in', '?')}in H x "
                        f"{dims.get('width_in', '?')}in W x {dims.get('length_in', '?')}in L, "
                        f"Weight: {dims.get('weight_lb', '?')} lb")

        context_parts.append(f"[Product {item_id}]: {full_text}{dims_str}")

    context = "\n\n".join(context_parts)   

    history_text = ""
    if history:
        turns = "\n\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history)
        history_text = f"Previous conversation in this session:\n{turns}\n\n"

    prompt = f"""{history_text}Answer the question using ONLY the product information below.
Do not add general knowledge and dont assume any information or do not generalise any info which is not explicitly present in the product data shown here.
If the products don't contain enough information to answer, say so honestly
rather than filling the gap with outside knowledge.

{context}
Question: {question}

Respond in EXACTLY this format:
ANSWER: <your answer, mentioning only genuinely relevant products>
USED_ITEM_IDS: <comma-separated item_ids you actually used in the answer, or NONE if none were relevant>"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content
    answer_text, used_ids = _parse_response(raw)

    lookup = {r["item_id"]: r for r in retrieved}
    citations = []
    for item_id in used_ids:
        r = lookup.get(item_id)
        if r:
            citations.append({"item_id": item_id, "image_url": r.get("image_url")})

    return {"answer": answer_text, "citations": citations}

def contextualize_query(current_question: str, history: list[dict]) -> str:
    history_text = "\n".join(f"Q: {h['question']}\nA: {h['answer'][:150]}" for h in history)
    prompt = f"""Given this conversation history:
{history_text}

Rewrite this question into a short, catalog-style search
phrase 

Rules:
- Resolve pronouns  using history above, if present.
- Strip conversational filler words and phrases.
- Keep brand names and technical terms exactly as given.
- Correct obvious spelling mistakes only if confident.
- If already a short catalog-style phrase, return unchanged.

Follow-up: {current_question}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()




def _parse_response(raw: str) -> tuple[str, list[str]]:
    answer_text = raw
    used_ids: list[str] = []

    if "USED_ITEM_IDS:" in raw:
        answer_part, ids_part = raw.split("USED_ITEM_IDS:", 1)
        answer_text = answer_part.replace("ANSWER:", "", 1).strip()
        ids_str = ids_part.strip()
        if ids_str.upper() != "NONE":
            used_ids = [i.strip() for i in ids_str.split(",") if i.strip()]

    return answer_text, used_ids