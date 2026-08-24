from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client=Groq()

def generate_answer(question: str, retrieved: list) -> dict:
    context = "\n\n".join(
        f"[Product {r.item_id if hasattr(r, 'item_id') else r['item_id']}]: "
        f"{r.full_text if hasattr(r, 'full_text') else r.get('full_text', r.get('text', ''))}"
        for r in retrieved
    )

    prompt = f"""Answer the question using ONLY the product information below.
If a product is not actually relevant to the question, do not mention it and do not include it as used.

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

    lookup = {(r.item_id if hasattr(r, "item_id") else r["item_id"]): r for r in retrieved}
    citations = []
    for item_id in used_ids:
        r = lookup.get(item_id)
        if r:
            image_url = r.image_url if hasattr(r, "image_url") else r.get("image_url")
            citations.append({"item_id": item_id, "image_url": image_url})

    return {"answer": answer_text, "citations": citations}


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