def split_long_text(text:str,max_words:int)->list[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]



    
def chunk_for_clip(product: dict, max_words: int = 40) -> list[str]:
    core_parts = [p for p in [product.get("name"), product.get("brand"), product.get("color"), product.get("material")] if p]
    core_text = " | ".join(core_parts)

    core_words = core_text.split()
    if len(core_words) > max_words // 2:
        core_text = " ".join(core_words[:max_words // 2])
    core_word_count = len(core_text.split())

    bullets = [b for b in product.get("bullet_points", []) if b]
    if not bullets:
        return [core_text] if core_text else []

    remaining_budget = max(max_words - core_word_count, 10)
    expanded_bullets = []
    for b in bullets:
        if len(b.split()) > remaining_budget:
            expanded_bullets.extend(split_long_text(b, remaining_budget))
        else:
            expanded_bullets.append(b)

    chunks = []
    current_bullets = []
    current_word_count = core_word_count
    for bullet in expanded_bullets:
        bw = len(bullet.split())
        if current_bullets and current_word_count + bw > max_words:
            chunks.append(" | ".join([core_text] + current_bullets) if core_text else " | ".join(current_bullets))
            current_bullets = [bullet]
            current_word_count = core_word_count + bw
        else:
            current_bullets.append(bullet)
            current_word_count += bw
    if current_bullets:
        chunks.append(" | ".join([core_text] + current_bullets) if core_text else " | ".join(current_bullets))
    return chunks