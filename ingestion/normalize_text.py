from dataclasses import dataclass

@dataclass
class LocalizedResult:
    value: object
    language_used: str | None
    is_fallback: bool

def get_localized_single(field, preferred: str = "en_US", preferred_prefix: str = "en_")->LocalizedResult:

    if not field:
        return LocalizedResult(value=None,language_used=None,is_fallback=True)
    for entry in field:
        if entry.get("language_tag") == preferred:
            return LocalizedResult(value=entry.get("value"), language_used=preferred, is_fallback=False)
    for entry in field:
        lang = entry.get("language_tag") or ""
        if lang.startswith(preferred_prefix):
            return LocalizedResult(value=entry.get("value"), language_used=lang, is_fallback=False)
    first=field[0]
    return LocalizedResult(
        value=first.get("value"),
        language_used=first.get("language_tag"),
        is_fallback=True,

    )
def get_localized_list(field, preferred: str = "en_US") -> LocalizedResult:
    if not field:
        return LocalizedResult(value=[], language_used=None, is_fallback=True)

    by_lang: dict[str, list[str]] = {}
    for entry in field:
        lang = entry.get("language_tag")
        by_lang.setdefault(lang, []).append(entry.get("value"))

    if preferred in by_lang:
        return LocalizedResult(value=by_lang[preferred], language_used=preferred, is_fallback=False)

    fallback_lang = max(by_lang, key=lambda l: len(by_lang[l]))
    return LocalizedResult(value=by_lang[fallback_lang], language_used=fallback_lang, is_fallback=True)

    
