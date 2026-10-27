"""Conservative quote diagnostics; similarity never makes a quotation valid."""

import re
from difflib import SequenceMatcher
from itertools import zip_longest


def quote_diagnostic(quote: str | None, passages: list[str]) -> dict:
    """Identify observable differences, without claiming why the model made them."""
    if not quote or not passages:
        return {"code": "quote_not_found", "quote_excerpt": (quote or "")[:180]}
    normalized = " ".join(quote.split())
    for text in passages:
        tokens = list(re.finditer(r"\S+", text))
        normalized_text = " ".join(token.group() for token in tokens)
        start = normalized_text.find(normalized)
        if normalized and start >= 0:
            # Map normalized characters back to original offsets for diagnostics only.
            offsets = []
            for token in tokens:
                if offsets:
                    offsets.append(token.start() - 1)
                offsets.extend(range(token.start(), token.end()))
            excerpt = text[offsets[start] : offsets[start + len(normalized) - 1] + 1]
            return difference("whitespace_mismatch", quote, excerpt)
    candidates = []
    for text in passages:
        matcher = SequenceMatcher(None, quote, text, autojunk=False)
        blocks = [block for block in matcher.get_matching_blocks() if block.size]
        # Only call it non-contiguous when substantial verbatim blocks, in source
        # order, cover the entire quote except explicit omission markers/whitespace.
        substantial = [block for block in blocks if block.size >= 12]
        if len(substantial) >= 2:
            remainder = quote
            for block in reversed(substantial):
                remainder = remainder[: block.a] + remainder[block.a + block.size :]
            gaps = [b.b - (a.b + a.size) for a, b in zip(substantial, substantial[1:], strict=False)]
            if not re.sub(r"[\s.…\[\]]", "", remainder) and any(gap >= 20 for gap in gaps):
                excerpt = text[substantial[0].b : substantial[-1].b + substantial[-1].size]
                return difference("non_contiguous_quote", quote, excerpt)
        if not blocks or max(block.size for block in blocks) < 12:
            continue
        first, last = blocks[0], blocks[-1]
        start = max(0, first.b - first.a)
        end = min(len(text), last.b + last.size + len(quote) - last.a - last.size)
        excerpt = text[start:end]
        similarity = SequenceMatcher(None, quote, excerpt, autojunk=False).ratio()
        if similarity >= 0.65:
            candidates.append((similarity, excerpt))
    if candidates:
        candidates.sort(reverse=True)
        if (
            len(candidates) > 1
            and candidates[0][0] == candidates[1][0]
            and candidates[0][1] != candidates[1][1]
        ):
            return {"code": "quote_validation_failed", "quote_excerpt": quote[:180]}
        return difference("text_mismatch", quote, candidates[0][1])
    return {"code": "quote_not_found", "quote_excerpt": quote[:180]}


def difference(code: str, quote: str, source: str) -> dict:
    index = next((i for i, (a, b) in enumerate(zip_longest(quote, source)) if a != b), 0)
    start = max(0, index - 40)
    return {
        "code": code,
        "first_difference": {
            "offset": index,
            "quote_character": quote[index : index + 1] or None,
            "source_character": source[index : index + 1] or None,
        },
        "quote_excerpt": quote[start : start + 180],
        "source_excerpt": source[start : start + 180],
    }
