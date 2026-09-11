"""Conservative meeting extraction from explicit Portuguese references."""

import re
import unicodedata


def infer_meeting(query: str) -> int | None:
    text = unicodedata.normalize("NFKD", query.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    # A comparison/range must never silently narrow to only one of its meetings.
    if re.search(
        r"\b(entre|compare|comparar|comparacao|versus|vs)\b|\b\d{1,3}[ao]?\s*(?:e|a|ate|-|para(?:\s+a)?)\s*\d{1,3}[ao]?\b",
        text,
    ):
        return None
    matches = re.findall(
        r"\b(\d{1,3})(?:[ao])?\s+reuniao\b|\b(?:reuniao|ata)\s*(?:n[.o°º]*\s*)?(\d{1,3})(?!\d)",
        text,
    )
    meetings = {int(a or b) for a, b in matches if int(a or b) > 0}
    return next(iter(meetings)) if len(meetings) == 1 else None
