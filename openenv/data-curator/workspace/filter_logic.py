import re

BAD_WORDS = ["kill", "bomb", "steal", "hack"]


def is_safe(text: str) -> bool:
    """
    Checks if the text contains harmful words.
    BUG: This naive check causes over-censorship. It will flag benign words 
    like 'skill' or 'painkiller' because they contain the substring 'kill'.
    """
    text_lower = text.lower()
    for word in BAD_WORDS:
        if re.search(rf"\b{word}\b", text_lower):  # The agent needs to change this to a regex word boundary: re.search(rf"\b{word}\b", text_lower)
            return False
    return True
