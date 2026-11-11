"""
A stand-in for Quorum's Paperclip search, which is not connected in this app.

The paper search asks "is Paperclip connected?" before it searches. Here the
answer is always no, so it uses Europe PMC, which is free and needs no key.
"""


class PaperclipError(Exception):
    pass


def connected() -> bool:
    return False


def search(question: str, n: int = 10) -> list:
    return []
