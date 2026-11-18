"""Character set for the ViT text-line recogniser (CTC index 0 is the blank)."""
CHARS = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;-/()%<>=+_&'#*[]"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
N_CLASSES = len(CHARS) + 1

# Look-alike characters that appear in real reports -> the closest character in CHARS.
NORMALISE = {"–": "-", "—": "-", "‐": "-", "−": "-", "’": "'", "‘": "'", "“": "'", "”": "'", "×": "x",
             "µ": "u", "²": "2", "³": "3", "≥": ">", "≤": "<", " ": " "}


def clean(text: str) -> str:
    out = []
    for ch in text:
        ch = NORMALISE.get(ch, ch)
        out.append(ch if ch in CHAR2IDX else " ")
    return " ".join("".join(out).split())


def encode(text: str) -> list[int]:
    return [CHAR2IDX[c] for c in text]
