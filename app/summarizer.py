import re
from collections import Counter
from functools import lru_cache
from transformers import pipeline

# Model readiness flag
MODEL_READY = False

# Practical token limit for pipeline model
MAX_ABS_TOKENS = 1024

# Get pipeline
@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Handles extractive method
def extractive(text: str, max_sent: int = 3)-> str:
    text = (text or "").strip()
    if not text:
        return ""

    sents = re.split(r'(?<=[.!?])\s+', text)
    if len(sents) <= max_sent:
        return text

    freqs = Counter(w for w in re.findall(r'\w+', text.lower()) if len(w) > 3)

    scored = []
    for i, s in enumerate(sents):
        score = sum(freqs.get(w.lower(), 0) for w in re.findall(r'\w+', s))
        scored.append((score, i, s))

    # pick top-N by score, then restore original order
    top = sorted(sorted(scored, key=lambda x: x[0], reverse=True)[:max_sent], key=lambda x: x[1])
    return " ".join(s for _, _, s in top) if top else text

# Handles abstractive method with pipeline
def abstractive(text: str, max_len=120, min_len=30):
    p = get_summarizer()
    out = p(text, max_length=max_len, min_length=min_len, do_sample=False)
    return out[0]["summary_text"]

# Load pipeline model
def warmup_model():
    global MODEL_READY
    try:
        get_summarizer()
        MODEL_READY = True
    except Exception:
        MODEL_READY = False

# Count tokens in input text
def count_tokens(text: str) -> int:
    p = get_summarizer()
    return len(p.tokenizer.encode(text, truncation=False))