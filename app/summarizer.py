import re
from collections import Counter
from functools import lru_cache
from transformers import pipeline

# Practical token limit for pipeline model
MAX_ABS_TOKENS = 1024

@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extractive(text: str, max_sent: int = 3):
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    f = Counter(w for w in re.findall(r'\w+', text.lower()) if len(w) > 3)
    score = lambda sent: sum(f.get(w.lower(),0) for w in re.findall(r'\w+', sent))
    return " ".join(sorted(s, key=score, reverse=True)[:max_sent]) or text

def abstractive(text: str, max_len=120, min_len=30):
    p = get_summarizer()
    out = p(text, max_length=max_len, min_length=min_len, do_sample=False)
    return out[0]["summary_text"]

def count_tokens(text: str) -> int:
    p = get_summarizer()
    return len(p.tokenizer.encode(text, truncation=False))