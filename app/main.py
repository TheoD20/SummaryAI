from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Form, Request, HTTPException
from threading import Thread
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .summarizer import extractive, abstractive, warmup_model, count_tokens, MAX_ABS_TOKENS, get_summarizer

app = FastAPI(title="Summarizer API", version="0.1.0")

# Start warmup in background so server becomes responsive quickly
Thread(target=warmup_model, daemon=True).start()

# Templates setup
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

## Handles incoming data JSON API
class TextIn(BaseModel):  
    text: str; 
    method: str = "auto"

## Routes
# server renderer UI
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# JSON endpoint
@app.post("/summarize")
def summarize_json(p: TextIn):
    start = perf_counter()
    method = p.method
    if method == "auto":
        method = "extractive" if len(p.text.split()) < 120 else "auto"

    if method == "extractive":
        summary = extractive(p.text)
    else:
        tokens = count_tokens(p.text)
        if tokens > MAX_ABS_TOKENS:
            if method == "abstractive":
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Input too long for abstractive model: {tokens} tokens > {MAX_ABS_TOKENS}. "
                        "Try extractive mode or shorten the text."
                    ),
                )
            else :
                method = "extractive"
                summary = extractive(p.text)
        else:
            method = "abstractive"
            summary = abstractive(p.text)

    ms = int((perf_counter() - start) * 1000)
    return {
        "summary": summary,
        "method": method,
        "ms": ms,
        "word_count": len(p.text.split()),
    }

# htmx post - html response
@app.post("/hx/summarize", response_class=HTMLResponse)
def summarize_hx(request: Request, text: str = Form(...), method: str = Form("auto")):
    start = perf_counter()
    chosen = method if method != "auto" else ("extractive" if len(text.split()) < 120 else "auto")

    try:
        if chosen == "extractive":
            summary = extractive(text)
        else:
            tokens = count_tokens(text)
            if tokens > MAX_ABS_TOKENS:
                if chosen == "abstractive":
                    summary = (f"Input too long for abstractive model: {tokens} tokens > {MAX_ABS_TOKENS}. "
                           "Switch to extractive mode or shorten the input.")
                    chosen = "error"
                else:
                    chosen = "extractive"
                    summary = extractive(text)
            else:
                chosen = "abstractive"
                summary = abstractive(text)
    except Exception as e:
        summary = f"Error: {e}"
        chosen = "error"

    ms = int((perf_counter() - start) * 1000)
    ctx = {"request": request, "summary": summary, "method": chosen, "ms": ms, "word_count": len(text.split())}
    return templates.TemplateResponse("_summary_result.html", ctx)

# Debug: confirm backend & loaded model
@app.get("/debug/backend")
def debug_backend():
    p = get_summarizer()
    m = p.model
    return {
        "framework": "torch",
        "device": str(next(m.parameters()).device),  # e.g., "cpu"
        "model": getattr(m, "name_or_path", type(m).__name__),
    }

# health connection test
@app.get("/health")
def health():
    return {"status": "ok"}
