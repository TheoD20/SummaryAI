from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi import HTTPException

from .summarizer import extractive, abstractive, count_tokens, MAX_ABS_TOKENS

## App and Template setup
app = FastAPI(title="Summarizer API", version="0.1.0")
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

## Handles incoming data JSON API
class TextIn(BaseModel):  
    text: str; 
    method: str = "auto"

## Routes
# main
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# htmx post - html response
@app.post("/hx/summarize", response_class=HTMLResponse)
def summarize_hx(request: Request, text: str = Form(...), method: str = Form("auto")):
    start = perf_counter()
    chosen = method if method != "auto" else ("extractive" if len(text.split()) < 120 else "auto")
    print(f"Chosen method: {chosen}")

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
                summary = abstractive(text)
    except Exception as e:
        summary = f"Error: {e}"
        chosen = "error"

    ms = int((perf_counter() - start) * 1000)
    ctx = {"request": request, "summary": summary, "method": chosen, "ms": ms, "word_count": len(text.split())}
    return templates.TemplateResponse("_summary_result.html", ctx)

#connection test
@app.get("/health")
def health():
    return {"status": "ok"}
