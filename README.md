# TextSummarizerAI

A small AI text summarizer you can actually read and extend.
Mainly in python, backend is FastAPI (Uvicorn); UI is HTMX + Tailwind (CDN) + Jinja2.

Two summarization modes:
- Abstractive — Hugging Face Transformers (DistilBART on PyTorch CPU)
- Extractive — lightweight frequency-based sentence selection (no extra deps)

---

## Tech Stack

- **Python 3.12**, **FastAPI**, **Uvicorn**
- **HTMX**, **Tailwind (CDN)**, **Jinja2**, **python-multipart**
- **Transformers** (Hugging Face) + **PyTorch (CPU)** for abstractive summaries

> Model: `sshleifer/distilbart-cnn-12-6` (distilled BART fine-tuned on CNN/DailyMail)

---

## Features

- **API + UI**: JSON endpoints for programmatic use and an HTMX page for humans.
- **Two summarizers**:  
  - *Abstractive*: `sshleifer/distilbart-cnn-12-6` (fast distilled BART)  
  - *Extractive*: tiny frequency scorer (fast, dependency-free)
- **Auto mode**: chooses a strategy based on input length.
- **Model warm-up**: preloads the HF pipeline in the background.
- **Copy / Download**: one-click copy to clipboard & export summary as `.txt`.

---

## Getting Started

### 1) Clone & create a virtual environment

**Windows (PowerShell):**
```powershell
git clone <your-repo-url> TextSummarizerAI
cd TextSummarizerAI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
**MacOS/Linux:**
```bash
git clone <your-repo-url> TextSummarizerAI
cd TextSummarizerAI
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
uvicorn app.main:app --reload
```
Open: http://127.0.0.1:8000/

On first run the abstractive model will download.

---

## JSON Endpoints (quick tour)

- GET /health – {"status":"ok"}

- POST /summarize – JSON API
body: {"text": "...", "method": "auto|extractive|abstractive"}
returns: {"summary": "...", "method": "...", "ms": 123, "word_count": 456}

- POST /hx/summarize – HTMX form handler (returns HTML partial card)

- GET /debug/backend – shows model/backend info (cpu, model id)

# Example of usage:

**Windows (PowerShell):**
```powershell
# POST /summarize – JSON API
Invoke-WebRequest http://127.0.0.1:8000/summarize `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"text":"<Text_to_summarize>", "method":"<method>"}' |
  % Content

# GET /debug/backend
Invoke-WebRequest http://127.0.0.1:8000/debug/backend | % Content

´´´
---

## License
This project is licensed under the [MIT License](LICENSE).