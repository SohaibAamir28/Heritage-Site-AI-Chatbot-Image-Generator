# Heritage Site AI Chatbot & Image Generator

## Features

- Taxila-focused chatbot using Groq
- PDF-backed retrieval from the books inside `books/`
- Persistent local TF-IDF vector search over cached book chunks
- Persistent on-disk caching for extracted text, OCR results, chunks, and vector index files
- Optional OCR for scan-only PDF pages using Hugging Face `zai-org/GLM-OCR`
- Image generation using Hugging Face inference with prompts grounded in retrieved book facts

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set environment variables:

```powershell
$env:GROQ_API_KEY="your_key"
$env:HF_TOKEN="your_token"
```

Or create a local `.env` file in the project root:

```env
GROQ_API_KEY=your_key
HF_TOKEN=your_token
GROQ_MODEL=llama-3.1-8b-instant
HF_IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
```

For Streamlit Community Cloud, add the same keys in the app's **Secrets** settings:

```toml
GROQ_API_KEY = "your_key"
HF_TOKEN = "your_token"
GROQ_MODEL = "llama-3.1-8b-instant"
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
```

Optional:

```powershell
$env:HF_IMAGE_MODEL="black-forest-labs/FLUX.1-schnell"
```

If you do not set `HF_IMAGE_MODEL`, the app uses `stabilityai/stable-diffusion-xl-base-1.0`.

4. Run the app:

```powershell
streamlit run app.py
```

## Notes

- Place Taxila reference PDFs in the `books/` folder. The app currently auto-loads every `.pdf` there.
- The app also loads environment variables from a local `.env` file at startup.
- The sidebar shows whether the cached vector index is ready.
- If a PDF is scan-only, the app attempts OCR through Hugging Face `zai-org/GLM-OCR` when `HF_TOKEN` is configured. The first indexing pass can take longer because scanned pages need OCR.
- Answers are generated through Groq using `GROQ_API_KEY`, and the default chat model is `llama-3.1-8b-instant` unless `GROQ_MODEL` is set.
- Prepared search data is stored in `.cache/` and automatically reused until the source PDFs change.
- Answers prioritize retrieved excerpts from the books and include source references for the passages used.
- Generated images are prompted with factual details pulled from the retrieved book excerpts so they are more useful as archaeological concept references.
