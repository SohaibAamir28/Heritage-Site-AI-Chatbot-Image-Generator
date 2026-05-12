# Heritage Site AI Chatbot & Image Generator

This project has been refactored from a Colab notebook into a local Streamlit app.

## Features

- Taxila-focused chatbot using Gemini
- Image generation using Hugging Face inference
- Fallback knowledge-base answers when Gemini is unavailable

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
$env:GEMINI_API_KEY="your_key"
$env:HF_TOKEN="your_token"
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

- The original notebook remains in the workspace for reference.
- No secrets are stored in the project files anymore.
