import json
import math
import os
import pickle
import re
from pathlib import Path
from textwrap import dedent

import fitz
import requests
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


load_dotenv()


st.set_page_config(
    page_title="Heritage Site AI Assistant",
    page_icon="🏛️",
    layout="wide",
)


BOOKS_DIR = Path("books")
CACHE_DIR = Path(".cache")
CACHE_SCHEMA_VERSION = "v2"
DEFAULT_HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
GLM_OCR_MODEL = "zai-org/GLM-OCR"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
MAX_CONTEXT_CHUNKS = 5
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250
OCR_IMAGE_ZOOM = 1.6

TAXILA_KNOWLEDGE_BASE = {
    "name": "Taxila",
    "location": "Punjab, Pakistan, near the modern city of Rawalpindi.",
    "history": (
        "An ancient city and archaeological site dating back to the Gandhara period, "
        "with evidence of human habitation since the Neolithic era. It was a significant "
        "center of Buddhist learning and culture."
    ),
    "cultural_importance": (
        "A UNESCO World Heritage Site, renowned for archaeological remains representing "
        "Persian, Greek, and Buddhist influences. It was a crossroads of cultures and "
        "a major center of learning."
    ),
    "major_features": [
        "Bhir Mound",
        "Sirkap",
        "Sirsukh",
        "Dharmarajika Stupa and Monastery",
        "Mohra Muradu Monastery",
        "Jaulian Monastery",
    ],
    "architecture": (
        "A blend of Hellenistic, Persian, and indigenous South Asian design, especially "
        "visible in the stupas, monasteries, sculpture, and city planning."
    ),
    "visitor_information": (
        "Taxila Museum houses many site artifacts. The site is accessible by road from "
        "Islamabad and Rawalpindi, and cooler months are the most comfortable for visits."
    ),
}


def get_env(name: str) -> str:
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value
    try:
        secret_value = st.secrets.get(name, "")
        if secret_value:
            return str(secret_value).strip()
        for _, value in st.secrets.items():
            if hasattr(value, "get"):
                nested_value = value.get(name, "")
                if nested_value:
                    return str(nested_value).strip()
    except Exception:
        pass
    return ""


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(exist_ok=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", text.lower()))


def is_good_source_text(text: str) -> bool:
    normalized = text.lower()
    blocked_phrases = [
        "cornell university library",
        "there are no known copyright restrictions",
        "archive.org/details",
        "digitized by",
        "original of this book",
    ]
    if any(phrase in normalized for phrase in blocked_phrases):
        return False
    alpha_chars = sum(1 for char in text if char.isalpha())
    if alpha_chars < 80:
        return False
    return True


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_kb_context() -> str:
    lines: list[str] = []
    for key, value in TAXILA_KNOWLEDGE_BASE.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            lines.append(f"{label}: {', '.join(value)}")
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def list_book_files() -> list[Path]:
    return sorted(BOOKS_DIR.glob("*.pdf"))


def get_book_signature() -> list[dict]:
    signature: list[dict] = []
    for path in list_book_files():
        stat = path.stat()
        signature.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return signature


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if end < len(text):
            split_point = max(chunk.rfind(". "), chunk.rfind("\n"))
            if split_point > chunk_size // 2:
                end = start + split_point + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_json_file(path: Path, payload: dict) -> None:
    ensure_cache_dir()
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def read_pickle_file(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.PickleError):
        return None


def write_pickle_file(path: Path, payload) -> None:
    ensure_cache_dir()
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def get_cache_path(name: str) -> Path:
    ensure_cache_dir()
    return CACHE_DIR / name


def render_page_image(pdf_path: Path, page_number: int) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(OCR_IMAGE_ZOOM, OCR_IMAGE_ZOOM), alpha=False)
        return pixmap.tobytes("png")
    finally:
        doc.close()


def extract_ocr_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return normalize_whitespace(result)
    generated_text = getattr(result, "generated_text", "")
    if generated_text:
        return normalize_whitespace(generated_text)
    if isinstance(result, dict):
        if result.get("generated_text"):
            return normalize_whitespace(result["generated_text"])
        if result.get("text"):
            return normalize_whitespace(result["text"])
    return normalize_whitespace(str(result))


def hf_glm_ocr_page(pdf_path: Path, page_number: int) -> str:
    client = get_hf_client()
    if client is None:
        return ""

    image_bytes = render_page_image(pdf_path, page_number)

    try:
        result = client.image_to_text(image=image_bytes, model=GLM_OCR_MODEL)
        return extract_ocr_text(result)
    except Exception:
        pass

    try:
        response = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{GLM_OCR_MODEL}",
            headers={
                "Authorization": f"Bearer {get_env('HF_TOKEN')}",
                "Content-Type": "image/png",
            },
            data=image_bytes,
            timeout=180,
        )
        if response.ok:
            payload = response.json()
            return extract_ocr_text(payload)
    except Exception:
        pass

    return ""


def get_chunk_cache_name(use_ai_ocr: bool) -> str:
    suffix = "ocr" if use_ai_ocr else "text"
    return f"chunks_{suffix}_{CACHE_SCHEMA_VERSION}.json"


def build_book_chunks(use_ai_ocr: bool = False) -> list[dict]:
    chunks: list[dict] = []
    for pdf_path in list_book_files():
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = normalize_whitespace(page.extract_text() or "")
            if not page_text and use_ai_ocr:
                page_text = hf_glm_ocr_page(pdf_path, page_number)
            if not page_text:
                continue
            for part_number, chunk_text in enumerate(split_text_into_chunks(page_text), start=1):
                if not is_good_source_text(chunk_text):
                    continue
                chunks.append(
                    {
                        "book": pdf_path.name,
                        "page": page_number,
                        "part": part_number,
                        "text": chunk_text,
                        "tokens": sorted(tokenize(chunk_text)),
                    }
                )
    return chunks


def load_cached_chunks(use_ai_ocr: bool) -> list[dict] | None:
    payload = read_json_file(get_cache_path(get_chunk_cache_name(use_ai_ocr)))
    if not payload:
        return None
    if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        return None
    if payload.get("book_signature") != get_book_signature():
        return None
    return payload.get("chunks")


def save_chunk_cache(use_ai_ocr: bool, chunks: list[dict]) -> None:
    write_json_file(
        get_cache_path(get_chunk_cache_name(use_ai_ocr)),
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "book_signature": get_book_signature(),
            "use_ai_ocr": use_ai_ocr,
            "chunks": chunks,
        },
    )


@st.cache_data(show_spinner=False)
def load_book_chunks(use_ai_ocr: bool = False) -> list[dict]:
    cached_chunks = load_cached_chunks(use_ai_ocr)
    if cached_chunks is not None:
        return cached_chunks

    chunks = build_book_chunks(use_ai_ocr=use_ai_ocr)
    save_chunk_cache(use_ai_ocr, chunks)
    return chunks


@st.cache_resource
def get_hf_client() -> InferenceClient | None:
    api_key = get_env("HF_TOKEN")
    if not api_key:
        return None
    return InferenceClient(api_key=api_key, provider="hf-inference")


def groq_model_name() -> str:
    return get_env("GROQ_MODEL") or DEFAULT_GROQ_MODEL


def has_groq_client() -> bool:
    return bool(get_env("GROQ_API_KEY"))


@st.cache_data(show_spinner=False, ttl=600)
def validate_groq_key() -> tuple[bool, str]:
    api_key = get_env("GROQ_API_KEY")
    if not api_key:
        return False, "missing"
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if response.ok:
            return True, "ok"
        return False, f"http {response.status_code}"
    except Exception as exc:
        return False, str(exc)


@st.cache_data(show_spinner=False, ttl=600)
def validate_hf_key() -> tuple[bool, str]:
    api_key = get_env("HF_TOKEN")
    if not api_key:
        return False, "missing"
    try:
        HfApi(token=api_key).whoami()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def get_vector_cache_name(use_ai_ocr: bool) -> str:
    suffix = "ocr" if use_ai_ocr else "text"
    return f"tfidf_index_{suffix}_{CACHE_SCHEMA_VERSION}.pkl"


def load_cached_vector_index(use_ai_ocr: bool):
    payload = read_pickle_file(get_cache_path(get_vector_cache_name(use_ai_ocr)))
    if not payload:
        return None
    if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        return None
    if payload.get("book_signature") != get_book_signature():
        return None
    if payload.get("use_ai_ocr") != use_ai_ocr:
        return None
    return payload


def save_vector_index(use_ai_ocr: bool, vectorizer: TfidfVectorizer, matrix) -> None:
    write_pickle_file(
        get_cache_path(get_vector_cache_name(use_ai_ocr)),
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "book_signature": get_book_signature(),
            "use_ai_ocr": use_ai_ocr,
            "vectorizer": vectorizer,
            "matrix": matrix,
        },
    )


@st.cache_resource(show_spinner=False)
def build_vector_index(use_ai_ocr: bool):
    chunks = load_book_chunks(use_ai_ocr=use_ai_ocr)
    if not chunks:
        return {"available": False}

    cached_index = load_cached_vector_index(use_ai_ocr)
    if cached_index is not None:
        return {
            "available": True,
            "vectorizer": cached_index["vectorizer"],
            "matrix": cached_index["matrix"],
        }

    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    save_vector_index(use_ai_ocr, vectorizer, matrix)
    return {"available": True, "vectorizer": vectorizer, "matrix": matrix}


def prepare_search_index() -> dict[str, int | bool | str]:
    book_signature = get_book_signature()
    if not book_signature:
        return {
            "books": 0,
            "chunks": 0,
            "vector_docs": 0,
            "vector_ready": False,
            "ocr_used": False,
        }

    ocr_used = validate_hf_key()[0]
    chunks = load_book_chunks(use_ai_ocr=ocr_used)
    vector_index = build_vector_index(use_ai_ocr=ocr_used)
    return {
        "books": len(book_signature),
        "chunks": len(chunks),
        "vector_docs": len(chunks) if vector_index["available"] else 0,
        "vector_ready": bool(vector_index["available"]),
        "ocr_used": ocr_used,
    }


def retrieve_book_context(question: str, top_k: int = MAX_CONTEXT_CHUNKS) -> tuple[list[dict], str]:
    use_ai_ocr = bool(get_hf_client())
    chunks = load_book_chunks(use_ai_ocr=use_ai_ocr)
    if not chunks:
        return [], "No PDF books were found in the books folder."

    vector_index = build_vector_index(use_ai_ocr=use_ai_ocr)
    if vector_index["available"]:
        query_vector = vector_index["vectorizer"].transform([question])
        similarity_scores = (vector_index["matrix"] @ query_vector.T).toarray().ravel()
        ranked_pairs = sorted(
            zip(similarity_scores, chunks),
            key=lambda item: item[0],
            reverse=True,
        )
        ranked = [chunk for score, chunk in ranked_pairs if score > 0]
        if ranked:
            return ranked[:top_k], "Vector retrieval from cached TF-IDF index."

    query_tokens = tokenize(question)
    scored = []
    for chunk in chunks:
        overlap = len(query_tokens.intersection(chunk["tokens"]))
        if overlap:
            scored.append((overlap, len(chunk["tokens"]), chunk))
    ranked = [
        item[2]
        for item in sorted(scored, key=lambda entry: (entry[0], -entry[1]), reverse=True)
    ]
    return ranked[:top_k], "Keyword retrieval from cached PDF text."


def format_sources(chunks: list[dict]) -> str:
    return "\n".join(
        f"- {chunk['book']}, page {chunk['page']}, excerpt {chunk['part']}"
        for chunk in chunks
    )


def format_context_blocks(chunks: list[dict]) -> str:
    blocks: list[str] = []
    for chunk in chunks:
        blocks.append(
            dedent(
                f"""
                Source: {chunk['book']} | page {chunk['page']} | excerpt {chunk['part']}
                {chunk['text']}
                """
            ).strip()
        )
    return "\n\n".join(blocks)


def fallback_answer(question: str, chunks: list[dict]) -> str:
    if chunks:
        excerpt = chunks[0]["text"][:700].strip()
        return (
            "I could not use Groq, so here is the most relevant book excerpt I found:\n\n"
            f"{excerpt}\n\n"
            f"Source: {chunks[0]['book']}, page {chunks[0]['page']}"
        )

    normalized = question.lower()
    if "history" in normalized:
        return TAXILA_KNOWLEDGE_BASE["history"]
    if "location" in normalized or "where" in normalized:
        return TAXILA_KNOWLEDGE_BASE["location"]
    if "culture" in normalized or "importance" in normalized:
        return TAXILA_KNOWLEDGE_BASE["cultural_importance"]
    if "feature" in normalized or "site" in normalized or "monastery" in normalized:
        return "Major features include " + ", ".join(TAXILA_KNOWLEDGE_BASE["major_features"]) + "."
    if "architecture" in normalized:
        return TAXILA_KNOWLEDGE_BASE["architecture"]
    if "visit" in normalized or "museum" in normalized:
        return TAXILA_KNOWLEDGE_BASE["visitor_information"]
    return (
        "I can answer questions about Taxila using the loaded PDF books and the built-in "
        "Taxila summary."
    )


def groq_chat_completion(prompt: str) -> str:
    api_key = get_env("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": groq_model_name(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful heritage-site research assistant focused on Taxila. "
                        "Answer concisely, stay grounded in the supplied source material, and do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.2,
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return (payload["choices"][0]["message"]["content"] or "").strip()


def generate_answer(question: str) -> tuple[str, list[dict], str]:
    context_chunks, retrieval_mode = retrieve_book_context(question)

    groq_ok, groq_status = validate_groq_key()
    if not groq_ok:
        return fallback_answer(question, context_chunks), context_chunks, retrieval_mode

    prompt = dedent(
        f"""
        You are a careful heritage-site research assistant focused on Taxila.
        Prioritize the supplied PDF excerpts as your primary evidence base.
        Use the background knowledge only when it does not conflict with the books.
        If the excerpts do not fully support an answer, say so clearly instead of guessing.
        Keep the answer useful for an archaeologist: concrete, factual, and specific.
        End with a short "Sources:" list naming the book file and page numbers you used.

        Background knowledge:
        {build_kb_context()}

        Retrieved PDF context:
        {format_context_blocks(context_chunks) if context_chunks else "No PDF excerpts were retrieved."}

        User question:
        {question}
        """
    ).strip()

    try:
        text = groq_chat_completion(prompt)
        if text:
            return text, context_chunks, retrieval_mode
        return "I couldn't generate a response right now.", context_chunks, retrieval_mode
    except Exception as exc:
        answer = (
            f"{fallback_answer(question, context_chunks)}\n\n"
            f"Note: Groq request failed: {exc}"
        )
        return answer, context_chunks, retrieval_mode


def build_image_prompt(question: str, answer: str, context_chunks: list[dict]) -> str:
    source_facts = []
    for chunk in context_chunks[:3]:
        source_facts.append(chunk["text"][:350].strip())

    facts_text = " ".join(source_facts)[:900]
    return dedent(
        f"""
        Create an archaeologically grounded reconstruction or site-concept image of ancient Taxila.
        Base the scene on documented details from John Marshall's Taxila volumes and avoid fantasy elements.
        Preserve likely material culture, architecture, layout, and historical atmosphere inferred from the source excerpts.
        User request: {question}
        Answer summary: {answer[:500]}
        Book-derived facts: {facts_text}
        Visual goals: accurate ruins or reconstruction, stone masonry, stupas, monasteries, excavation realism,
        scholarly reference quality, natural lighting, high detail, no modern buildings, no anachronistic clothing,
        no speculative sci-fi or mythic imagery.
        """
    ).strip()


def generate_image(question: str, answer: str, context_chunks: list[dict]):
    hf_ok, hf_status = validate_hf_key()
    client = get_hf_client()
    if client is None:
        return None, "HF_TOKEN is not configured, so image generation is disabled."
    if not hf_ok:
        return None, f"Hugging Face token validation failed: {hf_status}"

    model_name = get_env("HF_IMAGE_MODEL") or DEFAULT_HF_IMAGE_MODEL
    prompt = build_image_prompt(question, answer, context_chunks)
    request_kwargs = {
        "prompt": prompt,
        "height": 768,
        "width": 768,
        "num_inference_steps": 20,
        "model": model_name,
    }

    try:
        image = client.text_to_image(**request_kwargs)
        return image, None
    except Exception as exc:
        return None, f"Image generation failed: {exc}"


st.title("🏛️ Heritage Site AI Chatbot & Image Generator")
st.caption("Ask about Taxila using the loaded books and generate source-grounded archaeological concept art.")

book_files = list_book_files()
groq_ok, groq_status = validate_groq_key()
hf_ok, hf_status = validate_hf_key()
ocr_enabled = hf_ok

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"Groq: {'configured' if groq_ok else 'not configured'}")
    if not groq_ok and has_groq_client():
        st.caption(f"Groq validation error: {groq_status}")
    st.write(f"Hugging Face: {'configured' if hf_ok else 'not configured'}")
    if not hf_ok and get_hf_client():
        st.caption(f"Hugging Face validation error: {hf_status}")
    st.text_input("Groq model", value=groq_model_name(), disabled=True)
    st.text_input("HF image model", value=get_env("HF_IMAGE_MODEL") or DEFAULT_HF_IMAGE_MODEL, disabled=True)

    st.subheader("Book Context")
    if book_files:
        st.write(f"Indexed books: {len(book_files)}")
        for path in book_files:
            st.write(f"- {path.name}")
        st.write(
            f"Scanned-page OCR: enabled with Hugging Face `{GLM_OCR_MODEL}`"
            if ocr_enabled
            else "Scanned-page OCR: disabled until `HF_TOKEN` is set"
        )
        if st.button("Prepare cached search index", use_container_width=True):
            with st.spinner("Building persistent OCR, chunk, and vector caches..."):
                st.session_state["index_status"] = prepare_search_index()
        index_status = st.session_state.get("index_status")
        if index_status:
            st.write(f"Cached chunks: {index_status['chunks']}")
            st.write(f"Vectorized chunks: {index_status['vector_docs']}")
            st.write(
                "Local vector index ready"
                if index_status["vector_ready"]
                else "Vector index unavailable; keyword fallback will be used"
            )
        else:
            st.write("Prepare the cache once, then later runs will reuse the saved files in `.cache/`.")
    else:
        st.warning("No PDF books found in the books folder.")

    st.markdown(
        "Set `GROQ_API_KEY`, `HF_TOKEN`, and optionally `GROQ_MODEL` or `HF_IMAGE_MODEL` in your environment before running Streamlit."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("retrieval_mode"):
            st.caption(message["retrieval_mode"])
        if message.get("sources"):
            st.caption(message["sources"])
        if message.get("image") is not None:
            st.image(message["image"], caption="Generated heritage concept art", use_container_width=True)

question = st.chat_input("Ask something about Taxila...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching in the sources and preparing answer..."):
            answer, context_chunks, retrieval_mode = generate_answer(question)
        st.session_state["index_status"] = prepare_search_index()
        st.markdown(answer)
        if context_chunks:
            st.caption(retrieval_mode)
            st.caption("Retrieved sources:\n" + format_sources(context_chunks))

        with st.spinner("Generating image..."):
            image, image_error = generate_image(question, answer, context_chunks)

        if image is not None:
            st.image(image, caption="Generated heritage concept art", use_container_width=True)
        elif image_error:
            st.info(image_error)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "image": image,
            "sources": "Retrieved sources:\n" + format_sources(context_chunks) if context_chunks else "",
            "retrieval_mode": retrieval_mode,
        }
    )
