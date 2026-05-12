import os
from textwrap import dedent

import streamlit as st
from google import genai
from huggingface_hub import InferenceClient


st.set_page_config(
    page_title="Heritage Site AI Assistant",
    page_icon="🏛️",
    layout="wide",
)


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
    "interesting_facts": [
        "Taxila is often associated with one of the earliest centers of higher learning.",
        "Alexander the Great reportedly visited Taxila in 326 BC.",
        "It was an important trading center connected to Silk Road networks.",
    ],
    "past_appearance": (
        "In its prime, Taxila was a busy urban center with large stupas, monasteries, "
        "planned streets, and a strong intellectual atmosphere shaped by Indic and "
        "Hellenistic influences."
    ),
}

DEFAULT_HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def get_env(name: str) -> str:
    return os.getenv(name, "").strip()


def build_kb_context() -> str:
    lines: list[str] = []
    for key, value in TAXILA_KNOWLEDGE_BASE.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            lines.append(f"{label}: {', '.join(value)}")
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def fallback_answer(question: str) -> str:
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
    if "fact" in normalized:
        return "Interesting facts: " + "; ".join(TAXILA_KNOWLEDGE_BASE["interesting_facts"])
    return (
        "I can answer questions about Taxila's history, architecture, cultural importance, "
        "major features, and visitor information."
    )


@st.cache_resource
def get_gemini_client() -> genai.Client | None:
    api_key = get_env("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


@st.cache_resource
def get_hf_client() -> InferenceClient | None:
    api_key = get_env("HF_TOKEN")
    if not api_key:
        return None
    return InferenceClient(api_key=api_key)


def generate_answer(question: str) -> str:
    client = get_gemini_client()
    if client is None:
        return fallback_answer(question)

    prompt = dedent(
        f"""
        You are a helpful heritage-site assistant focused on Taxila.
        Answer using the knowledge base below.
        If the answer is not clearly supported by the knowledge base, say so plainly.
        Keep the tone informative and concise.

        Knowledge base:
        {build_kb_context()}

        User question:
        {question}
        """
    ).strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = (response.text or "").strip()
        return text or "I couldn't generate a response right now."
    except Exception as exc:
        return f"{fallback_answer(question)}\n\nNote: Gemini request failed: {exc}"


def build_image_prompt(answer: str) -> str:
    return (
        "Ancient Taxila in present-day Pakistan, archaeological ruins, Buddhist stupas, "
        "Hellenistic and South Asian architecture, educational heritage, cinematic daylight, "
        f"high detail. Scene inspiration: {answer[:220]}"
    )


def generate_image(answer: str):
    client = get_hf_client()
    if client is None:
        return None, "HF_TOKEN is not configured, so image generation is disabled."

    model_name = get_env("HF_IMAGE_MODEL") or DEFAULT_HF_IMAGE_MODEL
    prompt = build_image_prompt(answer)
    request_kwargs = {
        "prompt": prompt,
        "height": 768,
        "width": 768,
        "num_inference_steps": 20,
    }
    request_kwargs["model"] = model_name

    try:
        image = client.text_to_image(**request_kwargs)
        return image, None
    except Exception as exc:
        return None, f"Image generation failed: {exc}"


st.title("🏛️ Heritage Site AI Chatbot & Image Generator")
st.caption("Ask about Taxila and generate an AI artwork inspired by the response.")

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"Gemini: {'configured' if get_gemini_client() else 'not configured'}")
    st.write(f"Hugging Face: {'configured' if get_hf_client() else 'not configured'}")
    st.text_input("HF image model", value=get_env("HF_IMAGE_MODEL") or DEFAULT_HF_IMAGE_MODEL, disabled=True)
    st.markdown(
        "Set `GEMINI_API_KEY`, `HF_TOKEN`, and optionally `HF_IMAGE_MODEL` in your environment before running Streamlit."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image") is not None:
            st.image(message["image"], caption="Generated heritage concept art", use_container_width=True)

question = st.chat_input("Ask something about Taxila...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Preparing answer..."):
            answer = generate_answer(question)
        st.markdown(answer)

        with st.spinner("Generating image..."):
            image, image_error = generate_image(answer)

        if image is not None:
            st.image(image, caption="Generated heritage concept art", use_container_width=True)
        elif image_error:
            st.info(image_error)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "image": image,
        }
    )
