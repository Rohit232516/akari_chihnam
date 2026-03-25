import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PIL import Image
import tempfile
import time
try:
    import streamlit as st
    for key, val in st.secrets.items():
        os.environ.setdefault(str(key), str(val))
except Exception:
    from dotenv import load_dotenv
    load_dotenv()

from src.pipeline import VQAPipeline

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Multimodal VQA",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 860px; }
    .stChatMessage { border-radius: 12px; }
    .upload-hint { color: #888; font-size: 13px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Pipeline init ────────────────────────────
@st.cache_resource
def load_pipeline():
    return VQAPipeline()

pipeline = load_pipeline()

# ── Session state ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None

if "pending_image_path" not in st.session_state:
    st.session_state.pending_image_path = None

# ── Header ────────────────────────────────────
st.title("🧠 Multimodal VQA Assistant")
st.caption("Powered by Gemini 2.5 Flash · Upload images · Ask anything")
st.divider()

# ── Chat history ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image") is not None:
            st.image(msg["image"], width=400)
        st.markdown(msg["content"])

# ── Sidebar: image upload ─────────────────────
with st.sidebar:
    st.header("📎 Attach Image")
    st.caption("Upload an image to ask questions about it.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Attached image", use_container_width=True)

        # save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(tmp.name)
        st.session_state.pending_image      = image
        st.session_state.pending_image_path = tmp.name
        st.success("Image ready — now ask your question")
    else:
        # clear if uploader is cleared
        st.session_state.pending_image      = None
        st.session_state.pending_image_path = None

    st.divider()
    st.caption("**How it works**")
    st.caption("• Upload an image → Gemini answers visually")
    st.caption("• No image → answers from indexed corpus via RAG")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat input ────────────────────────────────
user_input = st.chat_input("Ask anything...")

if user_input:
    image         = st.session_state.pending_image
    image_path    = st.session_state.pending_image_path

    # ── Show user message ─────────────────────
    with st.chat_message("user"):
        if image is not None:
            st.image(image, width=400)
        st.markdown(user_input)

    st.session_state.messages.append({
        "role":    "user",
        "content": user_input,
        "image":   image,
    })

    # ── Generate answer ───────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            try:
                if image_path:
                    # image uploaded → send directly to Gemini, skip RAG
                    answer = pipeline.generator.generate(
                        question=user_input,
                        image_paths=[image_path],
                    )
                else:
                    # no image → use full RAG pipeline over indexed corpus
                    answer = pipeline.query(question=user_input)

            except Exception as e:
                answer = f"❌ Error: {str(e)}"

            elapsed = time.time() - start

        st.markdown(answer)
        st.caption(f"⏱️ {elapsed:.2f}s")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "image":   None,
    })

    # clear pending image after sending
    st.session_state.pending_image      = None
    st.session_state.pending_image_path = None