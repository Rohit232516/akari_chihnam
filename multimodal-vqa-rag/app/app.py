import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image
import tempfile
import time

from src.pipeline import VQAPipeline

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal VQA",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Multimodal VQA Assistant")
st.caption("Ask questions about images — powered by RAG + Gemini")

# ─────────────────────────────────────────────
# INIT PIPELINE (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return VQAPipeline()

pipeline = load_pipeline()

# ─────────────────────────────────────────────
# SESSION STATE (CHAT HISTORY)
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# DISPLAY CHAT
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], use_column_width=True)

# ─────────────────────────────────────────────
# INPUT AREA
# ─────────────────────────────────────────────
st.divider()

col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.chat_input("Ask something about an image...")

with col2:
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

# ─────────────────────────────────────────────
# HANDLE INPUT
# ─────────────────────────────────────────────
if user_input:

    # Save image temporarily if uploaded
    image_path = None
    image_preview = None

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_preview = image

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file.name)
        image_path = temp_file.name

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "image": image_preview
    })

    with st.chat_message("user"):
        st.markdown(user_input)
        if image_preview:
            st.image(image_preview, use_column_width=True)

    # ─────────────────────────────────────────
    # ASSISTANT RESPONSE
    # ─────────────────────────────────────────
    with st.chat_message("assistant"):

        # Loading UI
        with st.spinner("Thinking... 🤔"):
            start = time.time()

            try:
                answer = pipeline.query(
                    question=user_input,
                    image_path=image_path
                )
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

            elapsed = time.time() - start

        # Output
        st.markdown(answer)

        # Feedback
        st.caption(f"⏱️ Response time: {elapsed:.2f}s")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })