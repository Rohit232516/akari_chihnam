import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",       "moondream")
OLLAMA_HOST        = os.getenv("OLLAMA_HOST",        "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.1))
OLLAMA_TOP_P       = float(os.getenv("OLLAMA_TOP_P",       0.9))
OLLAMA_MAX_TOKENS  = int(os.getenv("OLLAMA_MAX_TOKENS",    512))


SYSTEM_PROMPT = """You are an expert visual analyst.
You will be shown one or more retrieved images and asked a question about them.
Answer accurately based only on what you can observe in the images.
Focus on:
- Object counts: how many of each type of object
- Spatial locations: where objects are positioned (top-left, center, bottom-right etc.)
- Relationships: how objects relate to each other
Be specific and concise. Do not guess or hallucinate details."""


class VQAGenerator:
    """
    Sprint 3 — Visual Question Answering generation.

    Takes the top-k images retrieved by the retrieval pipeline
    and the user's question, passes both to moondream via Ollama,
    and returns a grounded answer.

    This is the final step in the pipeline:
    top-4 image paths + question → moondream → VQA answer
    """

    def __init__(self):
        import ollama as _ollama
        self._ollama = _ollama
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Check Ollama server is running and model is available."""
        try:
            models = self._ollama.list()
            available = [m.model for m in models.models]
            if not any(OLLAMA_MODEL in m for m in available):
                logger.warning(
                    f"Model '{OLLAMA_MODEL}' not found in Ollama. "
                    f"Available: {available}. "
                    f"Run: ollama pull {OLLAMA_MODEL}"
                )
            else:
                logger.info(f"VQAGenerator ready — model: {OLLAMA_MODEL}")
        except Exception as e:
            logger.error(
                f"Cannot connect to Ollama at {OLLAMA_HOST}. "
                f"Run 'ollama serve' first. Error: {e}"
            )

    def generate(
        self,
        question:    str,
        image_paths: list[str],
        model:       str | None = None,
    ) -> str:
        """
        Generate a VQA answer from retrieved images and a question.

        Args:
            question:    the user's question
            image_paths: list of absolute/relative paths to retrieved images
                         (top-k from the reranker — typically 4)
            model:       override the model from .env (optional)

        Returns:
            answer string from the model
        """
        model = model or OLLAMA_MODEL

        if not image_paths:
            logger.warning("No images passed to VQAGenerator")
            return "No images were retrieved for this query."

        # filter to only paths that actually exist on disk
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_paths:
            logger.error(f"None of the image paths exist: {image_paths}")
            return "Could not load the retrieved images."

        if len(valid_paths) < len(image_paths):
            logger.warning(
                f"{len(image_paths) - len(valid_paths)} image paths were invalid "
                f"and skipped"
            )

        logger.info(
            f"Generating answer for: '{question[:60]}' "
            f"using {len(valid_paths)} image(s) with {model}"
        )

        try:
            response = self._ollama.chat(
                model=model,
                messages=[
                    {
                        "role":    "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role":    "user",
                        "content": question,
                        "images":  valid_paths,
                    },
                ],
                options={
                    "temperature": OLLAMA_TEMPERATURE,
                    "top_p":       OLLAMA_TOP_P,
                    "num_predict": OLLAMA_MAX_TOKENS,
                },
            )
            answer = response["message"]["content"].strip()
            logger.info(f"Answer generated ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"VQA generation failed: {e}")
            return f"Generation failed: {e}"