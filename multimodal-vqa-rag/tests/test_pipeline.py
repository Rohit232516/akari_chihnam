import os
from src.pipeline import VQAPipeline


def test_pipeline_end_to_end():
    pipeline = VQAPipeline()

    image_dir = "data/images"

    # Index first
    pipeline.index(image_dir=image_dir, force_reindex=True)

    # Take 5 images
    images = os.listdir(image_dir)[:5]

    for img in images:
        path = os.path.join(image_dir, img)

        answer = pipeline.query(
            question="What is in this image?",
            image_path=path
        )

        assert answer is not None
        assert isinstance(answer, str)
        print(f"\nImage: {img}\nAnswer: {answer}\n")