import argparse
import gc
import re
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template


MODEL_PATH = "/Users/cgoodwin/.lmstudio/models/mlx-community/Qwen2.5-VL-7B-Instruct-8bit"

OCR_PROMPT = (
    "Transcribe this page exactly. "
    "Do not summarize. "
    "Do not correct spelling. "
    "Do not modernize spelling. "
    "Preserve line breaks and paragraph structure as much as possible. "
    "If text is unreadable, write [unclear]. "
    "Return only the transcription."
)

DPI = 300
MAX_TOKENS = 4096
TEMPERATURE = 0.0


def clean_output(text: str) -> str:
    """Clean generated text without damaging transcription content."""
    text = text.strip()

    # Remove possible chat/template leftovers
    text = text.replace("<|im_end|>", "").strip()

    # Remove markdown fences if the model wraps output
    text = re.sub(r"^```(?:markdown|text)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def render_page_to_temp_image(page, page_number: int, temp_dir: Path) -> Path:
    """Render one PDF page to a temporary PNG image."""
    zoom = DPI / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    image_path = temp_dir / f"page_{page_number:04d}.png"
    pix.save(image_path)

    return image_path


def run_qwen_ocr(model, processor, image_path: Path) -> str:
    """Run OCR on one image using an already-loaded Qwen2.5-VL model."""
    formatted_prompt = apply_chat_template(
        processor,
        model.config,
        OCR_PROMPT,
        num_images=1,
    )

    result = generate(
        model,
        processor,
        formatted_prompt,
        image=[str(image_path)],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        verbose=False,
    )

    # mlx-vlm may return a GenerationResult object rather than a string
    if hasattr(result, "text"):
        output = result.text
    elif hasattr(result, "output"):
        output = result.output
    else:
        output = str(result)

    cleaned = clean_output(output)

    if not cleaned:
        raise RuntimeError("OCR returned empty text.")

    return cleaned


def ocr_page_with_retry(model, processor, page, page_number: int, temp_dir: Path) -> str:
    """OCR one page, retrying once if it fails."""
    image_path = render_page_to_temp_image(page, page_number, temp_dir)

    for attempt in range(1, 3):
        try:
            print(f"Working on page {page_number} — attempt {attempt}")
            return run_qwen_ocr(model, processor, image_path)

        except Exception as error:
            print(f"ERROR on page {page_number}, attempt {attempt}: {error}")

            if attempt == 2:
                return (
                    f"<!-- OCR FAILED FOR PAGE {page_number}. "
                    f"Error: {str(error)} -->"
                )

    return f"<!-- OCR FAILED FOR PAGE {page_number}. Unknown error. -->"


def process_pdf(pdf_path: Path, output_dir: Path) -> Path:
    """Process one PDF and write one combined Markdown file."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.md"

    print("Loading model...")
    model, processor = load(MODEL_PATH)
    print("Model loaded.")

    try:
        doc = fitz.open(pdf_path)

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)

            with output_path.open("w", encoding="utf-8") as out:
                out.write(f"# OCR Output: {pdf_path.name}\n\n")

                for index, page in enumerate(doc, start=1):
                    out.write(f"\n\n<!-- page {index} -->\n\n")

                    page_text = ocr_page_with_retry(
                        model=model,
                        processor=processor,
                        page=page,
                        page_number=index,
                        temp_dir=temp_dir,
                    )

                    out.write(page_text)
                    out.write("\n")

    finally:
        print("Unloading model...")
        del model
        del processor
        gc.collect()
        mx.clear_cache()
        print("Model unloaded.")

    print(f"\nDone. Markdown written to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="OCR a PDF with Qwen2.5-VL and output a combined Markdown file."
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to input PDF, e.g. input/test.pdf",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for output Markdown file.",
    )

    args = parser.parse_args()
    process_pdf(args.pdf, args.output_dir)


if __name__ == "__main__":
    main()