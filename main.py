import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF


MODEL_PATH = "/Users/cgoodwin/.lmstudio/models/mlx-community/GLM-OCR-bf16"

OCR_PROMPT = (
    "Text Recognition: Extract all visible text from this scanned page as clean Markdown. "
    "Preserve headings, paragraphs, footnotes, page structure, and line order as much as possible. "
    "Do not summarize. Do not modernize spelling. Do not correct grammar. "
    "Transcribe the page as faithfully as possible."
)

DPI = 300
MAX_TOKENS = 4096


def clean_mlx_output(raw_output: str) -> str:
    """
    Extract only the model's generated text and strip logs.
    """
    text = raw_output.strip()

    # Remove deprecation warning
    text = re.sub(
        r"Calling `.*?` directly is deprecated\..*?\n",
        "",
        text,
        flags=re.DOTALL,
    )

    # Split on separators and try to grab the useful middle section
    parts = text.split("==========")
    if len(parts) >= 3:
        text = parts[-2].strip()

    # Remove <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove token stats
    text = re.sub(r"Prompt:.*?tokens-per-sec", "", text, flags=re.DOTALL)
    text = re.sub(r"Generation:.*?tokens-per-sec", "", text, flags=re.DOTALL)
    text = re.sub(r"Peak memory:.*", "", text, flags=re.DOTALL)

    return text.strip()


def pdf_page_to_temp_image(page, page_number: int, temp_dir: Path) -> Path:
    zoom = DPI / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    image_path = temp_dir / f"page_{page_number:04d}.png"
    pix.save(image_path)

    return image_path


def run_glm_ocr(image_path: Path) -> str:
    result = subprocess.run(
        [
            "python",
            "-m",
            "mlx_vlm",
            "generate",
            "--model",
            MODEL_PATH,
            "--image",
            str(image_path),
            "--prompt",
            OCR_PROMPT,
            "--max-tokens",
            str(MAX_TOKENS),
            "--temperature",
            "0.0",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Unknown OCR error")

    cleaned = clean_mlx_output(result.stdout)

    # Debug fallback: if cleaning nuked everything, show raw output
    if not cleaned:
        print("WARNING: Cleaned output empty. Raw output below:\n")
        print(result.stdout)
        raise RuntimeError("OCR returned empty text after cleaning.")

    return cleaned


def ocr_page_with_retry(page, page_number: int, temp_dir: Path) -> str:
    image_path = pdf_page_to_temp_image(page, page_number, temp_dir)

    for attempt in range(1, 3):
        try:
            print(f"Working on page {page_number} — attempt {attempt}")
            return run_glm_ocr(image_path)

        except Exception as error:
            print(f"ERROR on page {page_number}, attempt {attempt}: {error}")

            if attempt == 2:
                return (
                    f"<!-- OCR FAILED FOR PAGE {page_number}. "
                    f"Error: {str(error)} -->"
                )

    return f"<!-- OCR FAILED FOR PAGE {page_number}. Unknown error. -->"


def process_pdf(pdf_path: Path, output_dir: Path) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF.")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{pdf_path.stem}.md"

    doc = fitz.open(pdf_path)

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)

        with output_path.open("w", encoding="utf-8") as out:
            out.write(f"# OCR Output: {pdf_path.name}\n\n")

            for index, page in enumerate(doc, start=1):
                out.write(f"\n\n<!-- page {index} -->\n\n")

                page_text = ocr_page_with_retry(page, index, temp_dir)
                out.write(page_text)
                out.write("\n")

    print(f"\nDone. Markdown written to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="OCR a PDF with GLM-OCR (MLX) and output a combined Markdown file."
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the input PDF (e.g., input/test.pdf)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for output Markdown file",
    )

    args = parser.parse_args()

    process_pdf(args.pdf, args.output_dir)


if __name__ == "__main__":
    main()