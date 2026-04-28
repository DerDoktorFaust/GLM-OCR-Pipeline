# Qwen2.5-VL OCR Pipeline

A local OCR pipeline using Qwen2.5-VL (MLX) to transcribe scanned PDFs into a single Markdown file.

This project is designed for archival and historical research workflows where accuracy and control over OCR output matter, without relying on external services.

---

## Features

- Processes a single PDF at a time
- Converts each page to an image (in memory)
- Uses Qwen2.5-VL for OCR
- Outputs one combined Markdown file
- Inserts page markers (`<!-- page N -->`)
- Retries failed pages once
- Logs errors and continues processing
- No intermediate image files saved

---

## Project Structure

```
LLMOCR/
├── main.py
├── input/
├── output/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.10+
- Qwen2.5-VL model installed locally (via LM Studio or manually)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DerDoktorFaust/Qwen2.5VL-OCR-Pipeline.git
cd Qwen2.5VL-OCR-Pipeline
```

---

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install mlx-vlm pymupdf pillow torch torchvision
```

---

### 4. Install the model

Install the model using LM Studio:

```
mlx-community/Qwen2.5-VL-7B-Instruct-8bit
```

By default, it will be stored at:

```
/Users/cgoodwin/.lmstudio/models/mlx-community/Qwen2.5-VL-7B-Instruct-8bit
```

If your path differs, update this line in `main.py`:

```python
MODEL_PATH = "your/model/path"
```

---

## Usage

### 1. Place your PDF in the input folder

```
input/your_file.pdf
```

---

### 2. Run the OCR script

```bash
python main.py input/your_file.pdf
```

---

### 3. Output

The result will be saved as:

```
output/your_file.md
```

---

## Output Format

The generated Markdown file includes page markers:

```markdown
<!-- page 1 -->

Transcribed text...

<!-- page 2 -->

Transcribed text...
```

If a page fails after retry:

```markdown
<!-- OCR FAILED FOR PAGE 3. Error: ... -->
```

---

## Notes on Accuracy

This is an **LLM-based OCR system**, not a traditional OCR engine.

### Advantages
- Handles difficult scans and degraded documents well
- Better at interpreting layout and structure

### Limitations
- May normalize or slightly alter text
- May hallucinate unclear portions
- Not guaranteed to be a perfect transcription

For archival work, always verify against the original document.

---

## Development Notes

After confirming everything works, freeze your environment:

```bash
pip freeze > requirements.txt
```

This ensures reproducibility across systems.

---

## License

MIT