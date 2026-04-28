# OCR Pipeline (GLM-OCR + MLX)

This project extracts text from PDFs using GLM-OCR (MLX) and outputs structured Markdown for downstream processing (e.g., summarization with LLMs).

## Features
- PDF → image conversion
- OCR via GLM-OCR (MLX)
- Markdown output (page-level + combined)
- Designed for local, offline workflows

## Setup

### 1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate