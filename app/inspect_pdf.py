
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "sample.pdf"

keywords = [
    "dataset",
    "seven",
    "mental health conditions",
    "depression",
    "anxiety",
]

with pymupdf.open(PDF_PATH) as document:
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")
        lower_text = text.lower()

        matches = [
            keyword
            for keyword in keywords
            if keyword in lower_text
        ]

        if matches:
            print(f"\n{'=' * 20} PAGE {page_number} {'=' * 20}")
            print(f"Matched keywords: {', '.join(matches)}")
            print(text[:2500])
