# scrape_sections.py: Extract sections from PDFs into separate .txt files

import fitz  # PyMuPDF: pip install PyMuPDF
import re
import glob
import os
import argparse


def extract_sections_from_pdfs(pdf_folder: str, output_folder: str):
    # Define section heading patterns and canonical names
    section_patterns = {
        "abstract": r"abstract",
        "introduction": r"(?:introduction|background)",
        "methods": r"(?:methods|materials\s*and\s*methods|experimental\s*methods)",
        "results": r"(?:results|results\s*and\s*discussion)",
        "discussion": r"discussion",
        "conclusion": r"(?:conclusion|conclusions|summary)",
    }

    # Regex to capture each section (with optional numbering prefix)
    regex = re.compile(
        rf"(?sm)^\s*(?:\d+\.?\s*)?({'|'.join(section_patterns.values())})\s*[:\.\n]+\s*(.*?)"
        rf"(?=^\s*(?:\d+\.?\s*)?(?:{'|'.join(section_patterns.values())})\s*[:\.\n]+|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    # Container for sentences per section
    out_sentences = {sec: [] for sec in section_patterns}

    # Iterate PDFs in the input folder
    for pdf_path in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Failed to open {pdf_path}: {e}")
            continue

        text = "".join(page.get_text() for page in doc)

        # Remove footnote or symbol-prefixed lines before extraction
        lines = text.splitlines()
        lines = [L for L in lines if not re.match(r"^\s*[\*†‡]", L)]
        filtered_text = "\n".join(lines)

        # Find all section matches
        for match in regex.finditer(filtered_text):
            raw_heading = match.group(1).strip().lower()
            content = match.group(2).strip()

            # Map raw heading to canonical key
            section_key = None
            for key, pattern in section_patterns.items():
                if re.fullmatch(pattern, raw_heading, re.IGNORECASE):
                    section_key = key
                    break
            if not section_key:
                continue

            # Split into sentences (simple regex; replace with nltk.sent_tokenize if desired)
            sentences = re.split(r"(?<=[\.\?!])\s+", content)
            sentences = [s.strip() for s in sentences if s.strip()]
            out_sentences[section_key].extend(sentences)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Write sentences to individual files
    for sec, sentences in out_sentences.items():
        file_path = os.path.join(output_folder, f"{sec}s.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))

    # Summary
    print(f"Extraction complete. Files saved in '{output_folder}' folder:")
    for sec in section_patterns:
        print(f" - {sec}s.txt ({len(out_sentences[sec])} sentences)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract key sections (abstract, methods, results, discussion, conclusion) from PDFs."
    )
    parser.add_argument(
        "pdf_folder", help="Path to folder containing PDF files, e.g., papers/"
    )
    parser.add_argument(
        "output_folder",
        help="Path to folder where section .txt files will be saved, e.g., papers/sections_output/",
    )
    args = parser.parse_args()

    extract_sections_from_pdfs(args.pdf_folder, args.output_folder)
