
import fitz  # PyMuPDF
import re

def extract_paper_title(pdf_path):
    """
    Extract the paper title by prioritizing PDF metadata (if valid)
    and then falling back to font-size and position-based heuristics.
    Filters out arXiv-style identifiers and publication notices.
    Prefers fully uppercase title lines typical in scientific PDFs.
    """
    doc = fitz.open(pdf_path)
    try:
        # 1. Try metadata title (ignore arXiv-like IDs)
        meta_title = doc.metadata.get("title", "").strip()
        if meta_title and not re.match(r'^(arxiv:\d+\.\d+)', meta_title, re.I):
            return meta_title

        # 2. Analyze first page spans
        page = doc.load_page(0)
        spans = []  # list of (size, text, bbox)
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt:
                        size = span.get("size", 0)
                        bbox = span.get("bbox", [0,0,0,0])
                        spans.append((size, txt, bbox))
        if not spans:
            return "Could not extract title"

        # 3. Identify top-font spans (>=90% of max size), filter undesired texts
        max_size = max(s for s, _, _ in spans)
        threshold = max_size * 0.9
        big_spans = []
        for size, text, bbox in spans:
            if size < threshold:
                continue
            low = text.lower()
            if re.match(r'^(abstract|introduction|conclusion|references|figure|table)', low):
                continue
            if re.match(r'^(arxiv:\d+\.\d+)', low):
                continue
            if low.startswith('published'):
                continue
            big_spans.append((size, text, bbox))

        if big_spans:
            lines = {}
            for _, text, bbox in big_spans:
                x0, y0, _, _ = bbox
                key = round(y0 / 5) * 5
                lines.setdefault(key, []).append((x0, text))
            for key in sorted(lines.keys()):
                parts = lines[key]
                sorted_texts = [txt for _, txt in sorted(parts, key=lambda x: x[0])]
                cand = " ".join(sorted_texts)
                if cand.isupper() and len(cand) > 5:
                    return cand
            for key in sorted(lines.keys()):
                parts = lines[key]
                sorted_texts = [txt for _, txt in sorted(parts, key=lambda x: x[0])]
                title = " ".join(sorted_texts)
                if len(title) > 5:
                    return title

        # 4. Fallback: scan first 20 text lines for a plausible title
        text_lines = page.get_text().split("\n")[:20]
        for L in text_lines:
            Ls = L.strip()
            low = Ls.lower()
            if not (15 < len(Ls) < 200):
                continue
            if re.match(r'^(abstract|introduction|conclusion|references)', low):
                continue
            if re.match(r'^(arxiv:\d+\.\d+)', low):
                continue
            if low.startswith('published'):
                continue
            if re.search(r'(http|www|©|page\s+\d+)', Ls):
                continue
            if Ls.isupper():
                return Ls
            return Ls

        return "Could not extract title"
    except Exception:
        return "Could not extract title"
    finally:
        doc.close()
