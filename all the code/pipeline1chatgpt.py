import argparse
import fitz
import requests
from bs4 import BeautifulSoup
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

nltk.download("punkt", quiet=True)

CLASSIFIER_MODEL = "your-org/scideberta-cs-section-classifier"
tokenizer_clf = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
model_clf = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL)
model_clf.eval()

LABELS = ["abstract", "introduction", "methods", "results", "conclusion"]


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def parse_grobid_sections(
    pdf_path: str, grobid_url: str = "http://localhost:8070"
) -> dict:
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{grobid_url}/api/processFulltextDocument", files={"input": f}
        )
    soup = BeautifulSoup(resp.content, "lxml")
    sections = {}
    for div in soup.find_all("div", attrs={"type": True}):
        sec = div["type"].lower()
        if sec in LABELS:
            text = " ".join(p.get_text() for p in div.find_all("p"))
            sections[sec] = text
    return sections


def classify_section(text: str) -> str:
    inputs = tokenizer_clf(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model_clf(**inputs).logits
    idx = torch.argmax(logits, dim=1).item()
    return LABELS[idx]


def regex_fallback_segment(raw_text: str) -> dict:
    parts = re.split(r"\n([A-Z][A-Za-z ]{2,50})\n", raw_text)
    sections = {}
    for i in range(1, len(parts), 2):
        heading = parts[i].strip().lower()
        body = parts[i + 1].strip()
        if heading in LABELS:
            sections[heading] = body
    return sections


def classification_fallback_segment(raw_text: str) -> dict:
    paras = [p.strip() for p in re.split(r"\n{2,}", raw_text) if p.strip()]
    sections = {label: [] for label in LABELS}
    for p in paras:
        label = classify_section(p)
        sections[label].append(p)
    return {label: " ".join(ps) for label, ps in sections.items() if ps}


def extractive_summary(text: str, sentence_count: int = 3) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary)


def run_pipeline(pdf_path: str) -> dict:
    raw = extract_text(pdf_path)
    try:
        sections = parse_grobid_sections(pdf_path)
    except Exception:
        sections = {}
    if not sections:
        sections = regex_fallback_segment(raw)
    if not sections:
        sections = classification_fallback_segment(raw)
    summaries = {}
    for label, content in sections.items():
        if content.strip():
            summaries[label] = extractive_summary(content)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline 1: PDF to Sections to Summaries"
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    args = parser.parse_args()
    for label, summ in run_pipeline(args.pdf).items():
        print(f"== {label.upper()} ==")
        print(summ, "\n")
