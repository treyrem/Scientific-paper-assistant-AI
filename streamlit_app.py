# streamlit_app.py
# Integrates Paper Analysis, Quiz Generation, and Chatbot into a Streamlit interface.

# --- Core Imports ---
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, TypedDict
from dataclasses import (
    dataclass,
    field,
    asdict,
    fields,
)  # **** Added fields import ****
from pathlib import Path
import numpy as np
import torch
import datetime
import statistics
import tempfile  # For handling uploaded files

# --- Library Imports with Error Handling ---
try:
    import streamlit as st
except ImportError:
    print("Error: Streamlit library not found. Please run 'pip install streamlit'")
    st = None  # Assign None to allow script to be imported without error

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF library not found. Please run 'pip install PyMuPDF'")
    fitz = None

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None  # Assign None if import fails
    OpenAIError = Exception  # Define a base exception if openai not installed
    logging.warning("OpenAI library not found. pip install openai")

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    logging.warning("python-dotenv library not found. pip install python-dotenv.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    AutoTokenizer, AutoModelForSequenceClassification = None, None
    logging.warning("Transformers library not found. pip install transformers torch")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None
    logging.warning("scikit-learn library not found. pip install scikit-learn")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
except ImportError:
    nltk = None
    sent_tokenize = None
    stopwords = None
    logging.warning("NLTK library not found. pip install nltk")

# --- Streamlit Page Config (MUST be the first Streamlit command) ---
# Check if st exists before calling config (for non-streamlit environments)
if st:
    st.set_page_config(layout="wide", page_title="Paper Analysis Tool")

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Download NLTK data if necessary ---
# Using Streamlit's caching to avoid re-downloading
if st:  # Only define Streamlit functions if st is available

    @st.cache_resource
    def download_nltk_data():
        if not nltk:
            return  # Skip if nltk not imported
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=True)

    download_nltk_data()  # Call the download function

# Load stopwords after ensuring download
try:
    if nltk and stopwords:
        STOPWORDS = set(stopwords.words("english"))
    else:
        STOPWORDS = set()
except Exception as e:
    logger.error(f"Failed to load NLTK stopwords: {e}")
    STOPWORDS = set()  # Fallback to empty set


# --- Type Hinting for Fitz Blocks ---
class FitzSpan(TypedDict):
    size: float
    flags: int
    font: str
    color: int
    ascender: float
    descender: float
    origin: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    text: str


class FitzLine(TypedDict):
    spans: List[FitzSpan]
    bbox: Tuple[float, float, float, float]
    wmode: int
    dir: Tuple[float, float]


class FitzBlock(TypedDict):
    number: int
    type: int
    bbox: Tuple[float, float, float, float]
    lines: List[FitzLine]


# --- Dataclasses ---
@dataclass
class PaperSection:
    title: str
    content: str
    section_type: str
    page_numbers: List[int]
    confidence: float = 0.0
    start_block_idx: Optional[int] = None
    end_block_idx: Optional[int] = None


@dataclass
class KeyConcept:
    term: str
    definition: str
    importance_score: float
    source_sections: List[str]
    context: str


@dataclass
class PaperAnalysis:
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    sections: Dict[str, Any] = field(
        default_factory=dict
    )  # Use Any temporarily for flexibility
    abstract_summary: Optional[str] = None
    introduction_summary: Optional[str] = None
    methods_summary: Optional[str] = None
    results_summary: Optional[str] = None
    discussion_summary: Optional[str] = None
    conclusion_summary: Optional[str] = None
    key_concepts: List[KeyConcept] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    full_summary: str = ""
    significance: str = ""

    def get_extractive_summaries(self) -> Dict[str, Optional[str]]:
        return {
            "abstract_summary": self.abstract_summary,
            "introduction_summary": self.introduction_summary,
            "methods_summary": self.methods_summary,
            "results_summary": self.results_summary,
            "discussion_summary": self.discussion_summary,
            "conclusion_summary": self.conclusion_summary,
        }

    def to_json_serializable(self) -> Dict:
        """Converts the analysis object to a JSON-serializable dictionary."""
        data = asdict(self)
        # Ensure sections are dictionaries before saving
        if "sections" in data and isinstance(data["sections"], dict):
            serializable_sections = {}
            for k, v in data["sections"].items():
                if hasattr(
                    v, "__dataclass_fields__"
                ):  # Check if it's a dataclass instance
                    serializable_sections[k] = asdict(v)
                else:  # Assume it's already serializable (like a dict)
                    serializable_sections[k] = v
            data["sections"] = serializable_sections
        # Ensure key concepts are dictionaries
        if "key_concepts" in data and isinstance(data["key_concepts"], list):
            serializable_concepts = []
            for item in data["key_concepts"]:
                if hasattr(item, "__dataclass_fields__"):
                    serializable_concepts.append(asdict(item))
                else:
                    serializable_concepts.append(item)
            data["key_concepts"] = serializable_concepts
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "PaperAnalysis":
        """Creates a PaperAnalysis object from a dictionary (e.g., loaded from JSON)."""
        # Reconstruct PaperSection objects
        if "sections" in data and isinstance(data["sections"], dict):
            sections_obj = {}
            for k, v_dict in data["sections"].items():
                if isinstance(v_dict, dict):
                    try:
                        sections_obj[k] = PaperSection(**v_dict)
                    except TypeError:
                        logger.warning(
                            f"Could not reconstruct PaperSection '{k}' from dict."
                        )
                        sections_obj[k] = v_dict
                else:
                    logger.warning(
                        f"Unexpected type for section '{k}': {type(v_dict)}. Storing as is."
                    )
                    sections_obj[k] = v_dict
            data["sections"] = sections_obj
        # Reconstruct KeyConcept objects
        if "key_concepts" in data and isinstance(data["key_concepts"], list):
            concepts_obj = []
            for kc_dict in data["key_concepts"]:
                if isinstance(kc_dict, dict):
                    try:
                        concepts_obj.append(KeyConcept(**kc_dict))
                    except TypeError:
                        logger.warning(f"Could not reconstruct KeyConcept.")
                        concepts_obj.append(kc_dict)
                else:
                    logger.warning(
                        f"Unexpected type for key concept: {type(kc_dict)}. Storing as is."
                    )
                    concepts_obj.append(kc_dict)
            data["key_concepts"] = concepts_obj

        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


# --- Paper Processing Logic (Adapted from paper_analyzer2.py) ---
class PaperProcessor:
    """Processes papers using layout analysis and extractive summarization."""

    SECTION_PATTERNS = {
        "abstract": r"^\s*(Abstract)\s*$",
        "introduction": r"^\s*(?:[IVX\d]+\.?\d*\s+)?(Introduction|Background|Overview)\b",
        "related_work": r"^\s*(?:[IVX\d]+\.?\d*\s+)?(Related\s+Work|Literature\s+Review|Prior\s+Art)\b",
        "methods": r"^\s*(?:[IVX\d]+\.?\d*\s+)?(Methods?|Methodology|Materials?|Experimental(?:\s+Setup)?|Approach|Model(?:\s+Architecture)?|BERT)\b",
        "results": r"^\s*(?:[IVX\d]+\.?\d*\s+)?(Results?|Findings?|Observations?|Experiments?|Evaluation|Ablation\s+Studies)\b",
        "discussion": r"^\s*(?:[IVX\d]+\.?\d*\s+)?(Discussion|Interpretation|Implications)\b",
        "conclusion": r"^\s*(?:[IVX\d]+\.?\d*\s+)?(Conclusion|Summary|Future\s+Work|Future\s+Directions)\b",
        "references": r"^\s*(?:[IVX\d]+\.?\s*)?(References?|Bibliography)\b",
        "acknowledgements": r"^\s*(?:[IVX\d]+\.?\s*)?(Acknowledgements?|Acknowledgments?)\b",
        "appendix": r"^\s*(?:[IVX\d]+\.?\s*)?(Appendix|Appendices)\b",
    }
    SECTION_TYPE_MAP = {
        "abstract": "abstract",
        "introduction": "introduction",
        "related_work": "introduction",
        "methods": "methods",
        "results": "results",
        "discussion": "discussion",
        "conclusion": "conclusion",
        "references": "references",
        "acknowledgements": "acknowledgements",
        "appendix": "appendix",
    }
    CORE_SECTION_TYPES = [
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]
    HEADER_FONT_SIZE_FACTOR = 1.1
    HEADER_MIN_SIZE = 10.0
    BOLD_FLAG = 1 << 0

    def __init__(
        self,
        openai_client: Optional["OpenAI"] = None,
        openai_model: str = "gpt-3.5-turbo",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.sentence_vectorizer = (
            TfidfVectorizer(stop_words=list(STOPWORDS)) if TfidfVectorizer else None
        )
        self.keyword_tfidf = (
            TfidfVectorizer(
                max_features=100, stop_words=list(STOPWORDS), ngram_range=(1, 2)
            )
            if TfidfVectorizer
            else None
        )
        self.section_model = None
        self.section_tokenizer = None
        # ML Model loading is skipped as refinement is disabled

    # --- PDF Extraction and Section Identification Methods ---
    def extract_text_from_pdf(
        self, pdf_path: str
    ) -> Tuple[Optional[List[Tuple[int, FitzBlock]]], Optional[str], List[Dict]]:
        if not fitz:
            return None, None, []
        self.logger.info(f"Extracting text and layout from {pdf_path}")
        all_blocks_with_page = []
        full_plain_text = ""
        pages_plain_text_data = []
        try:
            doc = fitz.open(pdf_path)
            self.logger.info(f"PDF has {len(doc)} pages.")
            for page_num, page in enumerate(doc):
                try:
                    page_dict = page.get_text("dict", flags=4, sort=True)
                    blocks = page_dict.get("blocks", [])
                    for block in blocks:
                        if block.get("type") == 0 and "lines" in block:
                            all_blocks_with_page.append((page_num + 1, block))
                except Exception as e:
                    self.logger.error(
                        f"Error extracting structured blocks from page {page_num + 1}: {e}"
                    )
                page_text = page.get_text("text", sort=True)
                if not page_text.strip():
                    self.logger.warning(f"Page {page_num + 1} seems empty.")
                    pages_plain_text_data.append({"page_num": page_num + 1, "text": ""})
                    continue
                page_text = (
                    page_text.replace("\ufb00", "ff")
                    .replace("\ufb01", "fi")
                    .replace("\ufb02", "fl")
                    .replace("\ufb03", "ffi")
                    .replace("\ufb04", "ffl")
                )
                page_text = re.sub(r"(\r\n|\r|\n){2,}", "\n\n", page_text)
                page_text = re.sub(r"[ \t]+", " ", page_text).strip()
                page_marker = f"\n\n<PAGEBREAK NUM={page_num + 1}>\n\n"
                full_plain_text += page_marker + page_text
                pages_plain_text_data.append(
                    {"page_num": page_num + 1, "text": page_text}
                )
            doc.close()
            if not all_blocks_with_page:
                self.logger.error(f"Could not extract any text blocks from {pdf_path}")
            if not full_plain_text.strip():
                self.logger.error(f"Could not extract any plain text from {pdf_path}")
            self.logger.info(
                f"Successfully extracted {len(all_blocks_with_page)} text blocks."
            )
            return all_blocks_with_page, full_plain_text, pages_plain_text_data
        except fitz.fitz.FileNotFoundError:
            self.logger.error(f"PDF file not found at {pdf_path}")
            return None, None, []
        except Exception as e:
            self.logger.error(f"Error extracting text/layout from PDF {pdf_path}: {e}")
            return (
                all_blocks_with_page or None,
                full_plain_text or None,
                pages_plain_text_data,
            )

    def _get_block_text(self, block: FitzBlock) -> str:
        block_text = ""
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
                block_text += "\n"
        return block_text.strip()

    def _get_block_dominant_properties(
        self, block: FitzBlock
    ) -> Tuple[Optional[float], bool]:
        sizes = []
        total_chars = 0
        bold_chars = 0
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size")
                    flag = span.get("flags")
                    text = span.get("text", "")
                    char_len = len(text)
                    if size is not None:
                        sizes.extend([size] * char_len)
                    total_chars += char_len
                    if flag is not None and (flag & self.BOLD_FLAG):
                        bold_chars += char_len
        dominant_size = statistics.median(sizes) if sizes else None
        is_dominant_bold = total_chars > 0 and (bold_chars / total_chars) > 0.5
        return dominant_size, is_dominant_bold

    def identify_sections(
        self,
        blocks_with_page: List[Tuple[int, FitzBlock]],
        full_plain_text: Optional[str],
        pages_plain_text_data: List[Dict],
    ) -> Dict[str, PaperSection]:
        self.logger.info("Identifying sections using layout analysis and regex...")
        if not blocks_with_page:
            self.logger.warning("No text blocks for section identification.")
            return {}
        sections: Dict[str, PaperSection] = {}
        potential_headers = []
        self.logger.debug("Pass 1: Calculating baseline font size...")
        all_sizes = [
            size
            for _, block in blocks_with_page
            if (size := self._get_block_dominant_properties(block)[0]) is not None
        ]
        if not all_sizes:
            baseline_size = 10.0
            self.logger.warning("Could not determine font sizes.")
        else:
            try:
                baseline_size = statistics.median(all_sizes)
            except statistics.StatisticsError:
                baseline_size = np.mean(all_sizes) if all_sizes else 10.0
        self.logger.debug(f"Document baseline font size estimated: {baseline_size:.2f}")
        self.logger.debug("Pass 2: Identifying potential headers...")
        for idx, (page_num, block) in enumerate(blocks_with_page):
            block_text = self._get_block_text(block).strip()
            if not block_text:
                continue
            size, is_bold = self._get_block_dominant_properties(block)
            is_potential_layout_header = False
            if size and size >= self.HEADER_MIN_SIZE:
                if size > baseline_size * self.HEADER_FONT_SIZE_FACTOR:
                    is_potential_layout_header = True
                elif is_bold and size >= baseline_size * 0.95:
                    is_potential_layout_header = True
            num_lines = len(block.get("lines", []))
            num_words = len(block_text.split())
            looks_like_header_text = (
                num_lines < 3
                and num_words < 10
                and num_words > 0
                and not block_text.endswith(".")
                and (block_text[0].isupper() or block_text[0].isdigit())
            )
            if is_potential_layout_header and looks_like_header_text:
                matched_key = None
                for pattern_key, pattern in self.SECTION_PATTERNS.items():
                    try:
                        match = re.match(pattern, block_text, re.IGNORECASE)
                        if match and match.group(0) == block_text:
                            matched_key = pattern_key
                            break
                    except Exception:
                        continue
                if matched_key:
                    self.logger.debug(
                        f"Potential Header (Layout+Regex): Idx={idx}, Text='{block_text}', Key='{matched_key}'"
                    )
                    potential_headers.append((idx, page_num, block_text, matched_key))
        potential_headers.sort(key=lambda x: x[0])
        self.logger.debug("Pass 3: Grouping content blocks...")
        num_matches = len(potential_headers)
        processed_block_idx = -1
        for i in range(num_matches):
            header_block_idx, header_page_num, header_text, pattern_key = (
                potential_headers[i]
            )
            section_type = self.SECTION_TYPE_MAP.get(pattern_key, "unknown")
            if header_block_idx <= processed_block_idx or section_type == "unknown":
                continue
            next_header_block_idx = len(blocks_with_page)
            for j in range(i + 1, num_matches):
                next_idx, _, _, _ = potential_headers[j]
                if next_idx > header_block_idx:
                    next_header_block_idx = next_idx
                    break
            content_blocks_text = []
            content_pages = set([header_page_num])
            for content_idx in range(header_block_idx + 1, next_header_block_idx):
                page_num, block = blocks_with_page[content_idx]
                block_text = self._get_block_text(block)
                if block_text:
                    content_blocks_text.append(block_text)
                    content_pages.add(page_num)
            full_content = "\n\n".join(content_blocks_text).strip()
            if not full_content:
                self.logger.debug(f"Skipping '{header_text}': no content blocks.")
                continue
            page_numbers = sorted(list(content_pages))
            section_obj = PaperSection(
                title=header_text,
                content=full_content,
                section_type=section_type,
                page_numbers=page_numbers,
                confidence=0.75,
                start_block_idx=header_block_idx,
                end_block_idx=next_header_block_idx - 1,
            )
            if section_type not in sections:
                sections[section_type] = section_obj
                processed_block_idx = next_header_block_idx - 1
                self.logger.info(
                    f"Identified Section: '{section_type}' (Title: '{header_text}') Blocks: {header_block_idx+1}-{next_header_block_idx-1} Pages: {page_numbers}"
                )
            else:
                self.logger.warning(
                    f"Duplicate section type '{section_type}' (Header: '{header_text}'). Appending content."
                )
                sections[section_type].content += "\n\n" + full_content
                sections[section_type].page_numbers = sorted(
                    list(set(sections[section_type].page_numbers + page_numbers))
                )
                sections[section_type].end_block_idx = next_header_block_idx - 1
                processed_block_idx = next_header_block_idx - 1
        if "abstract" not in sections and full_plain_text:
            self.logger.debug(
                "Attempting abstract heuristic extraction from plain text..."
            )
            abstract_text, _ = self._extract_abstract_heuristic(
                full_plain_text, pages_plain_text_data
            )
            if abstract_text:
                self.logger.info("Found abstract using plain text heuristics.")
                abs_start_char = full_plain_text.find(abstract_text[:50])
                abs_page_nums = (
                    self._estimate_page_numbers_plain(
                        abs_start_char,
                        abs_start_char + len(abstract_text),
                        full_plain_text,
                    )
                    if abs_start_char != -1
                    else [1]
                )
                sections["abstract"] = PaperSection(
                    title="Abstract",
                    content=abstract_text,
                    section_type="abstract",
                    page_numbers=abs_page_nums,
                    confidence=0.5,
                )
            else:
                self.logger.warning("Abstract not found by layout or heuristics.")
        self.logger.info("ML section refinement step is disabled.")
        self.logger.info(f"Final identified sections: {list(sections.keys())}")
        return sections

    def _estimate_page_numbers_plain(
        self, start_char: int, end_char: int, full_plain_text: str
    ) -> List[int]:
        pages = set()
        current_char_count = 0
        for match in re.finditer(r"<PAGEBREAK NUM=(\d+)>", full_plain_text):
            marker_start = match.start()
            page_num = int(match.group(1))
            page_content_end_char = marker_start
            page_content_start_char = current_char_count
            if max(page_content_start_char, start_char) < min(
                page_content_end_char, end_char
            ):
                pages.add(max(1, page_num - 1))
            current_char_count = match.end()
        page_content_start_char = current_char_count
        page_content_end_char = len(full_plain_text)
        if max(page_content_start_char, start_char) < min(
            page_content_end_char, end_char
        ):
            last_page_match = list(
                re.finditer(r"<PAGEBREAK NUM=(\d+)>", full_plain_text)
            )
            last_page_num = int(last_page_match[-1].group(1)) if last_page_match else 1
            pages.add(last_page_num)
        return sorted(list(pages)) if pages else [1]

    def _extract_abstract_heuristic(
        self, text_content: str, pages_data: List[Dict]
    ) -> Tuple[Optional[str], Optional[int]]:
        try:
            text_to_search = (
                pages_data[0]["text"] if pages_data else text_content[:3000]
            )
            abstract_match = re.search(
                r"(?im)^\s*Abstract\s*?\n+(.*?)(?=\n\s*\n|\n\s*(?:[IVX\d]+\.?\d*\s+)?(?:Introduction|Background|Keywords|Related\s+Work|Methods?|Results?|Experiments?)\b)",
                text_to_search,
                re.S,
            )
            if abstract_match:
                abstract_content = abstract_match.group(1).strip()
                abstract_content = re.sub(
                    r"<PAGEBREAK NUM=\d+>", "", abstract_content
                ).strip()
                if 30 < len(abstract_content.split()) < 500:
                    page_num = 1 if pages_data else None
                    return abstract_content, page_num
                else:
                    self.logger.debug(
                        f"Heuristic abstract rejected len: {len(abstract_content.split())}"
                    )
        except Exception as e:
            self.logger.error(f"Abstract heuristic error: {e}")
        return None, None

    def _ml_refine_sections(
        self, sections: Dict[str, PaperSection]
    ) -> Dict[str, PaperSection]:
        self.logger.warning("ML section refinement is currently disabled.")
        return sections

    def extract_key_concepts(
        self, sections: Dict[str, PaperSection]
    ) -> Tuple[List[KeyConcept], List[str]]:
        self.logger.info("Extracting key concepts and keywords...")
        key_concepts = []
        keywords = []
        corpus_texts = []
        section_map = []
        for section_type in self.CORE_SECTION_TYPES:
            if section_type in sections:
                paragraphs = [
                    p.strip()
                    for p in sections[section_type].content.split("\n\n")
                    if len(p.strip().split()) > 10
                ]
                corpus_texts.extend(paragraphs)
                section_map.extend([section_type] * len(paragraphs))
        if not corpus_texts or not self.keyword_tfidf:
            self.logger.warning(
                "No substantial text or TFIDF model for keyword extraction."
            )
            return [], []
        try:
            if (
                hasattr(self.keyword_tfidf, "vocabulary_")
                and self.keyword_tfidf.vocabulary_
            ):
                tfidf_matrix = self.keyword_tfidf.transform(corpus_texts)
            else:
                tfidf_matrix = self.keyword_tfidf.fit_transform(corpus_texts)
            feature_names = self.keyword_tfidf.get_feature_names_out()
            term_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            scored_terms = sorted(
                zip(feature_names, term_scores), key=lambda x: x[1], reverse=True
            )
            keywords = [
                term
                for term, score in scored_terms
                if len(term) > 1 and not term.isdigit()
            ][:15]
            top_terms_for_concepts = keywords[:10]
            full_content_text = " ".join(corpus_texts)
            sentences = sent_tokenize(full_content_text)
            for term in top_terms_for_concepts:
                term_regex = r"\b" + re.escape(term) + r"\b"
                term_sentences = [
                    s for s in sentences if re.search(term_regex, s, re.IGNORECASE)
                ]
                if not term_sentences:
                    continue
                context = term_sentences[0]
                definition = self._extract_definition(term, term_sentences) or context
                source_sections = set()
                for i, text in enumerate(corpus_texts):
                    if re.search(term_regex, text, re.IGNORECASE):
                        source_sections.add(section_map[i])
                importance_score = self._calculate_importance_score(term, sections)
                key_concepts.append(
                    KeyConcept(
                        term=term,
                        definition=definition,
                        importance_score=importance_score,
                        source_sections=sorted(list(source_sections)),
                        context=context,
                    )
                )
        except ValueError as ve:
            if "empty vocabulary" in str(ve):
                self.logger.warning("Keyword TF-IDF vocabulary is empty.")
            else:
                self.logger.error(f"Keyword TF-IDF ValueError: {ve}")
        except Exception as e:
            self.logger.error(f"Error extracting key concepts/keywords: {e}")
        self.logger.info(
            f"Extracted {len(keywords)} keywords and {len(key_concepts)} key concepts."
        )
        return key_concepts, keywords

    def _extract_definition(self, term: str, sentences: List[str]) -> str:
        definition_patterns = [
            re.compile(
                r"\b"
                + re.escape(term)
                + r"\b\s*(?:\(.*\)\s*)?(?:is|are)\s+(?:defined\s+as|called|known\s+as)\s+(.*?(?:\.|\n|$))",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\b" + re.escape(term) + r"\b\s*(?:refers? to)\s+(.*?(?:\.|\n|$))",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\b"
                + re.escape(term)
                + r"\b\s*[,]?\s+a\s+(?:type|kind|form|method|technique|model|approach)\s+of\s+(.*?(?:\.|\n|$))",
                re.IGNORECASE | re.DOTALL,
            ),
        ]
        potential_definitions = []
        for sentence in sentences:
            found_definition = False
            for pattern in definition_patterns:
                match = pattern.search(sentence)
                if match:
                    definition_candidate = match.group(1).strip().rstrip(".")
                    if (
                        definition_candidate
                        and len(definition_candidate.split()) > 1
                        and definition_candidate.lower() != term.lower()
                    ):
                        potential_definitions.append(
                            (sentence, definition_candidate, 1)
                        )
                        found_definition = True
                        break
            if found_definition:
                continue
            term_regex = r"\b" + re.escape(term) + r"\b"
            if re.search(term_regex, sentence, re.IGNORECASE):
                potential_definitions.append((sentence, sentence, 0))
        potential_definitions.sort(key=lambda x: (-x[2], len(x[1])))
        if potential_definitions:
            best_sentence, best_definition, priority = potential_definitions[0]
            return best_definition if priority == 1 else best_sentence
        else:
            return ""

    def _calculate_importance_score(
        self, term: str, sections: Dict[str, PaperSection]
    ) -> float:
        score = 0.0
        total_occurrences = 0
        section_weights = {
            "abstract": 1.0,
            "title": 1.0,
            "introduction": 0.7,
            "conclusion": 0.8,
            "results": 0.6,
            "methods": 0.4,
            "discussion": 0.5,
            "related_work": 0.3,
            "references": 0.1,
            "appendix": 0.1,
            "acknowledgements": 0.1,
        }
        term_regex = r"\b" + re.escape(term) + r"\b"
        for section_type, section in sections.items():
            weight = section_weights.get(section_type, 0.2)
            try:
                count = len(re.findall(term_regex, section.content, re.IGNORECASE))
            except Exception:
                count = section.content.lower().count(term.lower())
            if count > 0:
                score += np.log1p(count) * weight
                total_occurrences += count
        normalized_score = min(1.0, score / 5.0)
        return round(normalized_score, 2)

    # --- Extractive Summarization Functions ---
    def extract_tfidf_sentences(self, text: str, num_sentences: int = 3) -> str:
        if not text or not self.sentence_vectorizer:
            return ""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return " ".join([re.sub(r"\s+", " ", s).strip() for s in sentences])
            tfidf_matrix = self.sentence_vectorizer.fit_transform(sentences)
            sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
            actual_num_sentences = min(num_sentences, len(sentences))
            top_indices = sentence_scores.argsort()[-actual_num_sentences:][::-1]
            top_indices_sorted = sorted(top_indices)
            key_sents = [
                re.sub(r"\s+", " ", sentences[i]).strip() for i in top_indices_sorted
            ]
            return " ".join(key_sents)
        except ValueError as ve:
            if "empty vocabulary" in str(ve):
                self.logger.warning(f"TF-IDF failed (empty vocab). Falling back.")
                return self.extract_key_sentences(text, num_sentences)
            else:
                raise ve
        except Exception as e:
            self.logger.error(f"Error in TF-IDF sentence extraction: {e}")
            return self.extract_key_sentences(text, num_sentences)

    def extract_key_sentences(self, text: str, num_sentences: int = 3) -> str:
        if not text:
            return ""
        try:
            sentences = sent_tokenize(text)
            key_sents = [
                re.sub(r"\s+", " ", s).strip() for s in sentences[:num_sentences]
            ]
            return " ".join(key_sents)
        except Exception as e:
            self.logger.error(f"Error tokenizing sentences: {e}")
            return text[:500] + "..."

    # --- OpenAI Synthesis Function ---
    def synthesize_summary_openai(
        self, input_text: str, context: str = "paper summary"
    ) -> Optional[str]:
        if not self.openai_client:
            self.logger.warning(
                f"OpenAI client not available. Skipping synthesis for {context}."
            )
            return None
        self.logger.info(
            f"Synthesizing {context} using OpenAI model: {self.openai_model}"
        )
        target_sentences = (
            "around 7-10 sentences"
            if context == "full summary"
            else "around 3-5 sentences"
        )
        prompt = f"""Synthesize the following key sentences extracted from different sections of a scientific research paper into a single, coherent, and concise {context}. Ensure the summary is factually consistent with the provided sentences and is approximately {target_sentences} long.

Extracted Sentences:
---
{input_text}
---

Synthesized {context}:"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant skilled in summarizing scientific text accurately and fluently.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=300,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            summary = response.choices[0].message.content.strip()
            self.logger.info(f"OpenAI synthesis successful for {context}.")
            return summary
        except OpenAIError as e:
            self.logger.error(f"OpenAI API error during {context} synthesis: {e}")
        except Exception as e:
            self.logger.error(
                f"Unexpected error during OpenAI {context} synthesis: {e}"
            )
        return None

    def generate_summaries(
        self, sections: Dict[str, PaperSection]
    ) -> Dict[str, Optional[str]]:
        self.logger.info(
            "Generating extractive summaries (TF-IDF) and preparing for OpenAI synthesis..."
        )
        summaries: Dict[str, Optional[str]] = {
            f"{st}_summary": None for st in self.CORE_SECTION_TYPES
        }
        summaries["full"] = None
        summaries["significance"] = None
        if "discussion" in sections:
            summaries["discussion_summary"] = None
        sections_to_extract = ["abstract", "introduction", "results", "conclusion"]
        for section_type in sections_to_extract:
            if section_type in sections:
                self.logger.debug(f"Extracting TF-IDF sentences from: {section_type}")
                try:
                    num_sents = 3
                    if section_type == "abstract":
                        num_sents = 3
                    elif section_type == "introduction":
                        num_sents = 3
                    elif section_type == "results":
                        num_sents = 4
                    elif section_type == "conclusion":
                        num_sents = 3
                    summaries[f"{section_type}_summary"] = self.extract_tfidf_sentences(
                        sections[section_type].content, num_sentences=num_sents
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error extracting TF-IDF sentences from {section_type}: {e}"
                    )
                    summaries[f"{section_type}_summary"] = self.extract_key_sentences(
                        sections[section_type].content, 3
                    )
        full_summary_input = self.prepare_openai_input(
            summaries, ["abstract", "introduction", "results", "conclusion"]
        )
        if full_summary_input:
            summaries["full"] = self.synthesize_summary_openai(
                full_summary_input, context="full summary"
            )
        else:
            self.logger.warning("No extracted sentences for full summary.")
            summaries["full"] = (
                "Could not generate full summary: Missing key section extractions."
            )
        significance_input = self.prepare_openai_input(
            summaries, ["abstract", "conclusion"]
        )
        if significance_input:
            summaries["significance"] = self.synthesize_summary_openai(
                significance_input, context="significance statement"
            )
        else:
            self.logger.warning("No extracted sentences for significance.")
            summaries["significance"] = (
                "Could not generate significance statement: Missing Abstract or Conclusion extractions."
            )
        if summaries["full"] is None:
            summaries["full"] = self.generate_structured_full_summary(summaries)
        if summaries["significance"] is None:
            summaries["significance"] = self.generate_significance_extractive(summaries)
        self.logger.info("Summarization process complete.")
        return summaries

    def prepare_openai_input(
        self,
        section_summaries: Dict[str, Optional[str]],
        sections_to_include: List[str],
    ) -> Optional[str]:
        input_parts = [
            f"{st.capitalize()}:\n{section_summaries.get(f'{st}_summary', '')}"
            for st in sections_to_include
            if section_summaries.get(f"{st}_summary")
        ]
        if not input_parts:
            return None
        return "\n\n---\n\n".join(input_parts)

    def generate_structured_full_summary(
        self, section_summaries: Dict[str, Optional[str]]
    ) -> str:
        self.logger.info(
            "Generating structured full summary by concatenating section extractions (Fallback)..."
        )
        summary_order = ["abstract", "introduction", "results", "conclusion"]
        summary_parts = [
            f"**{st.capitalize()}:** {section_summaries.get(f'{st}_summary', '')}"
            for st in summary_order
            if section_summaries.get(f"{st}_summary")
        ]
        if not summary_parts:
            self.logger.warning("No section extractions available.")
            return "Could not generate structured summary: Missing section extractions."
        return "\n\n".join(summary_parts)

    def generate_significance_extractive(
        self, section_summaries: Dict[str, Optional[str]]
    ) -> str:
        self.logger.info(
            "Generating significance statement from Abstract and Conclusion extractions (Fallback)..."
        )
        abstract_summary = section_summaries.get("abstract_summary")
        conclusion_summary = section_summaries.get("conclusion_summary")
        significance_parts = []
        if abstract_summary:
            significance_parts.append(f"**Abstract:** {abstract_summary}")
        if conclusion_summary:
            significance_parts.append(f"**Conclusion:** {conclusion_summary}")
        if not significance_parts:
            self.logger.warning(
                "Missing Abstract or Conclusion extraction for significance."
            )
            return "Could not determine significance: Missing Abstract or Conclusion extractions."
        return "\n\n".join(significance_parts)

    # --- Metadata Extraction (using plain text) ---
    def extract_metadata(
        self, full_plain_text: Optional[str], pages_plain_text_data: List[Dict]
    ) -> Dict[str, Any]:
        self.logger.info("Extracting metadata from plain text...")
        metadata = {"title": None, "authors": [], "year": None, "doi": None}
        if not full_plain_text or not pages_plain_text_data:
            return metadata
        first_page_text = pages_plain_text_data[0]["text"]
        title_end_line_index = -1
        try:  # --- Title Extraction ---
            lines = first_page_text.split("\n")
            potential_title_lines = []
            in_title = False
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or (
                    i < 5
                    and (
                        "@" in line
                        or re.match(
                            r"(?i)university|institute|department|inc\.|llc", line
                        )
                    )
                ):
                    if in_title:
                        title_end_line_index = i - 1
                        break
                        continue
                is_potential_title = (
                    len(line.split()) > 2
                    and len(line.split()) < 25
                    and not line.isupper()
                    and not re.match(
                        r"(?i)^\s*(?:abstract|introduction|keywords|contents)|(?:figure|table)\s*\d+",
                        line,
                    )
                    and not re.match(r"^\d+(\.\d+)*\s+", line)
                    and not ("arXiv:" in line or "doi:" in line or "ISBN:" in line)
                )
                if is_potential_title:
                    potential_title_lines.append(line)
                    in_title = True
                    title_end_line_index = i
                elif in_title:
                    if len(line.split()) > 5 or i > len(potential_title_lines) + 1:
                        break
                    else:
                        potential_title_lines.append(line)
                        title_end_line_index = i
                if i > 15 or re.match(r"(?i)^\s*abstract", line):
                    break
            if potential_title_lines:
                metadata["title"] = re.sub(
                    r"\s+", " ", " ".join(potential_title_lines)
                ).strip()
                self.logger.info(f"Extracted Title (heuristic): {metadata['title']}")
            else:
                self.logger.warning("Could not confidently extract title.")
                title_end_line_index = 2
        except Exception as e:
            self.logger.error(f"Error during title extraction: {e}")
            title_end_line_index = 2
        try:  # --- Author Extraction ---
            author_search_start_line = title_end_line_index + 1
            author_search_end_line = author_search_start_line + 10
            first_page_lines = first_page_text.split("\n")
            author_lines_candidates = []
            for i in range(
                author_search_start_line,
                min(author_search_end_line, len(first_page_lines)),
            ):
                line = first_page_lines[i].strip()
                if not line:
                    continue
                if re.match(
                    r"(?i)^\s*(?:abstract|introduction|keywords|index terms|e-?mail|correspondence)|(?:figure|table)\s*\d+",
                    line,
                ) or re.search(
                    r"(?i)\b(university|institute|department|inc|llc|gmbh|ltd)\b", line
                ):
                    break
                author_pattern = r"[A-Z][a-zA-Z\'\-\.]+(?:\s+[A-Z][a-zA-Z\'\-\.]+)*"
                temp_line_authors = re.split(r"\s*,\s*|\s+and\s+", line)
                cleaned_authors = []
                for potential_author in temp_line_authors:
                    potential_author = potential_author.strip()
                    if (
                        re.fullmatch(author_pattern, potential_author)
                        and len(potential_author) > 2
                        and not potential_author.isupper()
                    ):
                        if len(potential_author.split()) < 5 and not re.search(
                            r"(?i)university|institute|department|inc|llc|email",
                            potential_author,
                        ):
                            cleaned_authors.append(potential_author)
                if cleaned_authors:
                    author_lines_candidates.extend(cleaned_authors)
            seen = set()
            metadata["authors"] = [
                x for x in author_lines_candidates if not (x in seen or seen.add(x))
            ]
            self.logger.info(f"Extracted Authors (heuristic): {metadata['authors']}")
        except Exception as e:
            self.logger.error(f"Error during author extraction: {e}")
        try:  # --- Year Extraction ---
            year_match = re.search(
                r"(?:Published|Submitted|Accepted|©|\(c\)|Copyright)\s+(\d{4})",
                full_plain_text,
            )
            if not year_match:
                arxiv_match = re.search(
                    r"arXiv:.*\[.*\]\s+\d{1,2}\s+[A-Za-z]{3,}\s+(\d{4})",
                    full_plain_text,
                )
            if arxiv_match:
                year_match = arxiv_match
            if not year_match and pages_plain_text_data:
                possible_years = re.findall(
                    r"\b(19[89]\d|20[0-2]\d)\b",
                    pages_plain_text_data[0]["text"]
                    + "\n"
                    + pages_plain_text_data[-1]["text"],
                )
                if possible_years:
                    current_year = datetime.datetime.now().year + 1
                    plausible_years = [
                        int(y) for y in possible_years if 1980 <= int(y) <= current_year
                    ]
                if plausible_years:
                    metadata["year"] = max(plausible_years)
            if year_match and not metadata["year"]:
                year = int(year_match.group(1))
                current_year = datetime.datetime.now().year + 1
            if 1980 <= year <= current_year:
                metadata["year"] = year
            if metadata["year"]:
                self.logger.info(f"Extracted Year: {metadata['year']}")
            else:
                self.logger.warning("Could not extract publication year.")
        except NameError:
            self.logger.warning("datetime module not available.")
        except Exception as e:
            self.logger.error(f"Error during year extraction: {e}")
        try:  # --- DOI Extraction ---
            doi_match = re.search(
                r"(?:doi\.org/|doi:|DOI:)\s*(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
                full_plain_text,
                re.IGNORECASE,
            )
            if not doi_match:
                doi_match = re.search(
                    r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b",
                    full_plain_text,
                    re.IGNORECASE,
                )
            if doi_match:
                doi_str = (
                    doi_match.group(1) if doi_match.lastindex else doi_match.group(0)
                )
                metadata["doi"] = doi_str.strip().rstrip(".")
                self.logger.info(f"Extracted DOI: {metadata['doi']}")
            else:
                self.logger.warning("Could not extract DOI.")
        except Exception as e:
            self.logger.error(f"Error during DOI extraction: {e}")
        return metadata

    # --- Main Processing Pipeline ---
    def process_paper(self, pdf_path: str) -> Optional[PaperAnalysis]:
        """Process a paper: extract text, metadata, sections, concepts, summaries."""
        self.logger.info(f"Starting processing for paper: {pdf_path}")
        # 1. Extract text
        blocks_with_page, full_plain_text, pages_plain_text_data = (
            self.extract_text_from_pdf(pdf_path)
        )
        if not blocks_with_page and not full_plain_text:
            return None
        # 2. Identify sections
        sections = self.identify_sections(
            blocks_with_page, full_plain_text, pages_plain_text_data
        )
        # 3. Extract metadata
        metadata = self.extract_metadata(full_plain_text, pages_plain_text_data)
        # 4. Extract key concepts and keywords
        key_concepts, keywords = self.extract_key_concepts(sections)
        # 5. Generate EXTRACTIVE summaries and SYNTHESIZE with OpenAI
        summaries = self.generate_summaries(sections)
        # 6. Create the analysis object
        analysis = PaperAnalysis(
            title=metadata.get("title"),
            authors=metadata.get("authors", []),
            publication_year=metadata.get("year"),
            doi=metadata.get("doi"),
            sections=sections,
            key_concepts=key_concepts,
            keywords=keywords,
            full_summary=summaries.get("full", ""),
            significance=summaries.get("significance", ""),
            abstract_summary=summaries.get("abstract_summary"),
            introduction_summary=summaries.get("introduction_summary"),
            methods_summary=summaries.get("methods_summary"),
            results_summary=summaries.get("results_summary"),
            discussion_summary=summaries.get("discussion_summary"),
            conclusion_summary=summaries.get("conclusion_summary"),
        )
        self.logger.info(f"Finished processing paper: {pdf_path}")
        return analysis


# --- Quiz Generation Logic ---
def generate_quiz_openai(
    analysis_data: Dict[str, Any],
    openai_client: Optional["OpenAI"],
    openai_model: str,
    num_questions: int = 5,
    paper_title: Optional[str] = "the paper",
) -> Optional[List[Dict]]:
    if not openai_client:
        logger.error("OpenAI client not available for quiz generation.")
        return None
    logger.info(
        f"Generating {num_questions} quiz questions for '{paper_title}' using OpenAI model: {openai_model}"
    )
    input_parts = []
    key_sections = ["abstract", "introduction", "methods", "results", "conclusion"]
    for sec in key_sections:
        summary_key = f"{sec}_summary"
        content_to_use = analysis_data.get(summary_key)
        # **** FIX: Check type before accessing content ****
        section_info = analysis_data.get("sections", {}).get(sec)
        if not content_to_use and section_info:
            # Check if section_info is a PaperSection object or dict
            if isinstance(section_info, PaperSection):
                content_to_use = section_info.content
            elif isinstance(section_info, dict):
                content_to_use = section_info.get("content")
        if content_to_use:
            input_parts.append(f"{sec.capitalize()} Content/Summary:\n{content_to_use}")
    if not input_parts:
        logger.error("No extracted summaries/content for quiz generation.")
        return None
    input_text = "\n\n---\n\n".join(input_parts)
    logger.debug(f"Quiz Prompt Input (first 500 chars):\n{input_text[:500]}...")
    prompt = f"""Based on the following extracted content/summaries from a scientific paper titled "{paper_title}", generate exactly {num_questions} multiple-choice quiz questions to test understanding of the paper's key aspects (e.g., main topic, methods, findings, contributions).

For each question:
1. Start the question with the question number followed by a period (e.g., "1.").
2. Provide the question text immediately after the number and period.
3. On separate lines below the question, provide exactly 4 answer choices, each starting with A), B), C), or D) followed by the choice text.
4. On a separate line after the choices, clearly indicate the correct answer using the format "Correct Answer: [Letter]" (e.g., "Correct Answer: C").
5. Ensure questions and answers are directly supported by the provided text.

Extracted Content/Summaries:
---
{input_text}
---

Generate Quiz Questions:
"""
    try:
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant skilled at creating multiple-choice quizzes based on scientific text summaries, following specific formatting instructions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        quiz_text = response.choices[0].message.content.strip()
        logger.info("OpenAI quiz generation successful.")
        logger.debug(f"Raw Quiz Response:\n{quiz_text}")
        quiz_data = []
        question_blocks = re.split(r"\n\s*(?=\d+\.\s)", quiz_text)
        for block in question_blocks:
            block = block.strip()
            if not block:
                continue
            question_match = re.match(
                r"^\d+\.\s*(.*?)(?=\n\s*[A-D]\))", block, re.DOTALL | re.IGNORECASE
            )
            if not question_match:
                logger.warning(f"Could not parse question text from block:\n{block}")
                continue
            question = question_match.group(1).strip()
            choices = {"A": None, "B": None, "C": None, "D": None}
            parsed_choices = 0
            choice_matches = re.findall(
                r"([A-D])\)\s*(.*?)(?=\n\s*[A-D]\)|\n\s*Correct Answer:|$)",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            for letter, text in choice_matches:
                letter_upper = letter.upper()
                if letter_upper in choices:
                    choices[letter_upper] = text.strip()
                    parsed_choices += 1
            correct_answer_match = re.search(
                r"Correct Answer:\s*([A-D])", block, re.IGNORECASE
            )
            correct_answer = (
                correct_answer_match.group(1).upper() if correct_answer_match else None
            )
            if (
                question
                and parsed_choices == 4
                and all(choices.values())
                and correct_answer
            ):
                quiz_data.append(
                    {
                        "question": question,
                        "choices": choices,
                        "correct_answer": correct_answer,
                    }
                )
            else:
                logger.warning(
                    f"Could not fully parse question block (Q:{question is not None}, Choices:{parsed_choices}/4, Ans:{correct_answer is not None}):\n{block}"
                )
        if not quiz_data:
            logger.error("Failed to parse any valid questions from OpenAI response.")
            return None
        return quiz_data[:num_questions]
    except OpenAIError as e:
        logger.error(f"OpenAI API error during quiz generation: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI quiz generation: {e}")
    return None


# --- Chatbot Logic ---
def prepare_context_from_analysis(analysis_data: Dict, max_chars: int = 15000) -> str:
    context_parts = []
    paper_title = analysis_data.get("title", "the paper")
    context_parts.append(f"Paper Title: {paper_title}")
    core_sections = ["abstract", "introduction", "methods", "results", "conclusion"]
    sections_data = analysis_data.get("sections", {})
    for sec_type in core_sections:
        section_content = None
        if sec_type in sections_data:
            section_info = sections_data[sec_type]
            # **** FIX: Check type before accessing content ****
            if isinstance(section_info, PaperSection):
                section_content = section_info.content
            elif isinstance(
                section_info, dict
            ):  # Check if it's a dict (e.g., from JSON)
                section_content = section_info.get("content")
        if section_content:
            content = section_content
            content = re.sub(r"<PAGEBREAK NUM=\d+>", "", content)
            content = re.sub(r"\s+", " ", content).strip()
            context_parts.append(f"--- {sec_type.capitalize()} ---\n{content}")
    full_context = "\n\n".join(context_parts)
    if len(full_context) > max_chars:
        logger.warning(f"Context exceeds {max_chars} chars. Truncating.")
        full_context = full_context[:max_chars] + "\n... [Context Truncated]"
    return full_context


def get_chatbot_response(
    client: "OpenAI",
    model: str,
    conversation_history: List[Dict[str, str]],
    paper_context: str,
    user_question: str,
) -> Optional[str]:
    system_prompt = f"""You are an AI assistant designed to answer questions about a specific scientific research paper based ONLY on the provided context below. Do not use any external knowledge or information outside this context. If the answer cannot be found in the context, state that clearly.

Provided Context from the Paper:
--- START CONTEXT ---
{paper_context}
--- END CONTEXT ---

Answer the user's questions based solely on the information within the START CONTEXT and END CONTEXT markers."""
    messages_for_api = (
        [{"role": "system", "content": system_prompt}]
        + conversation_history
        + [{"role": "user", "content": user_question}]
    )
    logger.debug(f"Sending {len(messages_for_api)} messages to OpenAI.")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            temperature=0.3,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        assistant_response = response.choices[0].message.content.strip()
        logger.info("Received response from OpenAI.")
        return assistant_response
    except OpenAIError as e:
        logger.error(f"OpenAI API error during chat: {e}")
        return "Sorry, I encountered an error communicating with the AI model."
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI chat: {e}")
        return "Sorry, an unexpected error occurred."


# --- Streamlit App ---
if st:
    # --- API Key Loading ---
    @st.cache_resource
    def get_openai_key():
        if load_dotenv:
            env_path = Path(
                r"C:\LabGit\Scientific-paper-assistant-AI\api_keys\OPEN_AI_KEY.env"
            ).resolve()
            if env_path.is_file():
                logger.info(f"Loading OpenAI API key from: {env_path}")
                load_dotenv(dotenv_path=env_path)
                key = os.getenv("OPENAI_API_KEY")
                if not key:
                    logger.warning(f"OPENAI_API_KEY not found in {env_path}.")
                return key
            else:
                logger.warning(f".env file not found at {env_path}")
                return None
        else:
            logger.warning("python-dotenv not installed.")
            return None

    openai_api_key = get_openai_key()
    openai_client = None
    if openai_api_key and OpenAI:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized for Streamlit app.")
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
    elif not OpenAI:
        st.error("OpenAI library not installed. Please run `pip install openai`.")
    elif not openai_api_key:
        st.warning(
            "OpenAI API key not found in the specified .env file. Summarization, Quiz, and Chatbot tabs will be limited."
        )

    # --- Session State Initialization ---
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "paper_processor" not in st.session_state:
        st.session_state.paper_processor = PaperProcessor(
            openai_client=openai_client, openai_model="gpt-3.5-turbo"
        )
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None

    # --- UI Layout ---
    st.title("📄 Scientific Paper Assistant AI")
    tab1, tab2, tab3 = st.tabs(["📊 Analysis & Summary", "❓ Quiz", "💬 Chatbot"])

    # --- Tab 1: Analysis & Summary ---
    with tab1:
        st.header("1. Upload and Analyze PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", type="pdf", key="pdf_uploader"
        )
        analyze_button = st.button("Analyze Paper", disabled=(uploaded_file is None))

        if analyze_button and uploaded_file is not None:
            if st.session_state.processed_file_name != uploaded_file.name:
                st.session_state.analysis_data = None
                st.session_state.quiz_questions = None
                st.session_state.chat_history = []
                progress_bar = st.progress(0, text="Processing...")
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                logger.info(f"Uploaded file saved temporarily to: {tmp_file_path}")
                try:
                    if (
                        not st.session_state.paper_processor.openai_client
                        and openai_client
                    ):
                        st.session_state.paper_processor.openai_client = openai_client
                    analysis_result = st.session_state.paper_processor.process_paper(
                        tmp_file_path
                    )
                    if analysis_result:
                        st.session_state.analysis_data = (
                            analysis_result.to_json_serializable()
                        )
                        st.session_state.processed_file_name = uploaded_file.name
                        progress_bar.progress(100, text="Analysis Complete!")
                        st.success("Paper processed successfully!")
                    else:
                        st.error("Paper processing failed.")
                        st.session_state.analysis_data = None
                        progress_bar.progress(100, text="Processing Failed.")
                except Exception as e:
                    st.error(f"An error occurred during paper processing: {e}")
                    logger.exception("Error in process_paper")
                    st.session_state.analysis_data = None
                    progress_bar.progress(100, text="Processing Failed.")
                finally:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                        logger.info(f"Deleted temp file: {tmp_file_path}")
            else:
                st.info("This paper has already been analyzed.")

        st.header("2. Analysis Results")
        if st.session_state.analysis_data:
            try:
                analysis_obj = PaperAnalysis.from_dict(st.session_state.analysis_data)
            except Exception as e:
                st.error(f"Error loading analysis data: {e}")
                analysis_obj = None
            if analysis_obj:
                st.subheader("Metadata")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Title:** {analysis_obj.title or 'N/A'}")
                    st.markdown(
                        f"**Authors:** {', '.join(analysis_obj.authors) if analysis_obj.authors else 'N/A'}"
                    )
                with col2:
                    st.markdown(f"**Year:** {analysis_obj.publication_year or 'N/A'}")
                    st.markdown(f"**DOI:** {analysis_obj.doi or 'N/A'}")
                st.subheader("Keywords")
                st.write(
                    ", ".join(analysis_obj.keywords)
                    if analysis_obj.keywords
                    else "None extracted"
                )
                st.subheader("Synthesized Summary (via OpenAI)")
                st.markdown(
                    analysis_obj.full_summary
                    or "_Summary could not be generated or OpenAI key missing._"
                )
                st.subheader("Synthesized Significance (via OpenAI)")
                st.markdown(
                    analysis_obj.significance
                    or "_Significance statement could not be generated or OpenAI key missing._"
                )
                with st.expander("Show Extracted Section Sentences (Input to OpenAI)"):
                    st.write(analysis_obj.get_extractive_summaries())
                with st.expander("Show Identified Sections & Content"):
                    for sec_type, sec_data in analysis_obj.sections.items():
                        if isinstance(sec_data, PaperSection):
                            st.markdown(
                                f"**{sec_type.capitalize()} (Title: {sec_data.title}, Pages: {sec_data.page_numbers})**"
                            )
                            st.text_area(
                                f"Content_{sec_type}",
                                sec_data.content,
                                height=150,
                                key=f"sec_content_{sec_type}",
                            )
                        else:
                            st.markdown(f"**{sec_type.capitalize()}** (Raw Data)")
                            st.json(sec_data, expanded=False)
        else:
            st.info("Upload a PDF and click 'Analyze Paper' to see results.")

    # --- Tab 2: Quiz ---
    with tab2:
        st.header("❓ Paper Quiz")
        if st.session_state.analysis_data and openai_client:
            if st.button("Generate Quiz", key="generate_quiz_btn"):
                st.session_state.quiz_questions = None
                with st.spinner("Asking OpenAI to generate quiz..."):
                    st.session_state.quiz_questions = generate_quiz_openai(
                        analysis_data=st.session_state.analysis_data,
                        openai_client=openai_client,
                        openai_model=st.session_state.paper_processor.openai_model,
                        num_questions=5,
                        paper_title=st.session_state.analysis_data.get(
                            "title", "the paper"
                        ),
                    )
                    if not st.session_state.quiz_questions:
                        st.error("Quiz generation failed.")
                    else:
                        st.success("Quiz generated!")
            if st.session_state.quiz_questions:
                for i, q in enumerate(st.session_state.quiz_questions):
                    st.subheader(f"Question {i+1}:")
                    st.write(q["question"])
                    choices = q["choices"]
                    for letter, text in choices.items():
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{letter})** {text}")
                    with st.expander("Show Answer"):
                        st.write(
                            f"Correct Answer: **{q['correct_answer']}**) {choices[q['correct_answer']]}"
                        )
                        st.divider()
            elif st.session_state.analysis_data:
                st.info("Click 'Generate Quiz' to create questions.")
        elif not openai_client:
            st.warning("OpenAI client not available. Quiz generation disabled.")
        else:
            st.warning("Please upload and analyze a paper first.")

    # --- Tab 3: Chatbot ---
    with tab3:
        st.header("💬 Chat About the Paper")
        if st.session_state.analysis_data and openai_client:
            paper_context = prepare_context_from_analysis(
                st.session_state.analysis_data
            )
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt := st.chat_input("Ask a question about the paper..."):
                st.session_state.chat_history.append(
                    {"role": "user", "content": prompt}
                )
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking...")
                    assistant_response = get_chatbot_response(
                        client=openai_client,
                        model=st.session_state.paper_processor.openai_model,
                        conversation_history=st.session_state.chat_history[:-1],
                        paper_context=paper_context,
                        user_question=prompt,
                    )
                    message_placeholder.markdown(
                        assistant_response or "_Sorry, I couldn't get a response._"
                    )
                if assistant_response:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": assistant_response}
                    )
                max_history_messages = 10
                if len(st.session_state.chat_history) > max_history_messages:
                    st.session_state.chat_history = st.session_state.chat_history[
                        -max_history_messages:
                    ]
        elif not openai_client:
            st.warning("OpenAI client not available. Chatbot disabled.")
        else:
            st.warning("Please upload and analyze a paper first.")

# --- Fallback if Streamlit is not installed ---
else:
    print("Streamlit is not installed. Cannot run the web application.")
    print("Please install it using: pip install streamlit")

# --- Main execution for non-Streamlit context (optional, for testing) ---
# if __name__ == "__main__" and not st:
#      print("This script is intended to be run with Streamlit: streamlit run streamlit_app.py")
