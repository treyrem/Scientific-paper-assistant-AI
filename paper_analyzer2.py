# Scientific Paper Analysis Tool
# Version 5.1: Layout Analysis + Extractive Sentences + OpenAI Synthesis (Loads Key from .env file)

import os
import re
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any, TypedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import torch  # Still needed for ML refinement potentially
from tqdm import tqdm
import datetime  # Import datetime
import statistics  # For font size analysis

# PDF Extraction
import pdfplumber  # Keep for metadata fallback if needed
import fitz  # PyMuPDF

# --- OpenAI Integration ---
try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None
    OpenAIError = None
    logging.warning("OpenAI library not found. pip install openai")
# --- End OpenAI Integration ---

# --- .env File Loading ---
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    logging.warning(
        "python-dotenv library not found. pip install python-dotenv. API key cannot be loaded from .env file."
    )
# --- End .env File Loading ---


# NLP and ML
from transformers import (  # Keep for section classifier
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,  # Added for embedding fallback
)

# from sentence_transformers import SentenceTransformer # Comment out if not used for extractive
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # Keep for keywords & sentence scoring

# from sklearn.metrics.pairwise import cosine_similarity # Comment out if not used

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources if needed
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords", quiet=True)


# Setup logging
logger = logging.getLogger(__name__)


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
    type: int  # 0 for text, 1 for image
    bbox: Tuple[float, float, float, float]
    lines: List[FitzLine]


# --- Dataclasses ---
@dataclass
class PaperSection:
    """A section of an academic paper with relevant metadata"""

    title: str
    content: str
    section_type: str
    page_numbers: List[int]
    confidence: float = 0.0
    start_block_idx: Optional[int] = None  # Track block indices
    end_block_idx: Optional[int] = None


@dataclass
class KeyConcept:
    """A key concept or term extracted from the paper"""

    term: str
    definition: str
    importance_score: float
    source_sections: List[str]
    context: str


@dataclass
class PaperAnalysis:
    """Complete analysis results for an academic paper"""

    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    sections: Dict[str, PaperSection] = field(default_factory=dict)
    # Store extractive summaries (used as input for OpenAI)
    abstract_summary: Optional[str] = None
    introduction_summary: Optional[str] = None
    methods_summary: Optional[str] = None
    results_summary: Optional[str] = None
    discussion_summary: Optional[str] = None
    conclusion_summary: Optional[str] = None
    key_concepts: List[KeyConcept] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    # Store OpenAI generated summaries
    full_summary: str = ""
    significance: str = ""

    def to_json(self) -> str:
        """Convert the analysis to a JSON string"""
        data = asdict(self)
        if "sections" in data and isinstance(data["sections"], dict):
            data["sections"] = {
                k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                for k, v in self.sections.items()
            }
        return json.dumps(data, indent=2)

    def save_to_file(self, output_path: str) -> None:
        """Save the analysis to a JSON file"""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(self.to_json())
        except IOError as e:
            logger.error(f"Error writing analysis to file {output_path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving file: {e}")


# --- Main Processor Class ---
class PaperProcessor:
    """Processes papers using layout analysis, extractive sentences, and OpenAI synthesis."""

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
        use_gpu: bool = True,
        section_classifier_model: str = "allenai/scibert_scivocab_uncased",
        embedding_model: str = "allenai/specter",
        openai_api_key: Optional[str] = None,  # Key is passed from main
        openai_model: str = "o3-mini",
    ):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        self.section_classifier_model_name = section_classifier_model
        self.embedding_model_name = embedding_model
        self.openai_api_key = openai_api_key  # Store the key passed from main
        self.openai_model = openai_model
        self.openai_client = None

        # Initialize OpenAI client if key is provided and library exists
        if self.openai_api_key and OpenAI:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.logger.info(
                    f"OpenAI client initialized for model: {self.openai_model}"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        elif not OpenAI:
            self.logger.warning(
                "OpenAI library not installed, synthesis step will be skipped."
            )
        else:
            self.logger.warning(
                "OpenAI API key not provided or loaded, synthesis step will be skipped."
            )

        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.logger.warning("NLTK stopwords not found. Downloading...")
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))
        self.sentence_vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
        self.load_models()

    def load_models(self):
        """Load required models (Classifier, Keyword TFIDF)"""
        self.logger.info("Loading NLP models...")
        self.section_model = None
        self.section_tokenizer = None
        self.embedding_model = None
        self.embedding_automodel = None
        self.embedding_tokenizer = None
        self.keyword_tfidf = None

        # Try loading section classifier
        try:
            self.section_tokenizer = AutoTokenizer.from_pretrained(
                self.section_classifier_model_name
            )
            num_labels_for_classifier = len(self.CORE_SECTION_TYPES)
            self.logger.info(
                f"Loading section classifier with num_labels={num_labels_for_classifier}"
            )
            self.section_model = AutoModelForSequenceClassification.from_pretrained(
                self.section_classifier_model_name,
                num_labels=num_labels_for_classifier,
                ignore_mismatched_sizes=True,
            ).to(self.device)
            self.section_model.eval()
        except Exception as e:
            self.logger.warning(
                f"Could not load section classifier model '{self.section_classifier_model_name}': {e}. ML refinement disabled."
            )
            self.section_model = None
            self.section_tokenizer = None

        self.logger.info("Skipping embedding model loading for extractive version.")

        # Keyword extraction TFIDF
        try:
            self.keyword_tfidf = TfidfVectorizer(
                max_features=100, stop_words=list(self.stop_words), ngram_range=(1, 2)
            )
            self.logger.info(
                "Models loaded (TF-IDF for keywords, potentially Classifier)."
            )
        except Exception as e:
            self.logger.error(f"Error initializing keyword TF-IDF: {e}")
            self.keyword_tfidf = None

    def extract_text_from_pdf(
        self, pdf_path: str
    ) -> Tuple[Optional[List[Tuple[int, FitzBlock]]], Optional[str], List[Dict]]:
        """Extract structured block data and plain text from PDF."""
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
        """Extracts plain text from a FitzBlock dictionary."""
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
        """Estimates dominant font size and boldness in a text block."""
        sizes = []
        flags = []
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
                    if flag is not None:
                        flags.extend([flag] * char_len)
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
        """Identify sections using layout analysis (font size, bold) and regex."""
        self.logger.info("Identifying sections using layout analysis and regex...")
        if not blocks_with_page:
            self.logger.warning("No text blocks for section identification.")
            return {}
        sections: Dict[str, PaperSection] = {}
        potential_headers = []

        # --- Pass 1: Calculate baseline font size ---
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

        # --- Pass 2: Identify potential headers ---
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

        # --- Pass 3: Group content blocks ---
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

        # Abstract Heuristic
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

        # ML Refinement is disabled
        self.logger.info("ML section refinement step is disabled.")

        self.logger.info(f"Final identified sections: {list(sections.keys())}")
        return sections

    def _estimate_page_numbers_plain(
        self, start_char: int, end_char: int, full_plain_text: str
    ) -> List[int]:
        """Estimate page numbers based on character offsets in plain text with markers."""
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
        """Try to extract abstract using heuristics (operates on plain text)."""
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
        """Use ML model to refine section classification and confidence (DISABLED)"""
        self.logger.warning("ML section refinement is currently disabled.")
        return sections  # Return sections unchanged

    def extract_key_concepts(
        self, sections: Dict[str, PaperSection]
    ) -> Tuple[List[KeyConcept], List[str]]:
        """Extract key concepts and keywords using TF-IDF and context"""
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
        """Extract a definition for a term based on sentence patterns"""
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
        """Calculate importance score based on frequency and location"""
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
        """Extracts the top N sentences from text based on TF-IDF scores."""
        if not text:
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
                self.logger.warning(
                    f"TF-IDF failed (empty vocab). Falling back to first {num_sentences} sentences."
                )
                return self.extract_key_sentences(text, num_sentences)
            else:
                raise ve
        except Exception as e:
            self.logger.error(f"Error in TF-IDF sentence extraction: {e}")
            return self.extract_key_sentences(text, num_sentences)

    def extract_key_sentences(self, text: str, num_sentences: int = 3) -> str:
        """Fallback: Extracts the first N sentences from the text."""
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
        """Uses OpenAI API to synthesize a summary from extracted sentences."""
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
        """Generate extractive summaries and then synthesize using OpenAI."""
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

        # --- Step 1: Extract key sentences using TF-IDF ---
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

        # --- Step 2: Prepare input and call OpenAI for final summaries ---
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

        # Fallback if OpenAI failed or wasn't used
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
        """Concatenates specified extractive summaries for OpenAI input."""
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
        """Fallback: Generate a structured summary by concatenating key section EXTRACTIVE summaries."""
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
        """Fallback: Generate a significance statement by concatenating Abstract and Conclusion EXTRACTIVE summaries."""
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
        """Extract metadata like title, authors, year, DOI using heuristics from plain text."""
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
            full_summary=summaries.get(
                "full", ""
            ),  # Get the OpenAI synthesized summary (or fallback)
            significance=summaries.get(
                "significance", ""
            ),  # Get the OpenAI synthesized significance (or fallback)
            # Store the intermediate extractive summaries as well
            abstract_summary=summaries.get("abstract_summary"),
            introduction_summary=summaries.get("introduction_summary"),
            methods_summary=summaries.get("methods_summary"),
            results_summary=summaries.get("results_summary"),
            discussion_summary=summaries.get("discussion_summary"),
            conclusion_summary=summaries.get("conclusion_summary"),
        )
        self.logger.info(f"Finished processing paper: {pdf_path}")
        return analysis


def main():
    """Main function to run the paper processor from command line"""
    parser = argparse.ArgumentParser(
        description="Process academic papers (Layout + Extractive + OpenAI Synthesis)"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: alongside PDF with _analysis_openai.json suffix)",
        default=None,
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage (for ML refinement if enabled)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    # --- Add OpenAI arguments ---
    # **** REMOVED --openai-key argument ****
    parser.add_argument(
        "--openai-model",
        help="OpenAI model to use for synthesis",
        default="gpt-3.5-turbo",
    )
    # --- End OpenAI arguments ---

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # --- Load API Key from .env file ---
    openai_api_key = None
    if load_dotenv:
        # Construct the absolute path to the .env file
        env_path = Path(
            r"C:\LabGit\Scientific-paper-assistant-AI\api_keys\OPEN_AI_KEY.env"
        ).resolve()
        if env_path.is_file():
            logger.info(f"Loading OpenAI API key from: {env_path}")
            load_dotenv(dotenv_path=env_path)
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning(f"OPENAI_API_KEY not found in {env_path}.")
        else:
            logger.warning(f".env file not found at specified path: {env_path}")
    else:
        logger.warning("python-dotenv not installed. Cannot load key from .env file.")

    # Check for OpenAI library if key was loaded
    if openai_api_key and OpenAI is None:
        logger.error(
            "OpenAI API key loaded, but the 'openai' library is not installed. Please run 'pip install openai'."
        )
        return
    elif not openai_api_key:
        logger.warning(
            "OpenAI API key not found in .env file or environment variables. OpenAI synthesis will be skipped."
        )
    # --- End Load API Key ---

    if not os.path.exists(args.pdf_path):
        logger.error(f"Input PDF not found: {args.pdf_path}")
        return
    if not args.output:
        output_dir = os.path.dirname(args.pdf_path) or "."
        base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        args.output = os.path.join(
            output_dir, f"{base_name}_analysis_openai.json"
        )  # Changed default suffix
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

    try:
        # Pass loaded OpenAI key to processor
        processor = PaperProcessor(
            use_gpu=not args.no_gpu,
            openai_api_key=openai_api_key,  # Pass the loaded key
            openai_model=args.openai_model,
        )
        logger.info(f"Processing paper: {args.pdf_path}")
        analysis = processor.process_paper(args.pdf_path)
        if analysis:
            logger.info(f"Attempting to save analysis to: {args.output}")
            analysis.save_to_file(args.output)
            print(f"\nAnalysis complete. Results saved to: {args.output}")
            print(
                "\n=== Paper Analysis Quick Look (OpenAI Synthesis) ==="
            )  # Updated title
            print(f"Title: {analysis.title or 'N/A'}")
            print(
                f"Authors: {', '.join(analysis.authors) if analysis.authors else 'N/A'}"
            )
            print(f"Year: {analysis.publication_year or 'N/A'}")
            print(f"DOI: {analysis.doi or 'N/A'}")
            # Display synthesized summaries
            if analysis.full_summary:
                print(
                    f"\nSynthesized Full Summary ({args.openai_model}):\n{analysis.full_summary}"
                )
            if analysis.significance:
                print(
                    f"\nSynthesized Significance ({args.openai_model}):\n{analysis.significance}"
                )
            print("\nKeywords:")
            print(
                ", ".join(analysis.keywords) if analysis.keywords else "None extracted"
            )
            print(f"\nFull analysis details saved in: {args.output}")
        else:
            print("Paper processing failed. Check logs for details.")
    except ImportError as ie:
        if "datetime" in str(ie):
            logger.error("datetime module required.")
        elif "openai" in str(ie):
            logger.error("openai library required. Please run 'pip install openai'")
        elif "dotenv" in str(ie):
            logger.error(
                "python-dotenv library required. Please run 'pip install python-dotenv'"
            )
        else:
            logger.error(f"ImportError: {ie}.")
        print(f"Error: Missing required library. Details: {ie}")
    except Exception as e:
        logger.exception("Unexpected error in main workflow.")
        print(f"An error occurred: {e}.")


if __name__ == "__main__":
    main()
