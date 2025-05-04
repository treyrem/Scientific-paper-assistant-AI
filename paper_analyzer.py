# Scientific Paper Analysis Tool
# A comprehensive system for academic paper summarization and analysis

import os
import re
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import datetime  # Import datetime

# PDF Extraction
import pdfplumber
import fitz  # PyMuPDF

# NLP and ML
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer,
    PegasusForConditionalGeneration,  # Added for Pegasus
    PegasusTokenizer,  # Added for Pegasus
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
# Basic config is set in main() based on args.debug
logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """A section of an academic paper with relevant metadata"""

    title: str
    content: str
    section_type: (
        str  # abstract, introduction, methods, results, discussion, conclusion
    )
    page_numbers: List[int]
    confidence: float = 0.0  # Confidence in section classification
    # Add start/end char indices if needed for precise location tracking
    # start_char: Optional[int] = None
    # end_char: Optional[int] = None

    def get_summary(self, summarizer, max_length: int = 150) -> str:
        """Generate a summary of the section using the provided summarizer"""
        # This method might not be needed if summarization is handled centrally
        # Keeping it for potential future use or direct section summary access
        if len(self.content.split()) < 30:  # If content is too short, don't summarize
            return self.content

        try:
            # Assuming summarizer is the PaperProcessor's summarize_text method
            return summarizer(self.content, max_length=max_length)
        except Exception as e:
            logger.error(f"Error summarizing {self.section_type}: {e}")
            # Return first few sentences as fallback
            try:
                sentences = sent_tokenize(self.content)
                return " ".join(sentences[:2]) + "..."
            except Exception:  # Handle potential errors in sent_tokenize
                return self.content[:150] + "..."


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

    title: Optional[str] = None  # Made optional
    authors: List[str] = field(default_factory=list)  # Made optional
    publication_year: Optional[int] = None
    doi: Optional[str] = None

    # Paper sections (raw content)
    sections: Dict[str, PaperSection] = field(default_factory=dict)

    # Section Summaries
    abstract_summary: Optional[str] = None
    introduction_summary: Optional[str] = None
    methods_summary: Optional[str] = None
    results_summary: Optional[str] = None
    discussion_summary: Optional[str] = None
    conclusion_summary: Optional[str] = None

    # Extracted data
    key_concepts: List[KeyConcept] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Generated content
    full_summary: str = ""  # Renamed from 'summary'
    significance: str = ""

    def to_json(self) -> str:
        """Convert the analysis to a JSON string"""
        # Custom dict conversion to handle potential None values gracefully
        data = asdict(self)

        # Convert PaperSection objects to dictionaries within the sections field
        if "sections" in data and isinstance(data["sections"], dict):
            # Use a comprehension to handle potential errors if a section is not a dataclass instance
            data["sections"] = {
                k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                for k, v in self.sections.items()
            }

        # Remove fields that are None or empty lists/dicts if desired, or keep them
        # Example: data = {k: v for k, v in data.items() if v is not None and v != [] and v != {}}

        return json.dumps(data, indent=2)

    def save_to_file(self, output_path: str) -> None:
        """Save the analysis to a JSON file"""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)  # Create output dir if needed
        try:
            with open(output_path, "w", encoding="utf-8") as f:  # Specify encoding
                f.write(self.to_json())
        except IOError as e:
            logger.error(f"Error writing analysis to file {output_path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving the file: {e}")


class PaperProcessor:
    """Main class for processing academic papers"""

    # Section patterns - Trying more flexible patterns, allowing optional colons, more numbering
    SECTION_PATTERNS = {
        # Abstract: Often just the word 'Abstract'
        "abstract": r"^\s*(Abstract)\s*$",
        # Introduction: Optional number (e.g., 1 or 1.), word, optional colon
        "introduction": r"^\s*(?:[IVX\d]+\.?\s*)?(Introduction|Background|Overview)\s*:?$",
        # Related Work: Optional number, phrase, optional colon
        "related_work": r"^\s*(?:[IVX\d]+\.?\d*\s*)?(Related\s+Work|Literature\s+Review|Prior\s+Art)\s*:?$",
        # Methods: Optional number, common words, optional colon
        "methods": r"^\s*(?:[IVX\d]+\.?\d*\s*)?(Methods?|Methodology|Materials?|Experimental|Approach|BERT)\s*:?$",  # Added BERT as it's a key section in that paper
        # Results/Experiments: Optional number, common words, optional colon
        "results": r"^\s*(?:[IVX\d]+\.?\d*\s*)?(Results?|Findings?|Observations?|Experiments?|Ablation\s+Studies)\s*:?$",  # Added Ablation Studies
        # Discussion: Optional number, word, optional colon
        "discussion": r"^\s*(?:[IVX\d]+\.?\d*\s*)?(Discussion|Interpretation|Implications)\s*:?$",
        # Conclusion: Optional number, common words, optional colon
        "conclusion": r"^\s*(?:[IVX\d]+\.?\d*\s*)?(Conclusion|Summary|Future\s+Work|Future\s+Directions)\s*:?$",
        # References: Optional number, word, optional colon
        "references": r"^\s*(?:[IVX\d]+\.?\s*)?(References?|Bibliography)\s*:?$",
        # Acknowledgements: Optional number, word, optional colon
        "acknowledgements": r"^\s*(?:[IVX\d]+\.?\s*)?(Acknowledgements?|Acknowledgments?)\s*:?$",
        # Appendix: Optional number, word, optional colon
        "appendix": r"^\s*(?:[IVX\d]+\.?\s*)?(Appendix|Appendices)\s*:?$",
    }
    # Mapping from detailed patterns to standard types (remains the same)
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
    # Standard section types we usually care about summarizing/analyzing
    CORE_SECTION_TYPES = [
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]

    def __init__(
        self,
        use_gpu: bool = True,
        section_classifier_model: str = "allenai/scibert_scivocab_uncased",
        summarizer_model: str = "google/pegasus-pubmed",  # Changed summarizer model
        embedding_model: str = "allenai/specter",
    ):
        """Initialize the paper processor with necessary models"""

        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        # Store model names
        self.section_classifier_model_name = section_classifier_model
        self.summarizer_model_name = summarizer_model
        self.embedding_model_name = embedding_model

        # Load models (defer heavy loading until needed or do it here)
        self.load_models()

        # Initialize NLTK tools
        self.stop_words = set(stopwords.words("english"))

    def load_models(self):
        """Load all required models"""
        self.logger.info("Loading NLP models...")
        try:
            # Section classifier (SciBERT)
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
            self.section_model.eval()  # Set to evaluation mode

            # Summarization model (Pegasus-Pubmed)
            self.summarizer_tokenizer = PegasusTokenizer.from_pretrained(
                self.summarizer_model_name
            )
            self.summarizer_model = PegasusForConditionalGeneration.from_pretrained(
                self.summarizer_model_name
            ).to(self.device)
            self.summarizer_model.eval()  # Set to evaluation mode
            self.summarizer_max_input_length = (
                self.summarizer_tokenizer.model_max_length
            )

            # Embedding model (Specter)
            self.embedding_model = SentenceTransformer(self.embedding_model_name).to(
                self.device
            )
            self.embedding_model.eval()  # Set to evaluation mode

            # Keyword extraction setup
            self.tfidf = TfidfVectorizer(
                max_features=100, stop_words="english", ngram_range=(1, 2)
            )

            self.logger.info("All models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.logger.error(
                "Please ensure models are available and dependencies are installed."
            )
            if "section_tokenizer" not in self.__dict__:
                self.section_model = None
            if "summarizer_tokenizer" not in self.__dict__:
                self.summarizer_model = None
            if "embedding_model" not in self.__dict__:
                self.embedding_model = None

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[Optional[str], List[Dict]]:
        """Extract text and structure from a PDF file using PyMuPDF"""
        self.logger.info(f"Extracting text from {pdf_path}")

        full_text = ""
        pages_data = []  # Store text per page

        try:
            doc = fitz.open(pdf_path)
            self.logger.info(f"PDF has {len(doc)} pages.")

            for page_num, page in enumerate(doc):
                page_text = page.get_text(
                    "text", sort=True
                )  # Add sort=True for reading order
                if not page_text.strip():  # Skip empty pages
                    self.logger.warning(f"Page {page_num + 1} seems empty.")
                    continue

                # Basic cleaning: replace common ligatures, normalize whitespace
                page_text = (
                    page_text.replace("\ufb00", "ff")
                    .replace("\ufb01", "fi")
                    .replace("\ufb02", "fl")
                    .replace("\ufb03", "ffi")
                    .replace("\ufb04", "ffl")
                )
                # Normalize line breaks and excessive whitespace
                page_text = re.sub(
                    r"(\r\n|\r|\n){2,}", "\n\n", page_text
                )  # Normalize paragraph breaks
                page_text = re.sub(
                    r"[ \t]+", " ", page_text
                ).strip()  # Normalize spaces/tabs

                # Add page marker - use a distinct, less common pattern
                page_marker = f"\n\n<PAGEBREAK NUM={page_num + 1}>\n\n"
                full_text += page_marker + page_text
                pages_data.append(
                    {"page_num": page_num + 1, "text": page_text}
                )  # Use 1-based indexing

            doc.close()

            if not full_text.strip():
                self.logger.error(f"Could not extract any text from {pdf_path}")
                return None, []

            self.logger.info(
                f"Successfully extracted text (approx {len(full_text)} chars)."
            )
            return full_text, pages_data

        except fitz.fitz.FileNotFoundError:
            self.logger.error(f"PDF file not found at {pdf_path}")
            return None, []
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return full_text if full_text else None, pages_data

    def identify_sections(
        self, full_text: str, pages_data: List[Dict]
    ) -> Dict[str, PaperSection]:
        """Identify and classify sections using regex and ML"""
        self.logger.info("Identifying paper sections...")
        if not full_text:
            return {}

        sections: Dict[str, PaperSection] = {}
        section_matches: List[Tuple[int, str, str]] = (
            []
        )  # (start_index, matched_header, pattern_key)

        # Add a preliminary split by lines to help regex anchoring
        lines = full_text.splitlines()
        current_char_offset = 0

        # Iterate through lines to find potential headers using anchored regex
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:  # Skip empty lines
                current_char_offset += len(line) + 1  # Account for newline character
                continue

            for pattern_key, pattern in self.SECTION_PATTERNS.items():
                try:
                    # Use re.fullmatch for potentially stricter matching on the line
                    match = re.fullmatch(pattern, line_stripped, re.IGNORECASE)
                    if match:
                        # Calculate the approximate start index in the full_text
                        # This is less precise than finditer but works with line iteration
                        start_index = current_char_offset + line.find(line_stripped)

                        matched_header = match.group(
                            0
                        ).strip()  # Use group(0) for full match
                        # Capture the core keyword (e.g., "Introduction" from group 1)
                        core_header = (
                            match.group(1) if match.groups() else matched_header
                        )

                        # Basic check: Avoid matching things like "et al." if patterns are too broad
                        if len(core_header.split()) > 4 or len(core_header) < 3:
                            continue

                        self.logger.debug(
                            f"Regex matched line: '{line_stripped}', Start={start_index}, Header='{matched_header}', Core='{core_header}', Key='{pattern_key}'"
                        )
                        section_matches.append(
                            (start_index, matched_header, pattern_key)
                        )
                        # Break after first match for a line to avoid multiple patterns matching same header
                        break
                except Exception as e:
                    self.logger.error(
                        f"Regex error for pattern {pattern_key} on line '{line_stripped}': {e}"
                    )
                    continue  # Skip problematic patterns

            current_char_offset += len(line) + 1  # Move offset to next line start

        # Sort matches by their starting position
        section_matches.sort(key=lambda x: x[0])

        # Create PaperSection objects from matches
        num_matches = len(section_matches)
        processed_content_end = 0
        for i in range(num_matches):
            start_index, header, pattern_key = section_matches[i]
            section_type = self.SECTION_TYPE_MAP.get(pattern_key, "unknown")

            if start_index < processed_content_end or section_type == "unknown":
                continue

            # Find the actual start of content (first non-empty line after header)
            content_start_index = -1
            temp_index = start_index + len(
                header
            )  # Start searching after the matched header text
            while temp_index < len(full_text):
                line_end = full_text.find("\n", temp_index)
                if line_end == -1:
                    line_end = len(full_text)
                line_content = full_text[temp_index:line_end].strip()
                if line_content:  # Found the first non-empty line
                    content_start_index = temp_index + full_text[temp_index:].find(
                        line_content
                    )
                    break
                temp_index = line_end + 1  # Move to the start of the next line
                if temp_index >= len(full_text):
                    break  # Reached end of text

            if content_start_index == -1:  # No content found after header
                self.logger.debug(
                    f"Skipping section '{header}' as no content found after header line."
                )
                continue

            # Find the start of the *next* valid section header
            next_section_start_index = len(full_text)
            for j in range(i + 1, num_matches):
                next_start, _, _ = section_matches[j]
                if (
                    next_start >= content_start_index
                ):  # Ensure next header starts after current content begins
                    next_section_start_index = next_start
                    break

            content_end_index = next_section_start_index
            content = full_text[content_start_index:content_end_index].strip()

            # Clean content: remove page break markers
            content = re.sub(r"\n\n<PAGEBREAK NUM=\d+>\n\n", "\n\n", content).strip()

            if not content:
                self.logger.debug(
                    f"Skipping section '{header}' as it has no content after cleaning."
                )
                continue

            page_numbers = self._estimate_page_numbers(
                start_index, content_end_index, pages_data
            )

            section_obj = PaperSection(
                title=header,
                content=content,
                section_type=section_type,
                page_numbers=page_numbers,
                confidence=0.6,
            )
            if section_type not in sections:
                sections[section_type] = section_obj
                processed_content_end = content_end_index
                self.logger.info(
                    f"Identified Section: '{section_type}' (Title: '{header}') Pages: {page_numbers}"
                )
            else:
                self.logger.warning(
                    f"Duplicate section type '{section_type}' (Header: '{header}'). Appending content."
                )
                sections[section_type].content += "\n\n" + content
                sections[section_type].page_numbers = sorted(
                    list(set(sections[section_type].page_numbers + page_numbers))
                )
                processed_content_end = content_end_index

        # Special handling for abstract if not found by regex
        if "abstract" not in sections:
            self.logger.debug("Attempting abstract heuristic extraction...")
            abstract_text, abstract_page = self._extract_abstract_heuristic(
                full_text, pages_data
            )
            if abstract_text:
                self.logger.info("Found abstract using heuristics.")
                sections["abstract"] = PaperSection(
                    title="Abstract",
                    content=abstract_text,
                    section_type="abstract",
                    page_numbers=[abstract_page] if abstract_page else [1],
                    confidence=0.7,
                )
            else:
                self.logger.warning("Abstract not found by regex or heuristics.")

        # Refine with ML classification if model is available
        if self.section_model and self.section_tokenizer:
            sections = self._ml_refine_sections(sections)
        else:
            self.logger.warning(
                "Section classification model not loaded. Skipping ML refinement."
            )

        self.logger.info(f"Final identified sections: {list(sections.keys())}")
        return sections

    def _estimate_page_numbers(
        self, start_char: int, end_char: int, pages_data: List[Dict]
    ) -> List[int]:
        """Estimate page numbers based on character offsets"""
        pages = set()
        current_char_count = 0
        marker_len_estimate = len("\n\n<PAGEBREAK NUM=10>\n\n")

        for page_info in pages_data:
            page_content_start_char = (
                current_char_count + marker_len_estimate
                if current_char_count > 0
                else 0
            )
            page_content_end_char = page_content_start_char + len(page_info["text"])

            if max(page_content_start_char, start_char) < min(
                page_content_end_char, end_char
            ):
                pages.add(page_info["page_num"])

            current_char_count = page_content_end_char

        if not pages and pages_data:
            temp_char_count = 0
            for page_info in pages_data:
                page_start = (
                    temp_char_count + marker_len_estimate if temp_char_count > 0 else 0
                )
                page_end = page_start + len(page_info["text"])
                if page_start <= start_char < page_end:
                    pages.add(page_info["page_num"])
                    break
                temp_char_count = page_end

        return sorted(list(pages)) if pages else []

    def _extract_abstract_heuristic(
        self, full_text: str, pages_data: List[Dict]
    ) -> Tuple[Optional[str], Optional[int]]:
        """Try to extract abstract using heuristics if regex fails"""
        try:
            text_to_search = pages_data[0]["text"] if pages_data else full_text[:3000]
            # Look for line starting exactly with 'Abstract' (case insensitive)
            abstract_match = re.search(
                r"(?im)^\s*Abstract\s*$(.*?)(?=\n\s*(?:(?:\d+\.?\s*)?(?:introduction|background|keywords|index\s+terms)|i\.?\s+introduction)|^\s*$)",
                text_to_search,
                re.S,
            )  # Use re.S for DOTALL

            if abstract_match:
                # Find the start of the actual content after the "Abstract" line
                header_end = abstract_match.start(
                    1
                )  # End of the header match group is start of content group
                abstract_content = abstract_match.group(1).strip()

                if 50 < len(abstract_content.split()) < 500:
                    page_num = 1 if pages_data else None
                    return abstract_content, page_num
                else:
                    self.logger.debug(
                        f"Heuristic abstract candidate rejected due to length: {len(abstract_content.split())} words"
                    )

        except Exception as e:
            self.logger.error(f"Error during abstract heuristic extraction: {e}")

        return None, None

    def _ml_refine_sections(
        self, sections: Dict[str, PaperSection]
    ) -> Dict[str, PaperSection]:
        """Use ML model to refine section classification and confidence"""
        self.logger.info("Refining section classification with ML model...")
        refined_sections_temp = {}
        original_keys = list(sections.keys())
        sections_copy = sections.copy()
        model_labels = self.CORE_SECTION_TYPES

        for section_key in original_keys:
            if section_key not in sections_copy:
                continue
            section = sections_copy[section_key]

            if section_key not in self.CORE_SECTION_TYPES or not self.section_model:
                refined_sections_temp[section_key] = section
                continue

            try:
                text_sample = section.content[:1024]
                inputs = self.section_tokenizer(
                    text_sample,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.section_tokenizer.model_max_length,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.section_model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                predicted_idx = np.argmax(probabilities)
                if predicted_idx < len(model_labels):
                    predicted_type = model_labels[predicted_idx]
                    confidence = float(probabilities[predicted_idx])
                else:
                    self.logger.error(
                        f"ML model predicted invalid index {predicted_idx} for section '{section.title}'. Skipping."
                    )
                    refined_sections_temp[section_key] = section
                    continue

                self.logger.debug(
                    f"Section '{section.title}' (orig: {section_key}): Predicted '{predicted_type}' (Conf: {confidence:.2f})"
                )
                current_section_object = section
                final_key = section_key

                if predicted_type != section.section_type and confidence > 0.75:
                    self.logger.info(
                        f"Relabeling section '{section.title}' from '{section.section_type}' to '{predicted_type}' (Conf: {confidence:.2f})"
                    )
                    current_section_object.section_type = predicted_type
                    current_section_object.confidence = confidence
                    final_key = predicted_type
                elif predicted_type == section.section_type:
                    current_section_object.confidence = max(
                        section.confidence, confidence
                    )
                else:
                    pass

                if final_key in refined_sections_temp:
                    existing_confidence = refined_sections_temp[final_key].confidence
                    if current_section_object.confidence > existing_confidence:
                        self.logger.warning(
                            f"Collision for '{final_key}'. Replacing entry due to higher confidence."
                        )
                        if final_key != section_key and section_key in sections_copy:
                            del sections_copy[section_key]
                        refined_sections_temp[final_key] = current_section_object
                    else:
                        self.logger.warning(
                            f"Collision for '{final_key}'. Keeping existing entry due to higher confidence."
                        )
                        if final_key != section_key and section_key in sections_copy:
                            del sections_copy[section_key]
                else:
                    refined_sections_temp[final_key] = current_section_object
                    if final_key != section_key and section_key in sections_copy:
                        del sections_copy[section_key]

            except Exception as e:
                self.logger.error(
                    f"ML classification error for section '{section.title}': {e}"
                )
                if section_key not in refined_sections_temp:
                    refined_sections_temp[section_key] = section

        final_refined_sections = {}
        processed_final_keys = set()
        original_section_objects_after_collision = sections_copy.values()

        for key in original_keys:
            if (
                key in refined_sections_temp
                and refined_sections_temp[key]
                in original_section_objects_after_collision
            ):
                if key not in processed_final_keys:
                    final_refined_sections[key] = refined_sections_temp[key]
                    processed_final_keys.add(key)
            elif key in sections_copy:
                original_section = sections_copy[key]
                relabeled_key = original_section.section_type
                if (
                    relabeled_key != key
                    and relabeled_key in refined_sections_temp
                    and relabeled_key not in processed_final_keys
                ):
                    final_refined_sections[relabeled_key] = refined_sections_temp[
                        relabeled_key
                    ]
                    processed_final_keys.add(relabeled_key)

        for key, section_obj in refined_sections_temp.items():
            if key not in processed_final_keys:
                final_refined_sections[key] = section_obj

        return final_refined_sections

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
        if not corpus_texts:
            self.logger.warning(
                "No substantial text found in core sections for keyword extraction."
            )
            return [], []
        try:
            if hasattr(self.tfidf, "vocabulary_") and self.tfidf.vocabulary_:
                tfidf_matrix = self.tfidf.transform(corpus_texts)
            else:
                tfidf_matrix = self.tfidf.fit_transform(corpus_texts)
            feature_names = self.tfidf.get_feature_names_out()
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
                self.logger.warning("TF-IDF vocabulary is empty.")
            else:
                self.logger.error(f"TF-IDF ValueError: {ve}")
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

    def generate_summaries(
        self, sections: Dict[str, PaperSection]
    ) -> Dict[str, Optional[str]]:
        """Generate summaries for key sections, the full paper, and significance"""
        self.logger.info("Generating summaries...")
        summaries: Dict[str, Optional[str]] = {
            f"{st}_summary": None for st in self.CORE_SECTION_TYPES
        }
        summaries["full"] = None
        summaries["significance"] = None
        if "discussion" in sections:
            summaries["discussion_summary"] = None
        if not self.summarizer_model or not self.summarizer_tokenizer:
            self.logger.error("Summarizer model not loaded.")
            return summaries
        sections_to_summarize = self.CORE_SECTION_TYPES + (
            ["discussion"] if "discussion" in sections else []
        )
        for section_type in sections_to_summarize:
            if section_type in sections:
                self.logger.debug(f"Summarizing section: {section_type}")
                try:
                    if section_type == "abstract":
                        max_len, min_len = 100, 30
                    elif section_type == "methods":
                        max_len, min_len = 200, 50
                    else:
                        max_len, min_len = 150, 40
                    summary = self.summarize_text(
                        sections[section_type].content,
                        max_length=max_len,
                        min_length=min_len,
                    )
                    summaries[f"{section_type}_summary"] = summary
                except Exception as e:
                    self.logger.error(f"Error summarizing section {section_type}: {e}")
                    try:
                        sentences = sent_tokenize(sections[section_type].content)
                        summaries[f"{section_type}_summary"] = (
                            " ".join(sentences[:3]) + "..."
                        )
                    except Exception:
                        summaries[f"{section_type}_summary"] = (
                            sections[section_type].content[:250] + "..."
                        )
        self.logger.debug("Generating structured full paper summary...")
        try:
            full_summary = self.generate_structured_full_summary(summaries)
            summaries["full"] = full_summary
        except Exception as e:
            self.logger.error(f"Error generating structured full paper summary: {e}")
            summaries["full"] = "Could not generate structured full summary."
        self.logger.debug("Generating significance statement...")
        try:
            significance = self.generate_significance(sections)
            summaries["significance"] = significance
        except Exception as e:
            self.logger.error(f"Error generating significance statement: {e}")
            summaries["significance"] = "Could not generate significance statement."
        self.logger.info("Summarization complete.")
        return summaries

    def summarize_text(
        self, text: str, max_length: int = 150, min_length: int = 40
    ) -> str:
        """Summarize text using the loaded model, handling long inputs by chunking."""
        if not self.summarizer_model or not self.summarizer_tokenizer:
            raise ValueError("Summarizer model/tokenizer not loaded.")
        text = re.sub(r"\s+", " ", text).strip()
        word_count = len(text.split())
        if not text or word_count < min(15, min_length / 2):
            self.logger.debug(f"Text too short ({word_count} words).")
            return text
        try:
            inputs = self.summarizer_tokenizer(
                text, return_tensors="pt", truncation=False
            )
            total_tokens = inputs.input_ids.shape[1]
        except Exception as e:
            self.logger.error(f"Tokenization error: {e}.")
            return text[: max_length * 5] + "..."
        if total_tokens <= self.summarizer_max_input_length:
            self.logger.debug(f"Direct summarization ({total_tokens} tokens).")
            try:
                input_ids = inputs.input_ids.to(self.device)
                summary_ids = self.summarizer_model.generate(
                    input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
                summary = self.summarizer_tokenizer.decode(
                    summary_ids[0], skip_special_tokens=True
                )
                return summary.strip()
            except Exception as e:
                self.logger.error(f"Direct summarization error: {e}. Falling back.")
                try:
                    truncated_inputs = self.summarizer_tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.summarizer_max_input_length,
                    )
                    input_ids = truncated_inputs.input_ids.to(self.device)
                    summary_ids = self.summarizer_model.generate(
                        input_ids,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                    )
                    summary = self.summarizer_tokenizer.decode(
                        summary_ids[0], skip_special_tokens=True
                    )
                    return summary.strip()
                except Exception as e_inner:
                    self.logger.error(f"Fallback summarization error: {e_inner}")
                    return text[: max_length * 5] + "..."
        self.logger.info(f"Chunking required ({total_tokens} tokens)...")
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            self.logger.error(
                f"Sentence tokenization failed: {e}. Splitting by paragraph."
            )
            sentences = text.split("\n\n")
        chunks = []
        current_chunk_sentences = []
        current_chunk_token_count = 0
        max_chunk_tokens = self.summarizer_max_input_length - 50
        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                sentence_tokens = self.summarizer_tokenizer.tokenize(sentence)
                sentence_token_count = len(sentence_tokens)
            except Exception as e:
                self.logger.warning(
                    f"Cannot tokenize sentence: '{sentence[:50]}...'. Skipping."
                )
                continue
            if sentence_token_count > max_chunk_tokens:
                self.logger.warning(
                    f"Truncating long sentence ({sentence_token_count} > {max_chunk_tokens})."
                )
                truncated_sentence = self.summarizer_tokenizer.convert_tokens_to_string(
                    sentence_tokens[:max_chunk_tokens]
                )
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                chunks.append(truncated_sentence)
                current_chunk_sentences = []
                current_chunk_token_count = 0
                continue
            if current_chunk_token_count + sentence_token_count <= max_chunk_tokens:
                current_chunk_sentences.append(sentence)
                current_chunk_token_count += sentence_token_count
            else:
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_token_count = sentence_token_count
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        if not chunks:
            self.logger.error("Chunking failed.")
            return text[: max_length * 5] + "..."
        self.logger.info(f"Split into {len(chunks)} chunks.")
        chunk_summaries = []
        num_chunks = len(chunks)
        target_chunk_summary_len = max(15, min_length // num_chunks)
        max_chunk_summary_len = max(30, max_length // num_chunks)
        max_chunk_summary_len = min(
            max_chunk_summary_len, max_length // 2 if num_chunks > 1 else max_length
        )
        for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks", leave=False)):
            try:
                chunk_summary = self.summarize_text(
                    chunk,
                    max_length=max_chunk_summary_len,
                    min_length=target_chunk_summary_len,
                )
                chunk_summaries.append(chunk_summary)
            except Exception as e:
                self.logger.error(f"Error summarizing chunk {i+1}: {e}")
                chunk_summaries.append(chunk[: max_chunk_summary_len * 5] + "...")
        combined_summary_text = "\n".join(chunk_summaries)
        if num_chunks > 1:
            self.logger.info("Hierarchical summarization...")
            try:
                final_summary = self.summarize_text(
                    combined_summary_text, max_length=max_length, min_length=min_length
                )
                return final_summary
            except Exception as e:
                self.logger.error(f"Hierarchical summarization error: {e}")
                return " ".join(chunk_summaries)  # Fallback
        else:
            return combined_summary_text

    def generate_structured_full_summary(
        self, section_summaries: Dict[str, Optional[str]]
    ) -> str:
        """Generate a structured 7-10 sentence summary from section summaries."""
        self.logger.info("Generating structured full summary...")
        summary_order = [
            "abstract",
            "introduction",
            "methods",
            "results",
            "discussion",
            "conclusion",
        ]
        available_summaries = [
            f"{st.capitalize()}: {section_summaries.get(f'{st}_summary', '')}"
            for st in summary_order
            if section_summaries.get(f"{st}_summary")
        ]
        if not available_summaries:
            self.logger.warning("No section summaries available.")
            return "Could not generate structured summary: Missing section summaries."
        text_to_summarize = "\n\n".join(available_summaries)
        target_min_length = 80
        target_max_length = 200
        self.logger.debug(
            f"Combined text for final summary (first 500 chars):\n{text_to_summarize[:500]}..."
        )
        try:
            structured_summary = self.summarize_text(
                text_to_summarize,
                max_length=target_max_length,
                min_length=target_min_length,
            )
            final_sentences = sent_tokenize(structured_summary)
            num_sentences = len(final_sentences)
            if num_sentences < 7:
                self.logger.warning(
                    f"Generated summary has {num_sentences} sentences (target 7-10)."
                )
            elif num_sentences > 10:
                self.logger.warning(
                    f"Generated summary has {num_sentences} sentences. Truncating to 10."
                )
                structured_summary = " ".join(final_sentences[:10])
            return structured_summary
        except Exception as e:
            self.logger.error(f"Error generating structured summary: {e}")
            fallback_summary = " ".join(available_summaries[:3])
            return fallback_summary[: target_max_length * 5] + "..."

    def generate_significance(self, sections: Dict[str, PaperSection]) -> str:
        """Generate a statement about the paper's significance."""
        self.logger.info("Generating significance statement...")
        text_for_significance = ""
        if "abstract" in sections and sections["abstract"].content:
            text_for_significance += (
                "Abstract:\n" + sections["abstract"].content + "\n\n"
            )
        if "introduction" in sections and sections["introduction"].content:
            try:
                intro_sentences = sent_tokenize(sections["introduction"].content)
                intro_snippet = (
                    " ".join(intro_sentences[:3] + intro_sentences[-2:])
                    if len(intro_sentences) > 5
                    else " ".join(intro_sentences)
                )
                text_for_significance += (
                    "Introduction Highlights:\n" + intro_snippet + "\n\n"
                )
            except Exception as e:
                self.logger.warning(f"Could not tokenize intro: {e}. Using full intro.")
                text_for_significance += (
                    "Introduction:\n" + sections["introduction"].content + "\n\n"
                )
        if "conclusion" in sections and sections["conclusion"].content:
            text_for_significance += (
                "Conclusion:\n" + sections["conclusion"].content + "\n\n"
            )
        if not text_for_significance.strip():
            self.logger.warning("Missing key sections for significance.")
            return "Could not determine significance: Missing key sections."
        try:
            significance_summary = self.summarize_text(
                text_for_significance, max_length=120, min_length=30
            )
            return significance_summary
        except Exception as e:
            self.logger.error(f"Error generating significance summary: {e}")
            return "Error occurred during significance generation."

    def extract_metadata(
        self, full_text: str, pages_data: List[Dict], sections: Dict[str, PaperSection]
    ) -> Dict[str, Any]:
        """Extract metadata like title, authors, year, DOI using heuristics."""
        self.logger.info("Extracting metadata...")
        metadata = {"title": None, "authors": [], "year": None, "doi": None}
        if not full_text or not pages_data:
            return metadata
        first_page_text = pages_data[0]["text"]
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
                ):
                    break
                author_pattern = r"[A-Z][a-zA-Z\'\-\.]+(?:\s+[A-Z][a-zA-Z\'\-\.]+)*"
                temp_line_authors = re.split(r"\s*,\s*|\s+and\s+", line)
                cleaned_authors = []
                for potential_author in temp_line_authors:
                    potential_author = potential_author.strip()
                    if re.fullmatch(author_pattern, potential_author):
                        if (
                            len(potential_author.split()) < 5
                            and len(potential_author) > 1
                            and not re.search(
                                r"(?i)university|institute|department|inc|llc|email",
                                potential_author,
                            )
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
                full_text,
            )
            if not year_match:
                arxiv_match = re.search(
                    r"arXiv:.*\[.*\]\s+\d{1,2}\s+[A-Za-z]{3,}\s+(\d{4})", full_text
                )
            if arxiv_match:
                year_match = arxiv_match
            if not year_match and pages_data:
                possible_years = re.findall(
                    r"\b(19[89]\d|20[0-2]\d)\b",
                    pages_data[0]["text"] + "\n" + pages_data[-1]["text"],
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
                full_text,
                re.IGNORECASE,
            )
            if not doi_match:
                doi_match = re.search(
                    r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", full_text, re.IGNORECASE
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

    def process_paper(self, pdf_path: str) -> Optional[PaperAnalysis]:
        """Process a paper: extract text, metadata, sections, concepts, summaries."""
        self.logger.info(f"Starting processing for paper: {pdf_path}")
        # 1. Extract text
        full_text, pages_data = self.extract_text_from_pdf(pdf_path)
        if not full_text:
            return None
        # 2. Identify sections
        sections = self.identify_sections(full_text, pages_data)
        # 3. Extract metadata
        metadata = self.extract_metadata(full_text, pages_data, sections)
        # 4. Extract key concepts and keywords
        key_concepts, keywords = self.extract_key_concepts(sections)
        # 5. Generate summaries
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


def main():
    """Main function to run the paper processor from command line"""
    parser = argparse.ArgumentParser(
        description="Process academic papers for analysis and summarization"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: alongside PDF with _analysis.json suffix)",
        default=None,
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)  # Re-get logger instance after config

    if not os.path.exists(args.pdf_path):
        logger.error(f"Input PDF not found: {args.pdf_path}")
        return
    if not args.output:
        output_dir = os.path.dirname(args.pdf_path) or "."
        base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        args.output = os.path.join(output_dir, f"{base_name}_analysis.json")
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

    try:
        processor = PaperProcessor(use_gpu=not args.no_gpu)
        logger.info(f"Processing paper: {args.pdf_path}")
        analysis = processor.process_paper(args.pdf_path)
        if analysis:
            logger.info(f"Attempting to save analysis to: {args.output}")
            analysis.save_to_file(args.output)
            print(f"\nAnalysis complete. Results saved to: {args.output}")
            print("\n=== Paper Analysis Quick Look ===")
            print(f"Title: {analysis.title or 'N/A'}")
            print(
                f"Authors: {', '.join(analysis.authors) if analysis.authors else 'N/A'}"
            )
            print(f"Year: {analysis.publication_year or 'N/A'}")
            print(f"DOI: {analysis.doi or 'N/A'}")
            if analysis.abstract_summary:
                print(f"\nAbstract Summary:\n{analysis.abstract_summary}")
            elif "abstract" in analysis.sections:
                print(
                    f"\nAbstract (Original - First 300 chars):\n{analysis.sections['abstract'].content[:300]}..."
                )
            if analysis.full_summary:
                print(
                    f"\nStructured Full Summary (7-10 sentences):\n{analysis.full_summary}"
                )
            if analysis.significance:
                print(f"\nSignificance Statement:\n{analysis.significance}")
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
        else:
            logger.error(f"ImportError: {ie}.")
        print(f"Error: Missing required library. Details: {ie}")
    except Exception as e:
        logger.exception("Unexpected error in main workflow.")
        print(f"An error occurred: {e}.")


if __name__ == "__main__":
    main()
