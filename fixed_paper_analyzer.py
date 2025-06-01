# Scientific Paper Analysis Tool - FIXED VERSION
# Main fixes: Better section detection, improved figure filtering, fallback text extraction

import os
import re
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any, TypedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import datetime
import statistics

# PDF Extraction
import pdfplumber
import fitz  # PyMuPDF

# OpenAI Integration
try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None
    OpenAIError = None
    logging.warning("OpenAI library not found. pip install openai")

# .env File Loading
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    logging.warning("python-dotenv library not found. pip install python-dotenv")

# NLP and ML
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Keep existing dataclasses...
@dataclass
class PaperSection:
    """A section of an academic paper with relevant metadata"""
    title: str
    content: str
    section_type: str
    page_numbers: List[int]
    confidence: float = 0.0
    start_block_idx: Optional[int] = None
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
    figures: List[Dict] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    figure_extraction_method: str = ""
    total_figures_extracted: int = 0
    total_tables_extracted: int = 0

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

class ImprovedPaperProcessor:
    """Improved processor with better section detection and figure filtering."""

    # IMPROVED SECTION PATTERNS - More flexible and comprehensive
    SECTION_PATTERNS = {
        "abstract": [
            r"^\s*abstract\s*$",
            r"^\s*summary\s*$",
        ],
        "introduction": [
            r"^\s*(?:\d+\.?\s+)?introduction\s*$",
            r"^\s*(?:\d+\.?\s+)?background\s*$",
            r"^\s*(?:\d+\.?\s+)?overview\s*$",
            r"^\s*1\.?\s*introduction\s*$",
        ],
        "related_work": [
            r"^\s*(?:\d+\.?\s+)?related\s+work\s*$",
            r"^\s*(?:\d+\.?\s+)?literature\s+review\s*$",
            r"^\s*(?:\d+\.?\s+)?prior\s+art\s*$",
            r"^\s*(?:\d+\.?\s+)?background\s*$",
        ],
        "methods": [
            r"^\s*(?:\d+\.?\s+)?methods?\s*$",
            r"^\s*(?:\d+\.?\s+)?methodology\s*$",
            r"^\s*(?:\d+\.?\s+)?approach\s*$",
            r"^\s*(?:\d+\.?\s+)?model\s*$",
            r"^\s*(?:\d+\.?\s+)?implementation\s*$",
            r"^\s*(?:\d+\.?\s+)?experimental\s+setup\s*$",
        ],
        "results": [
            r"^\s*(?:\d+\.?\s+)?results?\s*$",
            r"^\s*(?:\d+\.?\s+)?experiments?\s*$",
            r"^\s*(?:\d+\.?\s+)?evaluation\s*$",
            r"^\s*(?:\d+\.?\s+)?findings?\s*$",
            r"^\s*(?:\d+\.?\s+)?performance\s*$",
        ],
        "discussion": [
            r"^\s*(?:\d+\.?\s+)?discussion\s*$",
            r"^\s*(?:\d+\.?\s+)?analysis\s*$",
            r"^\s*(?:\d+\.?\s+)?interpretation\s*$",
        ],
        "conclusion": [
            r"^\s*(?:\d+\.?\s+)?conclusion\s*$",
            r"^\s*(?:\d+\.?\s+)?conclusions\s*$",
            r"^\s*(?:\d+\.?\s+)?summary\s*$",
            r"^\s*(?:\d+\.?\s+)?future\s+work\s*$",
        ],
        "references": [
            r"^\s*references?\s*$",
            r"^\s*bibliography\s*$",
        ]
    }

    SECTION_TYPE_MAP = {
        "abstract": "abstract",
        "introduction": "introduction", 
        "related_work": "introduction",
        "methods": "methods",
        "results": "results", 
        "discussion": "discussion",
        "conclusion": "conclusion",
        "references": "references"
    }

    CORE_SECTION_TYPES = [
        "abstract", "introduction", "methods", "results", "discussion", "conclusion"
    ]

    def __init__(self, use_gpu: bool = True, openai_api_key: Optional[str] = None, openai_model: str = "gpt-3.5-turbo"):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_client = None

        # Initialize OpenAI client
        if self.openai_api_key and OpenAI:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.logger.info(f"OpenAI client initialized for model: {self.openai_model}")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None

        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.logger.warning("NLTK stopwords not found. Downloading...")
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))

        self.sentence_vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
        self.load_models()

    def load_models(self):
        """Load required models"""
        self.logger.info("Loading NLP models...")
        
        # Keyword extraction TFIDF
        try:
            self.keyword_tfidf = TfidfVectorizer(
                max_features=100, stop_words=list(self.stop_words), ngram_range=(1, 2)
            )
            self.logger.info("TF-IDF model loaded for keywords.")
        except Exception as e:
            self.logger.error(f"Error initializing keyword TF-IDF: {e}")
            self.keyword_tfidf = None

    def extract_text_with_fallback(self, pdf_path: str) -> Tuple[Optional[str], List[Dict]]:
        """Extract text using multiple methods as fallback"""
        self.logger.info(f"Extracting text from {pdf_path}")
        
        full_plain_text = ""
        pages_plain_text_data = []
        
        # Method 1: Try PyMuPDF first (better layout preservation)
        try:
            doc = fitz.open(pdf_path)
            self.logger.info(f"PDF has {len(doc)} pages.")
            
            for page_num, page in enumerate(doc):
                try:
                    # Get text with layout preservation
                    page_text = page.get_text("text", sort=True)
                    if not page_text.strip():
                        # Fallback: try different extraction method
                        page_text = page.get_text("dict")
                        if isinstance(page_text, dict) and "blocks" in page_text:
                            text_parts = []
                            for block in page_text["blocks"]:
                                if "lines" in block:
                                    for line in block["lines"]:
                                        for span in line["spans"]:
                                            text_parts.append(span.get("text", ""))
                            page_text = " ".join(text_parts)
                    
                    # Clean up text
                    page_text = self._clean_text(page_text)
                    
                    page_marker = f"\n\n<PAGEBREAK NUM={page_num + 1}>\n\n"
                    full_plain_text += page_marker + page_text
                    pages_plain_text_data.append({
                        "page_num": page_num + 1, 
                        "text": page_text
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error extracting text from page {page_num + 1}: {e}")
                    pages_plain_text_data.append({
                        "page_num": page_num + 1, 
                        "text": ""
                    })
                    
            doc.close()
            
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {e}")
            
            # Method 2: Fallback to pdfplumber
            try:
                self.logger.info("Trying fallback extraction with pdfplumber...")
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text() or ""
                            page_text = self._clean_text(page_text)
                            
                            page_marker = f"\n\n<PAGEBREAK NUM={page_num + 1}>\n\n"
                            full_plain_text += page_marker + page_text
                            pages_plain_text_data.append({
                                "page_num": page_num + 1,
                                "text": page_text
                            })
                        except Exception as e:
                            self.logger.error(f"Error with pdfplumber on page {page_num + 1}: {e}")
                            pages_plain_text_data.append({
                                "page_num": page_num + 1,
                                "text": ""
                            })
                            
            except Exception as e:
                self.logger.error(f"pdfplumber extraction also failed: {e}")
                return None, []
        
        if not full_plain_text.strip():
            self.logger.error(f"Could not extract any text from {pdf_path}")
            return None, []
            
        self.logger.info(f"Successfully extracted text from {len(pages_plain_text_data)} pages")
        return full_plain_text, pages_plain_text_data

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
            
        # Replace ligatures
        text = (
            text.replace("\ufb00", "ff")
            .replace("\ufb01", "fi") 
            .replace("\ufb02", "fl")
            .replace("\ufb03", "ffi")
            .replace("\ufb04", "ffl")
        )
        
        # Normalize whitespace
        text = re.sub(r"(\r\n|\r|\n){2,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        
        return text

    def identify_sections_improved(self, full_plain_text: str, pages_plain_text_data: List[Dict]) -> Dict[str, PaperSection]:
        """Improved section identification using multiple strategies"""
        self.logger.info("Identifying sections using improved text-based analysis...")
        
        if not full_plain_text:
            self.logger.warning("No text available for section identification")
            return {}
        
        sections = {}
        
        # Strategy 1: Line-by-line analysis with section patterns
        lines = full_plain_text.split('\n')
        potential_headers = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean or len(line_clean) > 100:  # Skip very long lines
                continue
                
            # Check if line matches any section pattern
            for section_key, patterns in self.SECTION_PATTERNS.items():
                for pattern in patterns:
                    if re.match(pattern, line_clean, re.IGNORECASE):
                        potential_headers.append({
                            'line_index': i,
                            'text': line_clean,
                            'section_type': section_key,
                            'confidence': 0.8
                        })
                        self.logger.debug(f"Found potential header: '{line_clean}' -> {section_key}")
                        break
        
        # Strategy 2: Extract sections based on found headers
        if potential_headers:
            potential_headers.sort(key=lambda x: x['line_index'])
            
            for i, header in enumerate(potential_headers):
                section_type = self.SECTION_TYPE_MAP.get(header['section_type'], header['section_type'])
                
                # Determine content boundaries
                start_line = header['line_index'] + 1
                end_line = potential_headers[i + 1]['line_index'] if i + 1 < len(potential_headers) else len(lines)
                
                # Extract content
                content_lines = lines[start_line:end_line]
                content = '\n'.join(content_lines).strip()
                
                # Filter out very short content
                if len(content.split()) < 10:
                    continue
                
                # Estimate page numbers
                page_nums = self._estimate_page_numbers_from_content(content, full_plain_text)
                
                section = PaperSection(
                    title=header['text'],
                    content=content,
                    section_type=section_type,
                    page_numbers=page_nums,
                    confidence=header['confidence']
                )
                
                if section_type not in sections:
                    sections[section_type] = section
                    self.logger.info(f"Identified section: '{section_type}' with {len(content.split())} words")
                else:
                    # Append to existing section
                    sections[section_type].content += "\n\n" + content
                    sections[section_type].page_numbers = sorted(list(set(
                        sections[section_type].page_numbers + page_nums
                    )))
        
        # Strategy 3: Abstract extraction fallback
        if "abstract" not in sections:
            abstract_text, abstract_page = self._extract_abstract_heuristic(full_plain_text, pages_plain_text_data)
            if abstract_text:
                sections["abstract"] = PaperSection(
                    title="Abstract",
                    content=abstract_text,
                    section_type="abstract",
                    page_numbers=[abstract_page] if abstract_page else [1],
                    confidence=0.6
                )
                self.logger.info("Extracted abstract using heuristics")
        
        # Strategy 4: Content-based section detection for missing sections
        if not sections:
            self.logger.warning("No sections found with pattern matching, trying content-based detection...")
            sections = self._detect_sections_by_content(full_plain_text, pages_plain_text_data)
        
        self.logger.info(f"Final identified sections: {list(sections.keys())}")
        return sections

    def _estimate_page_numbers_from_content(self, content: str, full_text: str) -> List[int]:
        """Estimate page numbers based on content position in full text"""
        pages = set()
        
        # Find content position in full text
        content_start = full_text.find(content[:min(100, len(content))])
        if content_start == -1:
            return [1]
        
        # Count page breaks before this position
        page_breaks_before = len(re.findall(r'<PAGEBREAK NUM=(\d+)>', full_text[:content_start]))
        current_page = max(1, page_breaks_before)
        
        # Count page breaks within content
        page_breaks_in_content = len(re.findall(r'<PAGEBREAK NUM=(\d+)>', content))
        
        # Add pages
        for i in range(max(1, page_breaks_in_content + 1)):
            pages.add(current_page + i)
        
        return sorted(list(pages))

    def _detect_sections_by_content(self, full_text: str, pages_data: List[Dict]) -> Dict[str, PaperSection]:
        """Fallback: detect sections based on content analysis"""
        sections = {}
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip() and len(p.strip()) > 50]
        
        # Try to identify abstract (usually early in document)
        for i, para in enumerate(paragraphs[:5]):  # Look in first 5 paragraphs
            if 50 < len(para.split()) < 300:  # Typical abstract length
                # Check if it looks like an abstract
                if any(word in para.lower() for word in ['present', 'propose', 'show', 'demonstrate', 'paper', 'work']):
                    sections['abstract'] = PaperSection(
                        title="Abstract",
                        content=para,
                        section_type="abstract",
                        page_numbers=[1],
                        confidence=0.5
                    )
                    break
        
        # Group remaining paragraphs as introduction
        if len(paragraphs) > 1:
            intro_content = '\n\n'.join(paragraphs[1:min(5, len(paragraphs))])
            if intro_content and len(intro_content.split()) > 100:
                sections['introduction'] = PaperSection(
                    title="Introduction",
                    content=intro_content,
                    section_type="introduction", 
                    page_numbers=[1, 2],
                    confidence=0.4
                )
        
        return sections

    def _extract_abstract_heuristic(self, text_content: str, pages_data: List[Dict]) -> Tuple[Optional[str], Optional[int]]:
        """Extract abstract using heuristics"""
        try:
            # Look in first page primarily
            search_text = pages_data[0]["text"] if pages_data else text_content[:3000]
            
            # Pattern 1: Standard abstract section
            abstract_match = re.search(
                r"(?im)^\s*abstract\s*\n+(.*?)(?=\n\s*\n|\n\s*(?:\d+\.?\s+)?(?:introduction|background|keywords|I\.|1\.)\b)",
                search_text,
                re.DOTALL
            )
            
            if abstract_match:
                abstract_content = abstract_match.group(1).strip()
                # Clean page break markers
                abstract_content = re.sub(r"<PAGEBREAK NUM=\d+>", "", abstract_content).strip()
                
                # Validate abstract length
                word_count = len(abstract_content.split())
                if 30 < word_count < 500:
                    return abstract_content, 1
                
            # Pattern 2: Look for abstract-like content early in document
            first_paragraphs = search_text.split('\n\n')[:5]
            for para in first_paragraphs:
                para = para.strip()
                word_count = len(para.split())
                if 30 < word_count < 400:
                    # Check if it contains abstract-like keywords
                    if any(keyword in para.lower() for keyword in [
                        'present', 'propose', 'paper', 'study', 'work', 'approach', 'method'
                    ]):
                        return para, 1
                        
        except Exception as e:
            self.logger.error(f"Abstract heuristic error: {e}")
        
        return None, None

    def extract_key_concepts(self, sections: Dict[str, PaperSection]) -> Tuple[List[KeyConcept], List[str]]:
        """Extract key concepts and keywords using TF-IDF"""
        self.logger.info("Extracting key concepts and keywords...")
        
        key_concepts = []
        keywords = []
        
        # Collect text from all sections
        corpus_texts = []
        section_map = []
        
        for section_type in self.CORE_SECTION_TYPES:
            if section_type in sections:
                # Split into meaningful chunks
                paragraphs = [
                    p.strip() for p in sections[section_type].content.split("\n\n")
                    if len(p.strip().split()) > 10
                ]
                corpus_texts.extend(paragraphs)
                section_map.extend([section_type] * len(paragraphs))
        
        if not corpus_texts or not self.keyword_tfidf:
            self.logger.warning("No substantial text or TFIDF model for keyword extraction")
            return [], []
        
        try:
            # Fit TF-IDF
            tfidf_matrix = self.keyword_tfidf.fit_transform(corpus_texts)
            feature_names = self.keyword_tfidf.get_feature_names_out()
            
            # Get term scores
            term_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            scored_terms = sorted(
                zip(feature_names, term_scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Extract keywords
            keywords = [
                term for term, score in scored_terms 
                if len(term) > 1 and not term.isdigit() and score > 0
            ][:15]
            
            self.logger.info(f"Extracted {len(keywords)} keywords")
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
        
        return key_concepts, keywords

    def generate_summaries(self, sections: Dict[str, PaperSection]) -> Dict[str, Optional[str]]:
        """Generate extractive summaries and synthesize with OpenAI"""
        self.logger.info("Generating summaries...")
        
        summaries = {f"{st}_summary": None for st in self.CORE_SECTION_TYPES}
        summaries.update({"full": None, "significance": None})
        
        # Extract key sentences from each section
        for section_type in self.CORE_SECTION_TYPES:
            if section_type in sections:
                try:
                    content = sections[section_type].content
                    if len(content.split()) > 20:  # Only process substantial content
                        key_sentences = self._extract_key_sentences(content, num_sentences=3)
                        summaries[f"{section_type}_summary"] = key_sentences
                        self.logger.debug(f"Extracted summary for {section_type}")
                except Exception as e:
                    self.logger.error(f"Error extracting summary for {section_type}: {e}")
        
        # Generate full summary using OpenAI
        if self.openai_client:
            try:
                # Combine key sections for full summary
                full_input = self._prepare_openai_input(summaries, ["abstract", "introduction", "results", "conclusion"])
                if full_input:
                    summaries["full"] = self._synthesize_with_openai(full_input, "full summary")
                
                # Generate significance
                sig_input = self._prepare_openai_input(summaries, ["abstract", "conclusion"])
                if sig_input:
                    summaries["significance"] = self._synthesize_with_openai(sig_input, "significance statement")
                    
            except Exception as e:
                self.logger.error(f"Error in OpenAI synthesis: {e}")
        
        # Fallback summaries
        if not summaries["full"]:
            summaries["full"] = self._generate_fallback_summary(summaries)
        if not summaries["significance"]:
            summaries["significance"] = self._generate_fallback_significance(summaries)
        
        return summaries

    def _extract_key_sentences(self, text: str, num_sentences: int = 3) -> str:
        """Extract key sentences using simple heuristics"""
        if not text:
            return ""
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return " ".join(sentences)
            
            # Simple scoring: prefer sentences with important words
            important_words = {'important', 'significant', 'novel', 'propose', 'demonstrate', 'show', 'results', 'conclude'}
            
            scored_sentences = []
            for i, sent in enumerate(sentences):
                score = 0
                words = sent.lower().split()
                
                # Score based on important words
                score += sum(1 for word in words if word in important_words)
                
                # Prefer sentences in first half of text
                if i < len(sentences) // 2:
                    score += 0.5
                
                # Prefer longer sentences (but not too long)
                if 10 < len(words) < 30:
                    score += 0.3
                
                scored_sentences.append((score, i, sent))
            
            # Get top sentences, maintain order
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            selected = sorted(scored_sentences[:num_sentences], key=lambda x: x[1])
            
            return " ".join(sent[2] for sent in selected)
            
        except Exception as e:
            self.logger.error(f"Error extracting key sentences: {e}")
            return text[:500] + "..." if len(text) > 500 else text

    def _prepare_openai_input(self, summaries: Dict[str, Optional[str]], sections: List[str]) -> Optional[str]:
        """Prepare input for OpenAI synthesis"""
        parts = []
        for section in sections:
            summary_key = f"{section}_summary"
            if summaries.get(summary_key):
                parts.append(f"{section.capitalize()}:\n{summaries[summary_key]}")
        
        return "\n\n---\n\n".join(parts) if parts else None

    def _synthesize_with_openai(self, input_text: str, context: str) -> Optional[str]:
        """Synthesize summary using OpenAI"""
        if not self.openai_client:
            return None
        
        prompt = f"""Synthesize the following extracted content from a scientific paper into a coherent {context}. 
Keep it concise but comprehensive, around 3-5 sentences.

Content:
---
{input_text}
---

Synthesized {context}:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in summarizing scientific content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI synthesis error: {e}")
            return None

    def _generate_fallback_summary(self, summaries: Dict[str, Optional[str]]) -> str:
        """Generate fallback summary by combining section summaries"""
        parts = []
        for section in ["abstract", "introduction", "results", "conclusion"]:
            summary_key = f"{section}_summary"
            if summaries.get(summary_key):
                parts.append(f"**{section.capitalize()}:** {summaries[summary_key]}")
        
        return "\n\n".join(parts) if parts else "Could not generate summary: insufficient content extracted."

    def _generate_fallback_significance(self, summaries: Dict[str, Optional[str]]) -> str:
        """Generate fallback significance statement"""
        significance_parts = []
        
        if summaries.get("abstract_summary"):
            significance_parts.append(summaries["abstract_summary"])
        if summaries.get("conclusion_summary"):
            significance_parts.append(summaries["conclusion_summary"])
        
        return " ".join(significance_parts) if significance_parts else "Could not determine significance: insufficient content extracted."

    def extract_metadata(self, full_plain_text: Optional[str], pages_plain_text_data: List[Dict]) -> Dict[str, Any]:
        """Extract metadata like title, authors, year, DOI"""
        self.logger.info("Extracting metadata from plain text...")
        metadata = {"title": None, "authors": [], "year": None, "doi": None}
        
        if not full_plain_text or not pages_plain_text_data:
            return metadata
        
        first_page_text = pages_plain_text_data[0]["text"]
        
        # Extract title - look for the actual paper title, not conference info
        try:
            lines = first_page_text.split("\n")
            title_candidates = []
            
            for i, line in enumerate(lines[:10]):  # Look in first 10 lines
                line = line.strip()
                if not line:
                    continue
                
                # Skip obvious non-title lines
                if any(skip_phrase in line.lower() for skip_phrase in [
                    "published as", "conference paper", "arxiv:", "doi:", "@", "university", 
                    "institute", "abstract", "keywords", "email"
                ]):
                    continue
                
                # Look for title-like lines
                if (5 < len(line.split()) < 20 and 
                    not line.isupper() and 
                    not re.match(r"^\d+", line) and
                    len(line) > 20):
                    title_candidates.append(line)
            
            # Take the first substantial title candidate
            if title_candidates:
                # Look for the actual paper title - often the longest meaningful line
                best_title = max(title_candidates, key=len)
                metadata["title"] = best_title
                self.logger.info(f"Extracted title: {best_title}")
            
        except Exception as e:
            self.logger.error(f"Error extracting title: {e}")
        
        # Extract authors - improved pattern
        try:
            author_patterns = [
                r"^([A-Z][a-zA-Z\s\-\.]+(?:\s+[A-Z][a-zA-Z\s\-\.]*)*)\s*$",  # Name pattern
                r"^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$"  # First Last pattern
            ]
            
            authors_found = set()
            for line in first_page_text.split('\n')[:15]:
                line = line.strip()
                if not line or len(line) < 5:
                    continue
                
                # Skip non-author lines
                if any(skip in line.lower() for skip in [
                    "university", "institute", "email", "@", "abstract", "published", "conference"
                ]):
                    continue
                
                for pattern in author_patterns:
                    if re.match(pattern, line):
                        # Split by common separators
                        potential_authors = re.split(r'[,&]|\s+and\s+', line)
                        for author in potential_authors:
                            author = author.strip()
                            if (len(author.split()) >= 2 and 
                                len(author.split()) <= 4 and
                                all(part[0].isupper() for part in author.split())):
                                authors_found.add(author)
            
            metadata["authors"] = list(authors_found)[:5]  # Limit to 5 authors
            if metadata["authors"]:
                self.logger.info(f"Extracted authors: {metadata['authors']}")
            
        except Exception as e:
            self.logger.error(f"Error extracting authors: {e}")
        
        # Extract year
        try:
            current_year = datetime.datetime.now().year
            year_patterns = [
                r"(\d{4})",  # Simple 4-digit year
                r"(?:published|accepted|submitted|copyright).*?(\d{4})",
                r"arxiv:.*?(\d{4})",
            ]
            
            years_found = []
            search_text = first_page_text + "\n" + (pages_plain_text_data[-1]["text"] if len(pages_plain_text_data) > 1 else "")
            
            for pattern in year_patterns:
                matches = re.findall(pattern, search_text, re.IGNORECASE)
                for match in matches:
                    year = int(match)
                    if 1990 <= year <= current_year + 1:
                        years_found.append(year)
            
            if years_found:
                # Take the most recent plausible year
                metadata["year"] = max(years_found)
                self.logger.info(f"Extracted year: {metadata['year']}")
            
        except Exception as e:
            self.logger.error(f"Error extracting year: {e}")
        
        # Extract DOI
        try:
            doi_patterns = [
                r"doi[:\s]+(10\.\d{4,}/[^\s]+)",
                r"(10\.\d{4,}/[^\s]+)",
                r"https?://(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s]+)"
            ]
            
            for pattern in doi_patterns:
                match = re.search(pattern, full_plain_text, re.IGNORECASE)
                if match:
                    doi = match.group(1) if match.lastindex else match.group(0)
                    metadata["doi"] = doi.strip().rstrip(".")
                    self.logger.info(f"Extracted DOI: {metadata['doi']}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error extracting DOI: {e}")
        
        return metadata

    def improved_figure_extraction(self, pdf_path: str, output_dir: str = None) -> Dict:
        """Improved figure extraction with better filtering"""
        self.logger.info("Starting improved figure extraction...")
        
        if output_dir is None:
            pdf_dir = os.path.dirname(pdf_path)
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join(pdf_dir, f"{pdf_name}_figures")
        
        try:
            # Use existing extraction but with better filtering
            from publaynet_figure_extractor import extract_figures_with_publaynet
            raw_results = extract_figures_with_publaynet(pdf_path, output_dir)
            
            # Apply improved filtering
            filtered_figures = self._filter_figures(raw_results.get("figures", []))
            filtered_tables = self._filter_tables(raw_results.get("tables", []))
            
            self.logger.info(f"Filtered figures: {len(raw_results.get('figures', []))} -> {len(filtered_figures)}")
            self.logger.info(f"Filtered tables: {len(raw_results.get('tables', []))} -> {len(filtered_tables)}")
            
            return {
                "success": True,
                "figures": filtered_figures,
                "tables": filtered_tables,
                "total_figures": len(filtered_figures),
                "total_tables": len(filtered_tables),
                "method": "improved_hybrid",
                "raw_count": len(raw_results.get("figures", [])),
                "output_dir": output_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error during figure extraction: {e}")
            return {
                "success": False,
                "figures": [],
                "tables": [],
                "total_figures": 0,
                "total_tables": 0,
                "method": "failed",
                "error": str(e)
            }

    def _filter_figures(self, figures: List[Dict]) -> List[Dict]:
        """Filter out likely false positive figures"""
        filtered = []
        
        for fig in figures:
            # Skip very small figures (likely artifacts)
            dims = fig.get("dimensions", {})
            width = dims.get("width", 0)
            height = dims.get("height", 0)
            
            if width < 200 or height < 150:
                continue
            
            # Skip figures with very low confidence from layout analysis
            if fig.get("method") == "layout_analysis" and fig.get("confidence", 0) < 0.6:
                continue
            
            # Skip figures that are likely text artifacts
            if fig.get("method") == "contour_detection" and width * height < 50000:
                continue
            
            # Prefer embedded images over detected regions
            if fig.get("method") == "embedded_image":
                fig["confidence"] = 0.9
            
            filtered.append(fig)
        
        # Remove duplicates based on similar positions
        final_filtered = []
        for fig in filtered:
            is_duplicate = False
            fig_bbox = fig.get("bbox", [0, 0, 0, 0])
            
            for existing in final_filtered:
                existing_bbox = existing.get("bbox", [0, 0, 0, 0])
                
                # Check if bboxes overlap significantly
                if self._bbox_overlap(fig_bbox, existing_bbox) > 0.5:
                    # Keep the one with higher confidence
                    if fig.get("confidence", 0) <= existing.get("confidence", 0):
                        is_duplicate = True
                        break
                    else:
                        # Replace existing with current
                        final_filtered.remove(existing)
                        break
            
            if not is_duplicate:
                final_filtered.append(fig)
        
        return final_filtered

    def _filter_tables(self, tables: List[Dict]) -> List[Dict]:
        """Filter tables (similar logic to figures)"""
        # For now, apply similar filtering as figures
        return self._filter_figures(tables)

    def _bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate bbox overlap ratio"""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def process_paper(self, pdf_path: str, extract_figures: bool = True, figure_output_dir: str = None) -> Optional[PaperAnalysis]:
        """Main processing pipeline - improved"""
        self.logger.info(f"Starting improved processing for paper: {pdf_path}")
        
        # 1. Extract text with fallback methods
        full_plain_text, pages_plain_text_data = self.extract_text_with_fallback(pdf_path)
        if not full_plain_text:
            self.logger.error("Failed to extract any text from PDF")
            return None
        
        # 2. Extract figures with improved filtering
        figure_results = {}
        if extract_figures:
            figure_results = self.improved_figure_extraction(pdf_path, figure_output_dir)
        
        # 3. Identify sections with improved method
        sections = self.identify_sections_improved(full_plain_text, pages_plain_text_data)
        
        # 4. Extract metadata
        metadata = self.extract_metadata(full_plain_text, pages_plain_text_data)
        
        # 5. Extract key concepts and keywords
        key_concepts, keywords = self.extract_key_concepts(sections)
        
        # 6. Generate summaries
        summaries = self.generate_summaries(sections)
        
        # 7. Create analysis object
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
            figures=figure_results.get("figures", []),
            tables=figure_results.get("tables", []),
            figure_extraction_method=figure_results.get("method", "none"),
            total_figures_extracted=figure_results.get("total_figures", 0),
            total_tables_extracted=figure_results.get("total_tables", 0),
        )
        
        self.logger.info(f"Finished processing paper: {pdf_path}")
        self.logger.info(f"Found {len(sections)} sections, {len(keywords)} keywords, {analysis.total_figures_extracted} figures")
        
        return analysis


def main():
    """Main function with improved processor"""
    parser = argparse.ArgumentParser(description="Improved Paper Analysis Tool")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--output", "-o", help="Output JSON file path", default=None)
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--extract-figures", action="store_true", default=True, help="Extract figures")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure extraction")
    parser.add_argument("--figure-output", help="Directory to save extracted figures")
    parser.add_argument("--openai-model", help="OpenAI model to use", default="gpt-3.5-turbo")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True
    )
    logger = logging.getLogger(__name__)
    
    # Load API Key
    openai_api_key = None
    if load_dotenv:
        env_path = Path(r"C:\LabGit\Scientific-paper-assistant-AI\api_keys\OPEN_AI_KEY.env").resolve()
        if env_path.is_file():
            logger.info(f"Loading OpenAI API key from: {env_path}")
            load_dotenv(dotenv_path=env_path)
            openai_api_key = os.getenv("OPENAI_API_KEY")
        else:
            logger.warning(f".env file not found at: {env_path}")
    
    # Check inputs
    if not os.path.exists(args.pdf_path):
        logger.error(f"Input PDF not found: {args.pdf_path}")
        return
    
    # Setup output path
    if not args.output:
        output_dir = os.path.dirname(args.pdf_path) or "."
        base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        args.output = os.path.join(output_dir, f"{base_name}_analysis_improved.json")
    
    try:
        # Create improved processor
        processor = ImprovedPaperProcessor(
            use_gpu=not args.no_gpu,
            openai_api_key=openai_api_key,
            openai_model=args.openai_model
        )
        
        logger.info(f"Processing paper: {args.pdf_path}")
        
        # Process paper
        analysis = processor.process_paper(
            args.pdf_path,
            extract_figures=args.extract_figures and not args.no_figures,
            figure_output_dir=args.figure_output
        )
        
        if analysis:
            # Save results
            analysis.save_to_file(args.output)
            print(f"\n✅ Analysis complete! Results saved to: {args.output}")
            
            # Print summary
            print(f"\n=== Improved Paper Analysis Summary ===")
            print(f"📄 Title: {analysis.title or 'N/A'}")
            print(f"👥 Authors: {', '.join(analysis.authors) if analysis.authors else 'N/A'}")
            print(f"📅 Year: {analysis.publication_year or 'N/A'}")
            print(f"🔗 DOI: {analysis.doi or 'N/A'}")
            
            print(f"\n📋 Sections Found: {list(analysis.sections.keys())}")
            
            if analysis.full_summary:
                print(f"\n📝 Summary:\n{analysis.full_summary}")
            
            if analysis.keywords:
                print(f"\n🔑 Keywords: {', '.join(analysis.keywords[:10])}")
            
            print(f"\n🖼️ Figures: {analysis.total_figures_extracted}")
            print(f"📊 Tables: {analysis.total_tables_extracted}")
            
        else:
            print("❌ Paper processing failed. Check logs for details.")
            
    except Exception as e:
        logger.exception("Unexpected error in main workflow")
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()