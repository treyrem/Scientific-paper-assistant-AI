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
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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

    def get_summary(self, summarizer, max_length: int = 150) -> str:
        """Generate a summary of the section using the provided summarizer"""
        if len(self.content.split()) < 50:  # If content is too short, don't summarize
            return self.content

        try:
            return summarizer(self.content, max_length=max_length)
        except Exception as e:
            logger.error(f"Error summarizing {self.section_type}: {e}")
            return self.content[:200] + "..."


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

    title: str
    authors: List[str]
    publication_year: Optional[int]
    doi: Optional[str]

    # Paper sections
    abstract: Optional[str] = None
    introduction: Optional[str] = None
    methods: Optional[str] = None
    results: Optional[str] = None
    discussion: Optional[str] = None
    conclusion: Optional[str] = None

    # Extracted data
    sections: Dict[str, PaperSection] = field(default_factory=dict)
    key_concepts: List[KeyConcept] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Generated content
    summary: str = ""
    significance: str = ""

    def to_json(self) -> str:
        """Convert the analysis to a JSON string"""
        return json.dumps(asdict(self), indent=2)

    def save_to_file(self, output_path: str) -> None:
        """Save the analysis to a JSON file"""
        with open(output_path, "w") as f:
            f.write(self.to_json())


class PaperProcessor:
    """Main class for processing academic papers"""

    # Section patterns to identify paper structure
    SECTION_PATTERNS = {
        "abstract": r"abstract",
        "introduction": r"introduction|background|overview",
        "methods": r"methods|methodology|materials\s+and\s+methods|experimental|approach",
        "results": r"results|findings|observations",
        "discussion": r"discussion|interpretation|implications",
        "conclusion": r"conclusion|summary|future\s+work|future\s+directions",
    }

    def __init__(
        self,
        use_gpu: bool = True,
        section_classifier_model: str = "allenai/scibert_scivocab_uncased",
        summarizer_model: str = "facebook/bart-large-cnn",
        embedding_model: str = "allenai/specter",
    ):
        """Initialize the paper processor with necessary models"""

        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        # Load models
        self.load_models(section_classifier_model, summarizer_model, embedding_model)

        # Initialize NLTK tools
        self.stop_words = set(stopwords.words("english"))

    def load_models(self, section_classifier_model, summarizer_model, embedding_model):
        """Load all required models"""
        self.logger.info("Loading NLP models...")

        # Section classifier
        self.section_tokenizer = AutoTokenizer.from_pretrained(section_classifier_model)
        self.section_model = AutoModelForSequenceClassification.from_pretrained(
            section_classifier_model, num_labels=6
        ).to(self.device)

        # Summarization model
        self.summarizer = pipeline(
            "summarization",
            model=summarizer_model,
            device=0 if self.device == "cuda" else -1,
        )

        # Embedding model for similarity and concept extraction
        self.embedding_model = SentenceTransformer(embedding_model).to(self.device)

        # Keyword extraction setup
        self.tfidf = TfidfVectorizer(
            max_features=100, stop_words="english", ngram_range=(1, 2)
        )

        self.logger.info("All models loaded successfully")

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Extract text and structure from a PDF file"""
        self.logger.info(f"Extracting text from {pdf_path}")

        full_text = ""
        pages_data = []

        try:
            # Using PyMuPDF for text extraction
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                full_text += f"\n\n--- PAGE {page_num + 1} ---\n\n{text}"
                pages_data.append({"page_num": page_num, "text": text})

            doc.close()

            # Extract additional metadata using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                # The first page usually contains title and authors
                first_page = pdf.pages[0]
                first_page_text = first_page.extract_text()

                # Additional processing could be done here

            return full_text, pages_data

        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            raise

    def identify_sections(
        self, full_text: str, pages_data: List[Dict]
    ) -> Dict[str, PaperSection]:
        """Identify and classify sections in the paper"""
        self.logger.info("Identifying paper sections")

        # Start with rule-based section identification
        sections = self._rule_based_section_identification(full_text)

        # Apply ML-based classification for confirmation and confidence scoring
        sections = self._ml_classify_sections(sections)

        return sections

    def _rule_based_section_identification(
        self, full_text: str
    ) -> Dict[str, PaperSection]:
        """Use rule-based approaches to identify paper sections"""
        sections = {}

        # Define section headers to look for (without regex special characters)
        section_headers = [
            "abstract",
            "introduction",
            "background",
            "related work",
            "methods",
            "methodology",
            "materials and methods",
            "experimental",
            "experiments",
            "results",
            "findings",
            "discussion",
            "conclusion",
            "summary",
            "references",
        ]

        try:
            # Create simple patterns for each section header
            patterns = []
            for header in section_headers:
                # Escape any regex special characters in the header
                escaped_header = re.escape(header)
                # Create pattern with optional numbering and flexible whitespace
                pattern = r"^\s*(?:\d+\s*\.\s*)?(" + escaped_header + r")\s*$"
                patterns.append(pattern)

            # Combine patterns with OR operator
            combined_pattern = "|".join(patterns)

            # Compile the regex with appropriate flags
            regex = re.compile(combined_pattern, re.IGNORECASE | re.MULTILINE)

            # Find all matches in the text
            matches = list(regex.finditer(full_text))

            # Process matches to extract sections
            for i in range(len(matches)):
                match = matches[i]
                section_header = match.group(1)  # Get the matched header
                section_type = self._determine_section_type(section_header)

                # Get section content (text until next section or end)
                start_pos = match.end()
                end_pos = (
                    matches[i + 1].start() if i < len(matches) - 1 else len(full_text)
                )
                content = full_text[start_pos:end_pos].strip()

                # Extract page numbers
                page_numbers = self._extract_page_numbers(content)

                # Add to sections dictionary
                if section_type:
                    sections[section_type] = PaperSection(
                        title=section_header,
                        content=content,
                        section_type=section_type,
                        page_numbers=page_numbers,
                        confidence=0.8,
                    )

        except Exception as e:
            self.logger.error(f"Error in regex section identification: {e}")
            # Try a simpler fallback approach
            self._fallback_section_identification(full_text, sections)

        # Special handling for abstract if not found
        if "abstract" not in sections and full_text:
            # Try to extract abstract from the beginning of the paper
            first_chunk = " ".join(full_text.split()[:500])
            if "abstract" in first_chunk.lower():
                abstract_text = self._extract_abstract(full_text)
                if abstract_text:
                    sections["abstract"] = PaperSection(
                        title="Abstract",
                        content=abstract_text,
                        section_type="abstract",
                        page_numbers=[0, 1],  # Usually on first page
                        confidence=0.7,
                    )

        return sections

    def _fallback_section_identification(
        self, full_text: str, sections: Dict[str, PaperSection]
    ):
        """Fallback method if regex approach fails"""
        # Simple string-based approach
        lines = full_text.split("\n")
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line looks like a section header
            is_header = False
            for section_type, pattern in self.SECTION_PATTERNS.items():
                if re.search(pattern, line, re.IGNORECASE):
                    # Save previous section if exists
                    if current_section and current_content:
                        sections[current_section] = PaperSection(
                            title=line,
                            content="\n".join(current_content),
                            section_type=current_section,
                            page_numbers=[],  # No page tracking in fallback
                            confidence=0.5,  # Lower confidence for fallback
                        )

                    # Start new section
                    current_section = section_type
                    current_content = []
                    is_header = True
                    break

            # If not a header, add to current content
            if not is_header and current_section:
                current_content.append(line)

        # Add final section
        if current_section and current_content:
            sections[current_section] = PaperSection(
                title=current_section.capitalize(),
                content="\n".join(current_content),
                section_type=current_section,
                page_numbers=[],
                confidence=0.5,
            )

    def _determine_section_type(self, section_header: str) -> Optional[str]:
        """Map a section header to a standard section type"""
        section_header = section_header.lower()

        for section_type, patterns in self.SECTION_PATTERNS.items():
            if re.search(patterns, section_header, re.IGNORECASE):
                return section_type

        # If no match found, try more heuristics
        if re.search(r"related|previous|prior", section_header):
            return "introduction"

        # If still no match, return None
        return None

    def _extract_page_numbers(self, text: str) -> List[int]:
        """Extract page numbers mentioned in the text"""
        # This is a simple implementation - could be improved
        page_matches = re.findall(r"--- PAGE (\d+) ---", text)
        return [int(page) - 1 for page in page_matches]

    def _extract_abstract(self, full_text: str) -> Optional[str]:
        """Extract abstract from the beginning of the paper"""
        # Look for the abstract section
        abstract_match = re.search(
            r"(?i)abstract[:\.\s]+(.*?)(?:\n\n|\r\n\r\n|$)", full_text, re.DOTALL
        )

        if abstract_match:
            return abstract_match.group(1).strip()

        return None

    def _ml_classify_sections(
        self, sections: Dict[str, PaperSection]
    ) -> Dict[str, PaperSection]:
        """Use ML to classify and improve confidence in section identification"""
        # For each section, confirm its type and assign a confidence score
        for section_type, section in sections.items():
            try:
                # Use the section model to classify
                inputs = self.section_tokenizer(
                    section.content[:512],  # Use first 512 chars for classification
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.section_model(**inputs)
                    probs = (
                        torch.nn.functional.softmax(outputs.logits, dim=-1)
                        .cpu()
                        .numpy()[0]
                    )

                # Get the predicted label
                section_types = [
                    "abstract",
                    "introduction",
                    "methods",
                    "results",
                    "discussion",
                    "conclusion",
                ]
                predicted_idx = np.argmax(probs)
                predicted_type = section_types[predicted_idx]
                confidence = float(probs[predicted_idx])

                # Update the section if prediction is different with high confidence
                if predicted_type != section_type and confidence > 0.8:
                    section.section_type = predicted_type
                    # Rename key in the dictionary
                    sections[predicted_type] = sections.pop(section_type)

                # Update confidence
                section.confidence = confidence

            except Exception as e:
                self.logger.error(
                    f"Error during ML classification of section {section_type}: {e}"
                )

        return sections

    def extract_key_concepts(
        self, sections: Dict[str, PaperSection]
    ) -> List[KeyConcept]:
        """Extract key concepts and their definitions from the paper"""
        self.logger.info("Extracting key concepts")

        key_concepts = []

        # Combine text from important sections for concept extraction
        combined_text = ""
        for section_type in ["abstract", "introduction", "conclusion"]:
            if section_type in sections:
                combined_text += sections[section_type].content + " "

        # Tokenize into sentences
        sentences = sent_tokenize(combined_text)

        # Skip if no sentences
        if not sentences:
            return key_concepts

        # Extract terms using TF-IDF
        try:
            # Fit TF-IDF
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            feature_names = self.tfidf.get_feature_names_out()

            # Get important terms
            word_scores = zip(
                feature_names, np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            )
            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            top_terms = [word for word, score in word_scores[:20]]

            # Find definitions and context for top terms
            for term in top_terms:
                # Find sentences containing this term
                term_sentences = [s for s in sentences if term.lower() in s.lower()]

                if term_sentences:
                    # Use the first sentence as context
                    context = term_sentences[0]

                    # Look for definition patterns
                    definition = self._extract_definition(term, term_sentences)

                    # Find which sections contain this term
                    term_sections = []
                    for section_type, section in sections.items():
                        if term.lower() in section.content.lower():
                            term_sections.append(section_type)

                    # Calculate importance score
                    importance_score = self._calculate_importance_score(term, sections)

                    # Add to concepts
                    key_concepts.append(
                        KeyConcept(
                            term=term,
                            definition=definition,
                            importance_score=importance_score,
                            source_sections=term_sections,
                            context=context,
                        )
                    )

        except Exception as e:
            self.logger.error(f"Error extracting key concepts: {e}")

        return key_concepts

    def _extract_definition(self, term: str, sentences: List[str]) -> str:
        """Extract a definition for a term based on sentence patterns"""
        definition_patterns = [
            rf"{term}\s+(?:is|are|refers to|describes?|represents?|consists? of|defined as)\s+(.*?)[,\.]",
            rf"{term}[,\s]+(?:or|also called|also known as)[,\s]+(.*?)[,\.]",
            rf"(?:we|authors)\s+(?:define|call|use|refer to)\s+(?:the|a|an)?\s+{term}\s+(?:as|to mean)\s+(.*?)[,\.]",
        ]

        # Try to match definition patterns
        for sentence in sentences:
            for pattern in definition_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        # If no definition pattern found, use the sentence with the term
        for sentence in sentences:
            if term.lower() in sentence.lower():
                return sentence.strip()

        return ""

    def _calculate_importance_score(
        self, term: str, sections: Dict[str, PaperSection]
    ) -> float:
        """Calculate importance score for a concept based on frequency and location"""
        score = 0.0

        # Section weights
        section_weights = {
            "abstract": 1.0,
            "introduction": 0.7,
            "conclusion": 0.8,
            "results": 0.6,
            "methods": 0.5,
            "discussion": 0.6,
        }

        # Count occurrences in each section
        for section_type, section in sections.items():
            weight = section_weights.get(section_type, 0.3)
            count = section.content.lower().count(term.lower())
            score += count * weight

        # Normalize
        return min(1.0, score / 10.0)  # Cap at 1.0

    def generate_summaries(self, sections: Dict[str, PaperSection]) -> Dict[str, str]:
        """Generate summaries for each section and the entire paper"""
        self.logger.info("Generating summaries")

        summaries = {}

        # Summarize each section
        for section_type, section in sections.items():
            if section_type in ["abstract", "methods", "results", "conclusion"]:
                try:
                    max_length = 100 if section_type == "abstract" else 150
                    summary = self.summarize_text(
                        section.content, max_length=max_length
                    )
                    summaries[section_type] = summary
                except Exception as e:
                    self.logger.error(f"Error summarizing {section_type}: {e}")
                    summaries[section_type] = section.content[:200] + "..."

        # Generate full paper summary
        full_summary = self.generate_full_summary(sections, summaries)
        summaries["full"] = full_summary

        # Generate significance statement
        if "abstract" in sections and "conclusion" in sections:
            try:
                significance = self.generate_significance(
                    sections["abstract"].content, sections["conclusion"].content
                )
                summaries["significance"] = significance
            except Exception as e:
                self.logger.error(f"Error generating significance: {e}")
                summaries["significance"] = "Could not generate significance statement."

        return summaries

    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize a chunk of text using the summarization model"""
        if len(text.split()) < 50:  # Too short to summarize
            return text

        # Break into chunks if text is long
        words = text.split()

        if len(words) <= 800:  # Within model's ability to process at once
            try:
                summary = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min(30, max_length // 2),
                    do_sample=False,
                )[0]["summary_text"]
                return summary
            except Exception as e:
                self.logger.error(f"Error in summarization: {e}")
                return text[:200] + "..."
        else:
            # Process in chunks
            chunks = []
            for i in range(0, len(words), 600):
                chunk = " ".join(words[i : i + 800])
                chunks.append(chunk)

            # Summarize each chunk and combine
            chunk_summaries = []
            for chunk in chunks:
                try:
                    summary = self.summarizer(
                        chunk,
                        max_length=max_length // len(chunks),
                        min_length=min(20, max_length // (2 * len(chunks))),
                        do_sample=False,
                    )[0]["summary_text"]
                    chunk_summaries.append(summary)
                except Exception as e:
                    self.logger.error(f"Error in chunk summarization: {e}")
                    chunk_summaries.append(chunk[:100] + "...")

            return " ".join(chunk_summaries)

    def generate_full_summary(
        self, sections: Dict[str, PaperSection], section_summaries: Dict[str, str]
    ) -> str:
        """Generate a full summary of the paper"""
        # Compile a comprehensive summary using section summaries
        summary_parts = []

        for section_type in ["abstract", "methods", "results", "conclusion"]:
            if section_type in section_summaries:
                summary_parts.append(
                    f"{section_type.capitalize()}: {section_summaries[section_type]}"
                )

        if summary_parts:
            return "\n\n".join(summary_parts)
        else:
            # Fallback: Use a direct summarization of abstract and conclusion
            combined_text = ""
            if "abstract" in sections:
                combined_text += sections["abstract"].content + " "
            if "conclusion" in sections:
                combined_text += sections["conclusion"].content

            if combined_text:
                return self.summarize_text(combined_text, max_length=250)
            else:
                return "Could not generate a comprehensive summary."

    def generate_significance(self, abstract: str, conclusion: str) -> str:
        """Generate a statement about the paper's significance"""
        combined = abstract + " " + conclusion
        prompt = (
            "Based on the abstract and conclusion, what is the significance of this research? "
            "What are the main contributions and potential impact?"
        )

        # Use summarization model to generate significance statement
        try:
            significance = self.summarizer(
                combined, max_length=100, min_length=30, do_sample=True, temperature=0.7
            )[0]["summary_text"]

            return significance
        except Exception as e:
            self.logger.error(f"Error generating significance: {e}")
            return "Could not generate significance statement."

    def extract_metadata(self, full_text: str) -> Dict[str, Any]:
        """Extract metadata from the paper"""
        self.logger.info("Extracting metadata")

        metadata = {"title": "", "authors": [], "year": None, "doi": None}

        # Extract title - usually at the beginning of the paper
        first_page = (
            full_text.split("--- PAGE 2 ---")[0]
            if "--- PAGE 2 ---" in full_text
            else full_text[:1000]
        )
        title_match = re.search(
            r"^\s*(?!abstract)([\w\s:,\-–—]+)$", first_page, re.MULTILINE
        )

        if title_match:
            candidate_title = title_match.group(1).strip()
            # Keep only if it looks like a title (not too long or short)
            if 4 <= len(candidate_title.split()) <= 25:
                metadata["title"] = candidate_title

        # Extract authors - usually after the title
        author_section = (
            first_page.split("Abstract")[0] if "Abstract" in first_page else first_page
        )
        author_match = re.search(
            r"^\s*((?:[A-Z][a-z]+\s+[A-Z][a-z]+(?:,|\s+and|\s+&)?)+)",
            author_section,
            re.MULTILINE,
        )

        if author_match:
            author_text = author_match.group(1)
            # Split authors by common separators
            authors = re.split(r",|\s+and|\s+&", author_text)
            metadata["authors"] = [
                author.strip() for author in authors if author.strip()
            ]

        # Extract year - look for year pattern
        year_match = re.search(r"(?:©|\(c\)|\(C\)|\()\s*(\d{4})", full_text)
        if year_match:
            try:
                year = int(year_match.group(1))
                if 1900 <= year <= 2100:  # Sanity check
                    metadata["year"] = year
            except ValueError:
                pass

        # Extract DOI
        doi_match = re.search(
            r"(?:doi|DOI)[:.\s]+([0-9]+\.[0-9]+/[a-zA-Z0-9./_-]+)", full_text
        )
        if doi_match:
            metadata["doi"] = doi_match.group(1).strip()

        return metadata

    def process_paper(self, pdf_path: str) -> PaperAnalysis:
        """Process a paper and create a complete analysis"""
        self.logger.info(f"Processing paper: {pdf_path}")

        try:
            # 1. Extract text from PDF
            full_text, pages_data = self.extract_text_from_pdf(pdf_path)

            # 2. Extract metadata
            metadata = self.extract_metadata(full_text)

            # 3. Identify sections
            sections = self.identify_sections(full_text, pages_data)

            # 4. Extract key concepts and keywords
            key_concepts = self.extract_key_concepts(sections)

            # Extract simple keywords
            keywords = [concept.term for concept in key_concepts]

            # 5. Generate summaries
            summaries = self.generate_summaries(sections)

            # 6. Create the analysis object
            analysis = PaperAnalysis(
                title=metadata["title"],
                authors=metadata["authors"],
                publication_year=metadata["year"],
                doi=metadata["doi"],
                sections=sections,
                key_concepts=key_concepts,
                keywords=keywords[:10],
                summary=summaries.get("full", ""),
                significance=summaries.get("significance", ""),
            )

            # Add section summaries
            for section_type in ["abstract", "methods", "results", "conclusion"]:
                if section_type in summaries:
                    setattr(analysis, section_type, summaries[section_type])

            self.logger.info(f"Paper analysis complete: {pdf_path}")
            return analysis

        except Exception as e:
            self.logger.error(f"Error processing paper: {e}")
            raise


def main():
    """Main function to run the paper processor from command line"""
    parser = argparse.ArgumentParser(
        description="Process academic papers for analysis and summarization"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: based on input name)",
        default=None,
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")

    args = parser.parse_args()

    # Set output path if not specified
    if not args.output:
        output_dir = os.path.dirname(args.pdf_path) or "."
        base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        args.output = os.path.join(output_dir, f"{base_name}_analysis.json")

    # Initialize and run the processor
    processor = PaperProcessor(use_gpu=not args.no_gpu)

    print(f"Processing paper: {args.pdf_path}")
    analysis = processor.process_paper(args.pdf_path)

    # Save results
    analysis.save_to_file(args.output)
    print(f"Analysis complete. Results saved to: {args.output}")

    # Print quick summary
    print("\n=== Paper Analysis Summary ===")
    print(f"Title: {analysis.title}")
    print(f"Authors: {', '.join(analysis.authors)}")

    if analysis.abstract:
        print(f"\nAbstract Summary: {analysis.abstract[:200]}...")

    print("\nKey Concepts:")
    for concept in analysis.key_concepts[:5]:
        print(
            f"- {concept.term}: {concept.definition[:100]}..."
            if len(concept.definition) > 100
            else concept.definition
        )

    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()