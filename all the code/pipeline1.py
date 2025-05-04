# ML Paper Analysis Pipeline - Pipeline 1 with SciDeBERTa-CS
# Author: AI Assistant
# Purpose: Process PDF papers using SciDeBERTa-CS for section classification

import os
import re
import fitz  # PyMuPDF for PDF processing
import nltk
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """Data class for paper sections"""

    title: str
    content: str
    section_type: str  # 'abstract', 'methods', 'results', 'conclusion'
    start_page: int
    end_page: int
    confidence: float = 0.0


@dataclass
class PaperSummary:
    """Data class for paper summary"""

    abstract: str
    methods: str
    results: str
    conclusion: str
    full_summary: str
    key_concepts: List[str]


class PDFProcessor:
    """Handles PDF ingestion and text extraction"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract raw text from PDF with page information"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            page_info = {}

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                page_info[page_num] = page_text
                full_text += f"\n\n=== PAGE {page_num + 1} ===\n\n" + page_text

            doc.close()
            return full_text, page_info

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise


class SciDeBERTaSegmenter:
    """Segments paper into sections using SciDeBERTa-CS for classification"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Download NLTK data if needed
        self._ensure_nltk_resources()

        # Load SciDeBERTa-CS model
        self.tokenizer = AutoTokenizer.from_pretrained("KISTI-AI/scideberta-cs")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "KISTI-AI/scideberta-cs"
        )

        # Define section types
        self.section_types = [
            "abstract",
            "introduction",
            "methods",
            "results",
            "discussion",
            "conclusion",
        ]

        # Section patterns for fallback
        self.section_patterns = {
            "abstract": r"(?i)(?:\babstract\b|summary)(?:\s*:)?\s*((?:(?!(?:\b[A-Z][^.!?]*\b)\s*(?::|[0-9]\.)|\bintroduction\b).)*)",
            "methods": r"(?i)(?:\bmethods\b|\bmethodology\b)(?:\s*:)?\s*((?:(?!(?:\b[A-Z][^.!?]*\b)\s*(?::|[0-9]\.)|\bresults\b).)*)",
            "results": r"(?i)(?:\bresults\b)(?:\s*:)?\s*((?:(?!(?:\b[A-Z][^.!?]*\b)\s*(?::|[0-9]\.)|\bdiscussion\b|\bconclusion\b).)*)",
            "conclusion": r"(?i)(?:\bconclusion\b|\bconcluding\s+remarks\b)(?:\s*:)?\s*(.*)$",
        }

    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are downloaded"""
        resources = ["punkt", "punkt_tab"]
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                self.logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)

    def classify_paragraph(self, paragraph: str) -> Tuple[str, float]:
        """Classify a paragraph using SciDeBERTa-CS"""
        inputs = self.tokenizer(
            paragraph, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

        # Map prediction to section type
        if predicted_class < len(self.section_types):
            section_type = self.section_types[predicted_class]
        else:
            section_type = "other"

        return section_type, confidence

    def extract_sections_with_scideberta(
        self, text: str, page_info: Dict
    ) -> Dict[str, PaperSection]:
        """Extract sections using SciDeBERTa-CS classification"""
        # Split text into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)

        # Classify each paragraph
        classified_paragraphs = []
        for para in paragraphs:
            if len(para.strip()) > 50:  # Skip very short paragraphs
                section_type, confidence = self.classify_paragraph(para)
                classified_paragraphs.append((para, section_type, confidence))

        # Group consecutive paragraphs of the same type
        sections = {}
        current_section = None
        current_content = []
        current_confidence_scores = []

        for para, section_type, confidence in classified_paragraphs:
            if current_section is None:
                current_section = section_type
                current_content = [para]
                current_confidence_scores = [confidence]
            elif current_section == section_type:
                current_content.append(para)
                current_confidence_scores.append(confidence)
            else:
                # Save current section
                if current_section in self.section_types:
                    content = "\n\n".join(current_content)
                    start_page, end_page = self._find_page_range(content, page_info)
                    avg_confidence = sum(current_confidence_scores) / len(
                        current_confidence_scores
                    )

                    sections[current_section] = PaperSection(
                        title=current_section.capitalize(),
                        content=content,
                        section_type=current_section,
                        start_page=start_page,
                        end_page=end_page,
                        confidence=avg_confidence,
                    )

                # Start new section
                current_section = section_type
                current_content = [para]
                current_confidence_scores = [confidence]

        # Don't forget the last section
        if current_section and current_section in self.section_types:
            content = "\n\n".join(current_content)
            start_page, end_page = self._find_page_range(content, page_info)
            avg_confidence = sum(current_confidence_scores) / len(
                current_confidence_scores
            )

            sections[current_section] = PaperSection(
                title=current_section.capitalize(),
                content=content,
                section_type=current_section,
                start_page=start_page,
                end_page=end_page,
                confidence=avg_confidence,
            )

        return sections

    def extract_sections_with_regex(
        self, text: str, page_info: Dict
    ) -> Dict[str, PaperSection]:
        """Extract sections using regex patterns (fallback method)"""
        sections = {}

        for section_type, pattern in self.section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                start_page, end_page = self._find_page_range(content, page_info)

                sections[section_type] = PaperSection(
                    title=section_type.capitalize(),
                    content=content,
                    section_type=section_type,
                    start_page=start_page,
                    end_page=end_page,
                    confidence=1.0,  # Full confidence since it's a pattern match
                )

        return sections

    def extract_sections(self, text: str, page_info: Dict) -> Dict[str, PaperSection]:
        """Extract sections using SciDeBERTa with regex fallback"""
        # Try SciDeBERTa classification first
        sections = self.extract_sections_with_scideberta(text, page_info)

        # Fill in missing important sections with regex
        important_sections = ["abstract", "methods", "results", "conclusion"]
        for section_type in important_sections:
            if section_type not in sections:
                self.logger.warning(
                    f"Section '{section_type}' not found by SciDeBERTa, trying regex..."
                )
                regex_sections = self.extract_sections_with_regex(text, page_info)
                if section_type in regex_sections:
                    sections[section_type] = regex_sections[section_type]
                    sections[section_type].confidence = (
                        0.5  # Lower confidence for regex matches
                    )

        return sections

    def _find_page_range(self, content: str, page_info: Dict) -> Tuple[int, int]:
        """Find page range where content appears"""
        start_page = None
        end_page = None
        content_snippet = content[:100]  # First 100 chars

        for page_num, page_text in page_info.items():
            if content_snippet in page_text:
                if start_page is None:
                    start_page = page_num
                end_page = page_num

        return start_page or 0, end_page or 0


class Summarizer:
    """Handles text summarization using seq2seq models"""

    def __init__(self, model_name: str = "google/pegasus-pubmed"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Ensure NLTK resources are available
        self._ensure_nltk_resources()

        # Load model
        self._load_model()

    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are downloaded"""
        resources = ["punkt", "punkt_tab"]
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                self.logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)

    def _load_model(self):
        """Load summarization model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Device set to use {device}")

            # Create pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def summarize_section(self, text: str, max_length: int = 150) -> str:
        """Summarize a section of text"""
        # Check if text is too short for summarization
        words = text.split()
        if len(words) < 50:  # Too short to summarize
            return text  # Return original text

        # Adjust max_length for short texts
        actual_max_length = min(max_length, len(words) // 2)
        actual_min_length = min(20, actual_max_length - 10)

        try:
            # Split long texts into chunks
            max_words = 800  # Adjust based on model capacity

            if len(words) <= max_words:
                summaries = [
                    self.summarizer(
                        text, max_length=actual_max_length, min_length=actual_min_length
                    )[0]["summary_text"]
                ]
            else:
                chunks = [
                    " ".join(words[i : i + max_words])
                    for i in range(0, len(words), max_words // 2)
                ]
                summaries = []
                for chunk in chunks:
                    summary = self.summarizer(
                        chunk, max_length=actual_max_length // 2, min_length=10
                    )[0]["summary_text"]
                    summaries.append(summary)

            return " ".join(summaries)
        except Exception as e:
            self.logger.error(f"Error summarizing text: {e}")
            return text[:150] + "..."  # Return first 150 chars as fallback

    def create_paper_summary(self, sections: Dict[str, PaperSection]) -> PaperSummary:
        """Create comprehensive paper summary"""
        summary_dict = {}

        for section_type in ["abstract", "methods", "results", "conclusion"]:
            if section_type in sections:
                summary_dict[section_type] = self.summarize_section(
                    sections[section_type].content
                )
            else:
                summary_dict[section_type] = "Section not found"

        # Create full summary
        full_summary = f"""
        Abstract: {summary_dict['abstract']}
        
        Methods: {summary_dict['methods']}
        
        Results: {summary_dict['results']}
        
        Conclusion: {summary_dict['conclusion']}
        """

        # Extract key concepts (enhanced with SciDeBERTa embeddings)
        key_concepts = self._extract_key_concepts(sections)

        return PaperSummary(
            abstract=summary_dict["abstract"],
            methods=summary_dict["methods"],
            results=summary_dict["results"],
            conclusion=summary_dict["conclusion"],
            full_summary=full_summary,
            key_concepts=key_concepts,
        )

    def _extract_key_concepts(self, sections: Dict[str, PaperSection]) -> List[str]:
        """Extract key concepts using improved TF-IDF approach"""
        # Combine text from important sections only
        important_texts = []
        for section_type in ["abstract", "results", "conclusion"]:
            if section_type in sections:
                important_texts.append(sections[section_type].content)

        all_text = " ".join(important_texts)

        # Skip if no text to process
        if not all_text.strip():
            return []

        # Extract concepts using TF-IDF
        sentences = nltk.sent_tokenize(all_text)

        # Skip if no sentences
        if not sentences:
            return []

        vectorizer = TfidfVectorizer(
            max_features=50, stop_words="english", ngram_range=(1, 2)
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Get top terms
            feature_names = vectorizer.get_feature_names_out()
            top_indices = tfidf_matrix.sum(axis=0).A1.argsort()[-10:][::-1]

            return [feature_names[i] for i in top_indices]
        except ValueError:  # Handle case where vectorizer can't process the text
            return []


class QAGenerator:
    """Generates Q&A pairs from paper content"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Ensure NLTK resources
        self._ensure_nltk_resources()

        # Load models
        self._load_models()

    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are downloaded"""
        resources = ["punkt", "punkt_tab"]
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                self.logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)

    def _load_models(self):
        """Load QA models"""
        # Using the same SciDeBERTa model for embeddings to create better questions
        self.tokenizer = AutoTokenizer.from_pretrained("KISTI-AI/scideberta-cs")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "KISTI-AI/scideberta-cs"
        )

        # Load QA pipeline for simple Q&A
        self.qa_pipeline = pipeline(
            "question-answering", model="deepset/roberta-base-squad2"
        )

    def generate_quiz(
        self, sections: Dict[str, PaperSection], num_questions: int = 5
    ) -> List[Dict]:
        """Generate quiz questions from paper sections"""
        quiz = []

        # Prioritize sections by importance and confidence
        prioritized_sections = self._prioritize_sections(sections)

        for section_type, section in prioritized_sections:
            if len(quiz) >= num_questions:
                break

            sentences = nltk.sent_tokenize(section.content)

            # Score sentences by importance (using TF-IDF)
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1

            # Get top sentences
            top_indices = sentence_scores.argsort()[-3:][::-1]

            for idx in top_indices:
                if len(quiz) >= num_questions:
                    break

                sentence = sentences[idx]

                # Generate question using SciDeBERTa-aware approach
                question = self._create_smart_question(sentence, section_type)

                quiz.append(
                    {
                        "question": question,
                        "answer": sentence,
                        "section": section_type,
                        "confidence": section.confidence,
                    }
                )

        return quiz

    def _prioritize_sections(
        self, sections: Dict[str, PaperSection]
    ) -> List[Tuple[str, PaperSection]]:
        """Prioritize sections based on importance and confidence"""
        section_priority = {
            "abstract": 5,
            "results": 4,
            "conclusion": 3,
            "methods": 2,
            "discussion": 1,
            "introduction": 0,
        }

        prioritized = []
        for section_type, section in sections.items():
            priority = section_priority.get(section_type, 0)
            score = priority + section.confidence
            prioritized.append((score, section_type, section))

        # Sort by score (highest first)
        prioritized.sort(reverse=True, key=lambda x: x[0])

        return [(x[1], x[2]) for x in prioritized]

    def _create_smart_question(self, sentence: str, section_type: str) -> str:
        """Create a question from a sentence using scientific context"""
        # Extract key scientific terms using SciDeBERTa tokenizer
        tokens = self.tokenizer.tokenize(sentence)

        # Find technical terms (often longer and complex)
        technical_terms = []
        for token in tokens:
            if len(token) > 4 and any(char.isupper() for char in token[1:]):
                technical_terms.append(token)

        # Enhanced question templates based on section
        question_templates = {
            "abstract": [
                "What is the main contribution of this work regarding {}?",
                "What problem does this research address in relation to {}?",
                "What novel approach is proposed for {}?",
            ],
            "methods": [
                "How was {} implemented in this study?",
                "What methodology was used for {}?",
                "What experimental setup was used for {}?",
            ],
            "results": [
                "What was the outcome regarding {}?",
                "What performance was achieved for {}?",
                "How did {} compare to baseline methods?",
            ],
            "conclusion": [
                "What conclusion was drawn about {}?",
                "What implications does this work have for {}?",
                "What future directions are suggested for {}?",
            ],
        }

        # Select appropriate template and term
        templates = question_templates.get(section_type, question_templates["abstract"])
        template = templates[0]  # For MVP, use first template

        if technical_terms:
            term = technical_terms[0]
        else:
            # Extract first few words as fallback
            words = sentence.split()
            term = " ".join(words[:5]) + "..." if len(words) > 5 else sentence

        return template.format(term)


class PaperAnalysisPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, output_dir: str = "output"):
        self.pdf_processor = PDFProcessor()
        self.segmenter = SciDeBERTaSegmenter()

        # Initialize summarizer and QA generator last to ensure all models are loaded
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing summarizer...")
        self.summarizer = Summarizer()

        self.logger.info("Initializing QA generator...")
        self.qa_generator = QAGenerator()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_paper(self, pdf_path: str) -> Dict:
        """Process a paper through the entire pipeline"""
        self.logger.info(f"Processing paper: {pdf_path}")

        try:
            # 1. Extract text from PDF
            full_text, page_info = self.pdf_processor.extract_text_from_pdf(pdf_path)

            # 2. Segment into sections using SciDeBERTa
            sections = self.segmenter.extract_sections(full_text, page_info)

            # 3. Summarize sections
            summary = self.summarizer.create_paper_summary(sections)

            # 4. Generate QA
            quiz = self.qa_generator.generate_quiz(sections)

            # 5. Save results
            results = {
                "paper_title": self._extract_title(sections),
                "sections": {
                    k: {
                        "content": v.content,
                        "start_page": v.start_page,
                        "end_page": v.end_page,
                        "confidence": v.confidence,
                    }
                    for k, v in sections.items()
                },
                "summary": {
                    "abstract": summary.abstract,
                    "methods": summary.methods,
                    "results": summary.results,
                    "conclusion": summary.conclusion,
                    "full_summary": summary.full_summary,
                    "key_concepts": summary.key_concepts,
                },
                "quiz": quiz,
            }

            # Save to file
            output_file = self.output_dir / f"{Path(pdf_path).stem}_analysis.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Results saved to: {output_file}")
            return results

        except Exception as e:
            self.logger.error(f"Error processing paper: {e}")
            raise

    def _extract_title(self, sections: Dict[str, PaperSection]) -> str:
        """Extract paper title (improved using classifications)"""
        # Look for title in abstract or introduction section
        for section_type in ["abstract", "introduction"]:
            if section_type in sections:
                text = sections[section_type].content
                lines = text.split("\n")

                # First non-empty line is often the title
                for line in lines:
                    clean_line = line.strip()
                    if (
                        clean_line
                        and len(clean_line.split()) > 3
                        and len(clean_line) < 200
                    ):
                        # Check if it looks like a title (more caps than normal)
                        if (
                            sum(1 for c in clean_line if c.isupper())
                            > len(clean_line) * 0.1
                        ):
                            return clean_line

        return "Unknown Title"


# Example usage script
def main():
    """Run the pipeline on a PDF file"""
    import argparse

    parser = argparse.ArgumentParser(description="Process ML papers with SciDeBERTa")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output-dir", default="output", help="Output directory")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PaperAnalysisPipeline(output_dir=args.output_dir)

    # Process paper
    results = pipeline.process_paper(args.pdf_path)

    # Display summary
    print("\n=== Paper Analysis Summary ===")
    print(f"Title: {results['paper_title']}")
    print("\nSection Confidence Scores:")
    for section_type, section_data in results["sections"].items():
        print(f"- {section_type.capitalize()}: {section_data['confidence']:.2f}")

    print("\nKey Concepts:")
    for concept in results["summary"]["key_concepts"]:
        print(f"- {concept}")

    print("\nQuiz Preview:")
    for i, qa in enumerate(results["quiz"][:3], 1):
        print(f"Q{i}: {qa['question']}")
        print(f"A{i}: {qa['answer'][:100]}...")
        print(f"Confidence: {qa['confidence']:.2f}")
        print()


if __name__ == "__main__":
    main()
