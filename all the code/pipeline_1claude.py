# Fixed ML Paper Analysis Pipeline
# Author: AI Assistant
# Purpose: Process PDF papers using PubLayNet + SciDeBERTa-CS

import os
import sys
import logging

# Force environment variable to disable PubLayNet initially (safer)
os.environ["USE_PUBLAYNET"] = "false"

# Force CPU for PyTorch to avoid CUDA assertion errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set torch to use CPU
import torch

torch.cuda.is_available = lambda: False

# Import pipeline after environment setup
from pipeline_1claude import EnhancedAnalysisPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import re
import fitz  # PyMuPDF for PDF processing
import nltk
import torch
import cv2
import numpy as np

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

    detectron2_available = True
except ImportError:
    detectron2_available = False
    print("Warning: detectron2 not available. Will use fallback methods.")

try:
    from pdf2image import convert_from_path

    pdf2image_available = True
except ImportError:
    pdf2image_available = False
    print("Warning: pdf2image not available. Layout detection will be skipped.")

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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """Data class for paper sections with layout information"""

    title: str
    content: str
    section_type: str  # 'abstract', 'methods', 'results', 'conclusion'
    start_page: int
    end_page: int
    confidence: float = 0.0
    bounding_box: Optional[Dict] = None


@dataclass
class LayoutDetection:
    """Data class for layout detection results"""

    page_num: int
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    score: float
    class_name: str  # 'title', 'text', 'list', 'table', 'figure'


@dataclass
class PaperSummary:
    """Data class for paper summary"""

    abstract: str
    methods: str
    results: str
    conclusion: str
    full_summary: str
    key_concepts: List[str]


class PubLayNetProcessor:
    """Handles PDF layout detection using PubLayNet model"""

    def __init__(
        self, config_path="publaynet_config.yaml", weights_path="model_final.pth"
    ):
        self.logger = logging.getLogger(__name__)

        if not detectron2_available:
            self.logger.warning(
                "Detectron2 not available. Skipping PubLayNet initialization."
            )
            self.predictor = None
            return

        try:
            # Initialize PubLayNet predictor
            cfg = get_cfg()

            # Create a simple config if file doesn't exist
            if not os.path.exists(config_path):
                self.logger.warning(
                    f"Config file {config_path} not found. Using default config."
                )
                from detectron2.model_zoo import get_config_file

                cfg.merge_from_file(
                    get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
                )
            else:
                # Read and fix the config file
                with open(config_path, "r") as f:
                    content = f.read()
                    # Remove DEVICE key if present
                    content = content.replace('DEVICE: "cpu"\n', "")
                    content = content.replace("DEVICE: cpu\n", "")
                    # Save to a temporary file
                    with open("temp_config.yaml", "w") as temp_f:
                        temp_f.write(content)
                cfg.merge_from_file("temp_config.yaml")

            if os.path.exists(weights_path):
                cfg.MODEL.WEIGHTS = weights_path
            else:
                self.logger.warning(
                    f"Weights file {weights_path} not found. Using default weights."
                )
                cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

            self.predictor = DefaultPredictor(cfg)

            # Define class mappings for PubLayNet
            self.class_names = {
                0: "Text",
                1: "Title",
                2: "List",
                3: "Table",
                4: "Figure",
            }

        except Exception as e:
            self.logger.error(f"Error initializing PubLayNet: {e}")
            self.predictor = None

        # Section keywords for matching
        self.section_keywords = {
            "abstract": ["abstract", "summary"],
            "introduction": ["introduction", "background"],
            "methods": ["methods", "methodology", "approach"],
            "results": ["results", "findings", "evaluation"],
            "discussion": ["discussion", "analysis"],
            "conclusion": ["conclusion", "conclusions", "summary"],
        }

    def process_page(self, page_image: np.ndarray) -> List[LayoutDetection]:
        """Process a single page with PubLayNet"""
        if self.predictor is None:
            return []

        outputs = self.predictor(page_image)
        instances = outputs["instances"].to("cpu")

        detections = []
        for i in range(len(instances)):
            bbox = instances.pred_boxes[i].tensor.numpy()[0]
            class_id = instances.pred_classes[i].item()
            score = instances.scores[i].item()

            detection = LayoutDetection(
                page_num=None,  # Will be set by caller
                bbox=bbox.tolist(),
                class_id=class_id,
                score=score,
                class_name=self.class_names.get(class_id, "Unknown"),
            )
            detections.append(detection)

        return detections

    def extract_text_from_region(self, page, bbox: List[float]) -> str:
        """Extract text from a specific region of a PDF page"""
        rect = fitz.Rect(bbox)
        return page.get_text("text", clip=rect)


class EnhancedSegmenter:
    """Segments paper using PubLayNet + SciDeBERTa-CS"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize PubLayNet
        self.publaynet = PubLayNetProcessor()

        # Initialize SciDeBERTa-CS
        self.tokenizer = AutoTokenizer.from_pretrained("KISTI-AI/scideberta-cs")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "KISTI-AI/scideberta-cs"
        )

        # Ensure NLTK resources
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are downloaded"""
        resources = ["punkt", "punkt_tab"]
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                self.logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)

    def extract_sections(self, pdf_path: str) -> Dict[str, PaperSection]:
        """Extract sections using combined PubLayNet and SciDeBERTa approach"""
        doc = fitz.open(pdf_path)

        # Use text-only approach by default to avoid CUDA errors
        # Only attempt PubLayNet if explicitly configured
        use_publaynet = os.environ.get("USE_PUBLAYNET", "false").lower() == "true"

        if (
            use_publaynet
            and pdf2image_available
            and self.publaynet.predictor is not None
        ):
            try:
                page_images = convert_from_path(pdf_path, dpi=300)
                # Convert PIL images to OpenCV format
                cv_images = [
                    cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    for img in page_images
                ]
                # Process pages in parallel
                results = self._process_pages_parallel(doc, cv_images)
                # Merge results into sections
                sections = self._merge_sections(results, doc)
            except Exception as e:
                self.logger.error(
                    f"Error in PubLayNet processing: {e}. Falling back to text-only processing."
                )
                sections = self._process_text_only(doc)
        else:
            self.logger.info("Using text-only processing for PDF")
            sections = self._process_text_only(doc)

        doc.close()
        return sections

    def _process_text_only(self, doc) -> Dict[str, PaperSection]:
        """Process PDF using text extraction only (fallback)"""
        full_text = ""
        page_info = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            page_info[page_num] = page_text
            full_text += f"\n\n=== PAGE {page_num + 1} ===\n\n" + page_text

        # Use SciDeBERTa to classify text sections
        paragraphs = re.split(r"\n\s*\n", full_text)
        sections = {}

        for para in paragraphs:
            if len(para.strip()) > 50:  # Skip very short paragraphs
                section_type = self._classify_text_section(para)
                if section_type in sections:
                    sections[section_type].content += "\n\n" + para
                else:
                    sections[section_type] = PaperSection(
                        title=section_type.capitalize(),
                        content=para,
                        section_type=section_type,
                        start_page=0,
                        end_page=len(doc) - 1,
                        confidence=0.7,
                    )

        return sections

    def _process_pages_parallel(self, doc, cv_images):
        """Process pages in parallel using ThreadPoolExecutor"""
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for page_num, (pdf_page, cv_image) in enumerate(zip(doc, cv_images)):
                futures.append(
                    executor.submit(
                        self._process_single_page, page_num, pdf_page, cv_image
                    )
                )

            for future in as_completed(futures):
                results.append(future.result())

        return results

    def _process_single_page(self, page_num: int, pdf_page, cv_image):
        """Process a single page to extract layout and text"""
        # Get layout detections
        detections = self.publaynet.process_page(cv_image)

        # Extract text from detected regions
        page_data = {"page_num": page_num, "detections": [], "extracted_sections": []}

        for detection in detections:
            detection.page_num = page_num
            text = self.publaynet.extract_text_from_region(pdf_page, detection.bbox)

            # Classify text using SciDeBERTa if it's substantial
            if len(text.split()) > 20:  # Only classify meaningful text
                section_type = self._classify_text_section(text)
                page_data["extracted_sections"].append(
                    {
                        "text": text,
                        "section_type": section_type,
                        "bbox": detection.bbox,
                        "class_name": detection.class_name,
                        "score": detection.score,
                    }
                )

            page_data["detections"].append(detection)

        return page_data

    def _classify_text_section(self, text: str) -> str:
        """Use SciDeBERTa to classify text into sections"""
        # Check if text matches section keywords first
        for section, keywords in self.publaynet.section_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower()[:100]:  # Check first 100 chars
                    return section

        # Truncate text to avoid token length issues
        truncated_text = " ".join(text.split()[:300])  # Limit to 300 words

        # Use SciDeBERTa for classification
        inputs = self.tokenizer(
            truncated_text, return_tensors="pt", truncation=True, max_length=512
        )

        # Ensure tensor is on CPU to avoid CUDA errors
        device = "cpu"  # Force CPU for classification to avoid CUDA issues
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.to(device)(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()

        # Map to section type
        section_types = [
            "abstract",
            "introduction",
            "methods",
            "results",
            "discussion",
            "conclusion",
        ]
        if predicted_class < len(section_types):
            return section_types[predicted_class]
        else:
            return "other"

    def _merge_sections(self, page_results: List[Dict], doc) -> Dict[str, PaperSection]:
        """Merge page results into complete sections"""
        sections = {}

        # Sort results by page number
        sorted_results = sorted(page_results, key=lambda x: x["page_num"])

        # Group consecutive sections
        for page_data in sorted_results:
            for section_info in page_data["extracted_sections"]:
                section_type = section_info["section_type"]

                if section_type not in sections:
                    sections[section_type] = PaperSection(
                        title=section_type.capitalize(),
                        content=section_info["text"],
                        section_type=section_type,
                        start_page=page_data["page_num"],
                        end_page=page_data["page_num"],
                        confidence=section_info["score"],
                        bounding_box=section_info["bbox"],
                    )
                else:
                    # Append to existing section
                    sections[section_type].content += "\n\n" + section_info["text"]
                    sections[section_type].end_page = page_data["page_num"]

        return sections


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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Device set to use {device}")

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
            # Split long texts into smaller chunks to avoid CUDA errors
            max_words = 400  # Reduced from 800 to avoid CUDA memory issues

            device = "cpu"  # Force CPU for summarization to avoid CUDA issues
            self.summarizer.model = self.summarizer.model.to(device)
            self.summarizer.device = -1  # Force CPU

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

        # Extract key concepts
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


class ImprovedQAGenerator:
    """Enhanced QA generation using SciDeBERTa for keywords"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize SciDeBERTa for embedding and keyword extraction
        self.tokenizer = AutoTokenizer.from_pretrained("KISTI-AI/scideberta-cs")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "KISTI-AI/scideberta-cs"
        )

        # Initialize QA pipeline for question generation and answering
        self.qa_pipeline = pipeline(
            "question-answering", model="deepset/roberta-base-squad2"
        )

        # Ensure NLTK resources
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are downloaded"""
        resources = ["punkt", "punkt_tab"]
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                self.logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)

    def extract_keywords(self, sections: Dict[str, PaperSection]) -> List[Dict]:
        """Extract key concepts and provide additional information"""
        keywords_with_info = []

        # Combine important sections (limit text size to avoid CUDA errors)
        important_texts = []
        for section_type in ["abstract", "results", "conclusion"]:
            if section_type in sections:
                # Only take first 300 words from each section
                section_words = sections[section_type].content.split()[:300]
                important_texts.append(" ".join(section_words))

        all_text = " ".join(important_texts)

        # Use TF-IDF to find important terms
        sentences = nltk.sent_tokenize(all_text)
        if not sentences:
            return []

        vectorizer = TfidfVectorizer(
            max_features=20, stop_words="english", ngram_range=(1, 3)
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)

            feature_names = vectorizer.get_feature_names_out()
            top_indices = tfidf_matrix.sum(axis=0).A1.argsort()[-10:][::-1]

            # Force QA pipeline to CPU to avoid CUDA errors
            self.qa_pipeline.device = -1
            self.qa_pipeline.model = self.qa_pipeline.model.to("cpu")

            # Get top keywords and their context
            for idx in top_indices:
                keyword = feature_names[idx]

                # Find sentences containing this keyword
                context_sentences = [
                    sent for sent in sentences if keyword.lower() in sent.lower()
                ]

                # Get additional information using QA
                if context_sentences:
                    # Limit context size to avoid token length issues
                    context = " ".join(context_sentences[:1])  # Use just 1 sentence

                    # Ask for definition/explanation
                    qa_result = self.qa_pipeline(
                        {"question": f"What is {keyword}?", "context": context}
                    )

                    keywords_with_info.append(
                        {
                            "keyword": keyword,
                            "context": context,
                            "definition": qa_result["answer"],
                            "confidence": qa_result["score"],
                        }
                    )
        except ValueError:
            return []

        return keywords_with_info

    def generate_quiz(
        self, sections: Dict[str, PaperSection], num_questions: int = 5
    ) -> List[Dict]:
        """Generate quiz questions using SciDeBERTa for scoring"""
        quiz = []

        # Prioritize sections
        prioritized_sections = self._prioritize_sections(sections)

        for section_type, section in prioritized_sections:
            if len(quiz) >= num_questions:
                break

            # Extract key sentences
            sentences = nltk.sent_tokenize(section.content)

            # Score sentences using TF-IDF
            if sentences:
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(sentences)
                sentence_scores = tfidf_matrix.sum(axis=1).A1

                # Get top sentences
                top_indices = sentence_scores.argsort()[-3:][::-1]

                for idx in top_indices:
                    if len(quiz) >= num_questions:
                        break

                    sentence = sentences[idx]

                    # Generate simple questions based on the sentence
                    question = self._create_simple_question(sentence)

                    quiz.append(
                        {
                            "question": question,
                            "answer": sentence,
                            "context": sentence,
                            "section": section_type,
                            "confidence": section.confidence,
                        }
                    )

        return quiz

    def _create_simple_question(self, sentence: str) -> str:
        """Create a simple question from a sentence"""
        # Simple heuristic: look for key terms and create questions
        words = sentence.split()
        if len(words) > 5:
            # Common question templates
            templates = [
                f"What does the research say about {' '.join(words[:5])}?",
                f"Can you explain {' '.join(words[:4])}?",
                f"What is important about {' '.join(words[:5])}?",
            ]
            return templates[0]
        else:
            return f"What is mentioned in this section?"

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

        prioritized.sort(reverse=True, key=lambda x: x[0])
        return [(x[1], x[2]) for x in prioritized]


class EnhancedAnalysisPipeline:
    """Main pipeline with PubLayNet integration"""

    def __init__(self, output_dir: str = "output"):
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.segmenter = EnhancedSegmenter()
        self.summarizer = Summarizer()
        self.qa_generator = ImprovedQAGenerator()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_paper(self, pdf_path: str) -> Dict:
        """Process a paper with enhanced layout and content analysis"""
        self.logger.info(f"Processing paper: {pdf_path}")

        try:
            # 1. Extract sections using PubLayNet + SciDeBERTa
            sections = self.segmenter.extract_sections(pdf_path)

            # 2. Summarize sections
            summary = self.summarizer.create_paper_summary(sections)

            # 3. Extract keywords with context
            keywords_info = self.qa_generator.extract_keywords(sections)

            # 4. Generate quiz
            quiz = self.qa_generator.generate_quiz(sections)

            # 5. Compile results
            results = {
                "paper_title": self._extract_title(sections),
                "sections": {
                    k: {
                        "content": v.content,
                        "start_page": v.start_page,
                        "end_page": v.end_page,
                        "confidence": v.confidence,
                        "bounding_box": v.bounding_box,
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
                "keywords": keywords_info,
                "quiz": quiz,
            }

            # Save results
            output_file = (
                self.output_dir / f"{Path(pdf_path).stem}_enhanced_analysis.json"
            )
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Results saved to: {output_file}")
            return results

        except Exception as e:
            self.logger.error(f"Error processing paper: {e}")
            raise

    def _extract_title(self, sections: Dict[str, PaperSection]) -> str:
        """Extract paper title using PubLayNet detections"""
        # Title is often detected as a separate class by PubLayNet
        # or found in the abstract section
        if "abstract" in sections:
            text = sections["abstract"].content
            lines = text.split("\n")

            for line in lines:
                clean_line = line.strip()
                if clean_line and len(clean_line.split()) > 3 and len(clean_line) < 200:
                    if (
                        sum(1 for c in clean_line if c.isupper())
                        > len(clean_line) * 0.1
                    ):
                        return clean_line

        return "Unknown Title"


# Example usage
def main():
    """Run the fixed pipeline with safer defaults"""
    import argparse

    parser = argparse.ArgumentParser(description="Process scientific papers safely")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Try to use GPU (may cause errors)"
    )
    parser.add_argument(
        "--use-publaynet",
        action="store_true",
        help="Try to use PubLayNet (may cause errors)",
    )

    args = parser.parse_args()

    # Override environment variables if requested
    if args.use_gpu:
        logger.info("Attempting to use GPU as requested (may cause errors)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.is_available = lambda: True

    if args.use_publaynet:
        logger.info("Attempting to use PubLayNet as requested (may cause errors)")
        os.environ["USE_PUBLAYNET"] = "true"

    try:
        # Initialize pipeline with safer settings
        logger.info("Initializing pipeline")
        pipeline = EnhancedAnalysisPipeline(output_dir=args.output_dir)

        # Process paper
        logger.info(f"Processing paper: {args.pdf_path}")
        results = pipeline.process_paper(args.pdf_path)

        # Display summary
        print("\n=== Paper Analysis Summary ===")
        print(f"Title: {results['paper_title']}")

        print("\nKey Sections:")
        for section_type in results["sections"]:
            print(f"- {section_type.capitalize()}")

        if results.get("keywords"):
            print("\nTop 3 Keywords:")
            for kw in results["keywords"][:3]:
                print(f"- {kw['keyword']}: {kw['definition']}")

        print(
            f"\nFull results saved to: {args.output_dir}/{os.path.basename(args.pdf_path).split('.')[0]}_enhanced_analysis.json"
        )

    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        print(f"\nERROR: {str(e)}")
        print("\nTry running with --use-gpu=False or check logs for details")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
