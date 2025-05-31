import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import fitz  # PyMuPDF
import os
import json
from typing import List, Dict, Tuple, Optional
import requests
from pathlib import Path
import logging


class PubLayNetFigureExtractor:
    """
    Enhanced figure extractor using PubLayNet model for scientific papers.
    Extracts figures, tables, text blocks, and other layout elements.
    """

    # PubLayNet class mapping
    PUBLAYNET_CLASSES = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}

    def __init__(
        self, model_path: Optional[str] = None, confidence_threshold: float = 0.5
    ):
        """
        Initialize the PubLayNet figure extractor.

        Args:
            model_path: Path to pre-trained model. If None, downloads from HuggingFace
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.cfg = None
        self.predictor = None
        self.logger = logging.getLogger(__name__)

        # Setup the model
        self._setup_model(model_path)

    def _setup_model(self, model_path: Optional[str] = None):
        """Setup the Detectron2 model with PubLayNet configuration."""
        try:
            # Configure Detectron2
            self.cfg = get_cfg()

            # Use Mask R-CNN with ResNeXt-101 backbone (best for PubLayNet)
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
                )
            )

            # Set number of classes for PubLayNet (5 classes)
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

            # Set confidence threshold
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold

            # Download and set model weights
            if model_path is None:
                model_path = self._download_publaynet_model()

            self.cfg.MODEL.WEIGHTS = model_path

            # Set device
            self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            # Create predictor
            self.predictor = DefaultPredictor(self.cfg)

            self.logger.info(f"PubLayNet model initialized on {self.cfg.MODEL.DEVICE}")

        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            raise

    def _download_publaynet_model(self) -> str:
        """Download pre-trained PubLayNet model from HuggingFace."""
        model_dir = Path("models/publaynet")
        model_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / "model_final.pth"

        if not model_file.exists():
            self.logger.info("Downloading PubLayNet model...")

            # URL for the nlpconnect model (Faster R-CNN version)
            model_url = "https://huggingface.co/nlpconnect/PubLayNet-faster_rcnn_R_50_FPN_3x/resolve/main/model_final.pth"

            try:
                response = requests.get(model_url, stream=True)
                response.raise_for_status()

                with open(model_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                self.logger.info(f"Model downloaded to {model_file}")

            except Exception as e:
                self.logger.error(f"Failed to download model: {e}")
                # Fallback to detectron2 model zoo
                return model_zoo.get_checkpoint_url(
                    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
                )

        return str(model_file)

    def extract_from_pdf(
        self, pdf_path: str, output_dir: str = "extracted_figures"
    ) -> Dict:
        """
        Extract figures and layout elements from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted figures

        Returns:
            Dictionary containing extracted elements and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Open PDF
        pdf_document = fitz.open(pdf_path)

        results = {
            "pdf_path": pdf_path,
            "total_pages": len(pdf_document),
            "pages": [],
            "figures": [],
            "tables": [],
            "layout_analysis": [],
        }

        self.logger.info(f"Processing {len(pdf_document)} pages...")

        for page_num in range(len(pdf_document)):
            page_results = self._process_page(pdf_document, page_num, output_dir)
            results["pages"].append(page_results)

            # Aggregate results
            results["figures"].extend(page_results["figures"])
            results["tables"].extend(page_results["tables"])
            results["layout_analysis"].append(
                {"page": page_num + 1, "elements": page_results["layout_elements"]}
            )

        pdf_document.close()

        # Save results
        results_file = Path(output_dir) / "extraction_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Extraction complete. Results saved to {results_file}")
        return results

    def _process_page(self, pdf_document, page_num: int, output_dir: str) -> Dict:
        """Process a single PDF page for layout analysis and figure extraction."""
        page = pdf_document[page_num]

        # Convert page to image
        mat = fitz.Matrix(2.0, 2.0)  # High resolution
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")

        # Convert to OpenCV format
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run PubLayNet detection
        outputs = self.predictor(image)

        # Process detections
        page_results = {
            "page_number": page_num + 1,
            "figures": [],
            "tables": [],
            "layout_elements": [],
            "image_path": None,
        }

        if len(outputs["instances"]) > 0:
            # Extract predictions
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            classes = outputs["instances"].pred_classes.cpu().numpy()
            scores = outputs["instances"].scores.cpu().numpy()

            if hasattr(outputs["instances"], "pred_masks"):
                masks = outputs["instances"].pred_masks.cpu().numpy()
            else:
                masks = None

            # Process each detection
            for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
                if score < self.confidence_threshold:
                    continue

                element_type = self.PUBLAYNET_CLASSES[class_id]

                # Convert box coordinates (scaled back to original resolution)
                x1, y1, x2, y2 = box / 2.0  # Undo the 2x scaling

                element_info = {
                    "type": element_type,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(score),
                    "page": page_num + 1,
                }

                # Add mask if available
                if masks is not None:
                    element_info["has_mask"] = True

                page_results["layout_elements"].append(element_info)

                # Extract figures and tables
                if element_type == "figure":
                    figure_info = self._extract_figure(
                        page,
                        element_info,
                        page_num,
                        output_dir,
                        len(page_results["figures"]),
                    )
                    page_results["figures"].append(figure_info)

                elif element_type == "table":
                    table_info = self._extract_table(
                        page,
                        element_info,
                        page_num,
                        output_dir,
                        len(page_results["tables"]),
                    )
                    page_results["tables"].append(table_info)

            # Save annotated page image
            page_results["image_path"] = self._save_annotated_image(
                image, outputs, page_num, output_dir
            )

        return page_results

    def _extract_figure(
        self,
        page,
        element_info: Dict,
        page_num: int,
        output_dir: str,
        figure_index: int,
    ) -> Dict:
        """Extract and save a figure from the page."""
        x1, y1, x2, y2 = element_info["bbox"]

        # Create crop rectangle
        rect = fitz.Rect(x1, y1, x2, y2)

        # Extract figure as image
        mat = fitz.Matrix(3.0, 3.0)  # High resolution for figures
        pix = page.get_pixmap(matrix=mat, clip=rect)

        # Save figure
        figure_filename = f"figure_page{page_num+1}_{figure_index+1}.png"
        figure_path = Path(output_dir) / "figures" / figure_filename
        figure_path.parent.mkdir(parents=True, exist_ok=True)

        pix.save(str(figure_path))

        # Try to extract caption from nearby text
        caption = self._extract_figure_caption(page, element_info)

        figure_info = {
            **element_info,
            "figure_id": f"fig_{page_num+1}_{figure_index+1}",
            "file_path": str(figure_path),
            "caption": caption,
            "dimensions": {"width": pix.width, "height": pix.height},
        }

        return figure_info

    def _extract_table(
        self, page, element_info: Dict, page_num: int, output_dir: str, table_index: int
    ) -> Dict:
        """Extract and save a table from the page."""
        x1, y1, x2, y2 = element_info["bbox"]

        # Create crop rectangle
        rect = fitz.Rect(x1, y1, x2, y2)

        # Extract table as image
        mat = fitz.Matrix(3.0, 3.0)  # High resolution for tables
        pix = page.get_pixmap(matrix=mat, clip=rect)

        # Save table
        table_filename = f"table_page{page_num+1}_{table_index+1}.png"
        table_path = Path(output_dir) / "tables" / table_filename
        table_path.parent.mkdir(parents=True, exist_ok=True)

        pix.save(str(table_path))

        # Try to extract table text content
        table_text = page.get_textbox(rect)

        # Try to extract caption
        caption = self._extract_table_caption(page, element_info)

        table_info = {
            **element_info,
            "table_id": f"table_{page_num+1}_{table_index+1}",
            "file_path": str(table_path),
            "caption": caption,
            "text_content": table_text,
            "dimensions": {"width": pix.width, "height": pix.height},
        }

        return table_info

    def _extract_figure_caption(self, page, element_info: Dict) -> str:
        """Extract figure caption from text near the figure."""
        x1, y1, x2, y2 = element_info["bbox"]

        # Look for caption below the figure
        caption_rect = fitz.Rect(x1, y2, x2, y2 + 100)  # 100 points below
        caption_text = page.get_textbox(caption_rect)

        # Look for "Figure" or "Fig." in the text
        if caption_text and (
            "figure" in caption_text.lower() or "fig." in caption_text.lower()
        ):
            return caption_text.strip()

        # Look above the figure as fallback
        caption_rect = fitz.Rect(x1, max(0, y1 - 50), x2, y1)
        caption_text = page.get_textbox(caption_rect)

        if caption_text and (
            "figure" in caption_text.lower() or "fig." in caption_text.lower()
        ):
            return caption_text.strip()

        return ""

    def _extract_table_caption(self, page, element_info: Dict) -> str:
        """Extract table caption from text near the table."""
        x1, y1, x2, y2 = element_info["bbox"]

        # Look for caption above the table (more common for tables)
        caption_rect = fitz.Rect(x1, max(0, y1 - 50), x2, y1)
        caption_text = page.get_textbox(caption_rect)

        if caption_text and "table" in caption_text.lower():
            return caption_text.strip()

        # Look below as fallback
        caption_rect = fitz.Rect(x1, y2, x2, y2 + 50)
        caption_text = page.get_textbox(caption_rect)

        if caption_text and "table" in caption_text.lower():
            return caption_text.strip()

        return ""

    def _save_annotated_image(
        self, image, outputs, page_num: int, output_dir: str
    ) -> str:
        """Save page image with layout annotations."""
        # Create visualizer
        v = Visualizer(
            image[:, :, ::-1], metadata=None, scale=1.2  # Convert BGR to RGB
        )

        # Add detections
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Save annotated image
        annotated_filename = f"page_{page_num+1}_annotated.png"
        annotated_path = Path(output_dir) / "annotated_pages" / annotated_filename
        annotated_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert back to BGR for saving
        annotated_image = v.get_image()[:, :, ::-1]
        cv2.imwrite(str(annotated_path), annotated_image)

        return str(annotated_path)

    def analyze_layout_structure(self, extraction_results: Dict) -> Dict:
        """Analyze the overall document structure based on layout elements."""
        structure_analysis = {
            "document_type": "scientific_paper",
            "total_figures": len(extraction_results["figures"]),
            "total_tables": len(extraction_results["tables"]),
            "page_structure": [],
            "figure_distribution": {},
            "table_distribution": {},
        }

        # Analyze page-by-page structure
        for page_data in extraction_results["pages"]:
            page_num = page_data["page_number"]

            elements_by_type = {}
            for element in page_data["layout_elements"]:
                elem_type = element["type"]
                if elem_type not in elements_by_type:
                    elements_by_type[elem_type] = 0
                elements_by_type[elem_type] += 1

            structure_analysis["page_structure"].append(
                {
                    "page": page_num,
                    "elements": elements_by_type,
                    "figures": len(page_data["figures"]),
                    "tables": len(page_data["tables"]),
                }
            )

            # Track distributions
            if len(page_data["figures"]) > 0:
                structure_analysis["figure_distribution"][page_num] = len(
                    page_data["figures"]
                )

            if len(page_data["tables"]) > 0:
                structure_analysis["table_distribution"][page_num] = len(
                    page_data["tables"]
                )

        return structure_analysis


# Example usage and integration function
def integrate_with_existing_analyzer(
    pdf_path: str, output_dir: str = "extracted_content"
):
    """
    Integration function to use with your existing paper_analyzer2.py
    """
    # Initialize the enhanced figure extractor
    extractor = PubLayNetFigureExtractor(confidence_threshold=0.7)

    # Extract figures and layout elements
    results = extractor.extract_from_pdf(pdf_path, output_dir)

    # Analyze document structure
    structure_analysis = extractor.analyze_layout_structure(results)

    # Return combined results for integration with your existing system
    return {
        "extraction_results": results,
        "structure_analysis": structure_analysis,
        "figures_extracted": len(results["figures"]),
        "tables_extracted": len(results["tables"]),
        "output_directory": output_dir,
    }


def extract_figures_with_publaynet(pdf_path: str, output_dir: str = None) -> dict:
    """
    Direct replacement for your existing figure extraction function

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for extracted figures

    Returns:
        Dictionary with extracted figures and metadata
    """
    if output_dir is None:
        # Create output directory next to the PDF
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(pdf_dir, f"{pdf_name}_figures")

    try:
        # Use PubLayNet for figure extraction
        extractor = PubLayNetFigureExtractor(confidence_threshold=0.7)
        results = extractor.extract_from_pdf(pdf_path, output_dir)

        return {
            "success": True,
            "figures": results["figures"],
            "tables": results["tables"],
            "total_figures": len(results["figures"]),
            "total_tables": len(results["tables"]),
            "output_dir": output_dir,
            "method": "publaynet",
        }

    except Exception as e:
        print(f"PubLayNet extraction failed: {e}")
        return extract_figures_fallback(pdf_path, output_dir)


def extract_figures_fallback(pdf_path: str, output_dir: str) -> dict:
    """
    Fallback figure extraction method using basic PyMuPDF
    """
    import fitz
    from pathlib import Path

    doc = fitz.open(pdf_path)
    figures = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    figure_filename = f"figure_page{page_num+1}_{img_index+1}.png"
                    figure_path = Path(output_dir) / figure_filename
                    pix.save(str(figure_path))

                    figures.append(
                        {
                            "figure_id": f"fig_{page_num+1}_{img_index+1}",
                            "file_path": str(figure_path),
                            "page": page_num + 1,
                            "type": "figure",
                            "confidence": 1.0,  # No confidence score for fallback method
                            "method": "fallback",
                            "bbox": [0, 0, pix.width, pix.height],  # Approximate bbox
                            "caption": "",  # No caption extraction in fallback
                        }
                    )

                pix = None

            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num+1}: {e}")

    doc.close()

    return {
        "success": True,
        "figures": figures,
        "tables": [],
        "total_figures": len(figures),
        "total_tables": 0,
        "output_dir": output_dir,
        "method": "fallback",
    }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Extract figures using PubLayNet")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--output", "-o", default="extracted_figures", help="Output directory"
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.7, help="Confidence threshold"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run extraction
    results = integrate_with_existing_analyzer(args.pdf_path, args.output)

    print(f"\nExtraction completed!")
    print(f"Figures extracted: {results['figures_extracted']}")
    print(f"Tables extracted: {results['tables_extracted']}")
    print(f"Output directory: {results['output_directory']}")
