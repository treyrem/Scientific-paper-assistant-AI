import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
import fitz  # PyMuPDF
import os
import json
from typing import List, Dict, Tuple, Optional
import requests
from pathlib import Path
import logging


class HybridFigureExtractor:
    """
    Hybrid extractor that combines multiple methods for maximum figure detection.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        self.detection_methods = []

        # Initialize available detection methods
        self._setup_detection_methods()

    def _setup_detection_methods(self):
        """Setup multiple detection methods as fallbacks."""
        self.detection_methods = [
            self._detectron2_method,
            self._layout_analysis_method,
            self._image_extraction_method,
            self._contour_detection_method,
        ]

        self.logger.info(f"Initialized {len(self.detection_methods)} detection methods")

    def extract_from_pdf(
        self, pdf_path: str, output_dir: str = "hybrid_extraction"
    ) -> Dict:
        """Extract using all available methods."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        pdf_document = fitz.open(pdf_path)

        all_results = {
            "pdf_path": pdf_path,
            "total_pages": len(pdf_document),
            "figures": [],
            "tables": [],
            "methods_used": [],
            "method_results": {},
        }

        self.logger.info(
            f"Processing {len(pdf_document)} pages with hybrid approach..."
        )

        # Try each detection method
        for method_idx, method in enumerate(self.detection_methods):
            method_name = method.__name__.replace("_", " ").title()

            try:
                self.logger.info(f"Trying method {method_idx + 1}: {method_name}")

                method_results = method(pdf_document, output_dir)

                if method_results["figures"] or method_results["tables"]:
                    all_results["methods_used"].append(method_name)
                    all_results["method_results"][method_name] = method_results

                    # Add unique results (avoid duplicates)
                    new_figures = self._filter_unique_detections(
                        all_results["figures"], method_results["figures"]
                    )
                    new_tables = self._filter_unique_detections(
                        all_results["tables"], method_results["tables"]
                    )

                    all_results["figures"].extend(new_figures)
                    all_results["tables"].extend(new_tables)

                    self.logger.info(
                        f"✅ {method_name} found {len(new_figures)} new figures, {len(new_tables)} new tables"
                    )
                else:
                    self.logger.info(f"❌ {method_name} found nothing")

            except Exception as e:
                self.logger.warning(f"❌ {method_name} failed: {e}")
                continue

        pdf_document.close()

        # Save results
        results_file = Path(output_dir) / "hybrid_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        total_figures = len(all_results["figures"])
        total_tables = len(all_results["tables"])

        self.logger.info(
            f"🎯 Hybrid extraction complete: {total_figures} figures, {total_tables} tables using {len(all_results['methods_used'])} methods"
        )

        return all_results

    def _detectron2_method(self, pdf_document, output_dir: str) -> Dict:
        """Method 1: Try Detectron2 with standard models."""
        results = {"figures": [], "tables": []}

        try:
            # Setup basic Detectron2 model
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # COCO classes
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            )
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            predictor = DefaultPredictor(cfg)

            # Process a few pages to test
            for page_num in range(min(3, len(pdf_document))):
                page = pdf_document[page_num]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                outputs = predictor(image)

                if len(outputs["instances"]) > 0:
                    # Look for object classes that might be figures/tables
                    # COCO classes: computer, tv, book, etc.
                    for i, class_id in enumerate(outputs["instances"].pred_classes):
                        if class_id.item() in [62, 63, 72, 73]:  # book, laptop, tv, etc
                            box = (
                                outputs["instances"]
                                .pred_boxes[i]
                                .tensor.cpu()
                                .numpy()[0]
                            )
                            x1, y1, x2, y2 = box / 2.0

                            if (x2 - x1) * (y2 - y1) > 10000:  # Reasonable size
                                results["figures"].append(
                                    {
                                        "figure_id": f"detectron2_fig_{page_num+1}_{i+1}",
                                        "bbox": [
                                            float(x1),
                                            float(y1),
                                            float(x2),
                                            float(y2),
                                        ],
                                        "page": page_num + 1,
                                        "confidence": float(
                                            outputs["instances"].scores[i]
                                        ),
                                        "method": "detectron2",
                                    }
                                )

        except Exception as e:
            self.logger.warning(f"Detectron2 method failed: {e}")

        return results

    def _layout_analysis_method(self, pdf_document, output_dir: str) -> Dict:
        """Method 2: Layout analysis using text density."""
        results = {"figures": [], "tables": []}

        try:
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_rect = page.rect

                # Divide page into grid
                grid_size = 6
                cell_width = page_rect.width / grid_size
                cell_height = page_rect.height / grid_size

                for row in range(grid_size):
                    for col in range(grid_size):
                        cell_rect = fitz.Rect(
                            col * cell_width,
                            row * cell_height,
                            (col + 1) * cell_width,
                            (row + 1) * cell_height,
                        )

                        # Check text density
                        cell_text = page.get_textbox(cell_rect)
                        text_density = (
                            len(cell_text) / (cell_width * cell_height)
                            if cell_text
                            else 0
                        )

                        # Low text density might indicate figure/table
                        if (
                            text_density < 0.008
                            and cell_width > 80
                            and cell_height > 60
                        ):
                            # Render area to check if it has content
                            mat = fitz.Matrix(2.0, 2.0)
                            pix = page.get_pixmap(matrix=mat, clip=cell_rect)

                            if pix.width > 100 and pix.height > 80:
                                # Check pixel variance (content vs blank)
                                img_data = pix.tobytes("png")
                                if len(img_data) > 0:
                                    nparr = np.frombuffer(img_data, np.uint8)
                                    if np.std(nparr) > 15:  # Has some content variation

                                        # Save the area
                                        figure_dir = Path(output_dir) / "figures"
                                        figure_dir.mkdir(exist_ok=True)
                                        figure_path = (
                                            figure_dir
                                            / f"layout_fig_page{page_num+1}_r{row}c{col}.png"
                                        )
                                        pix.save(str(figure_path))

                                        results["figures"].append(
                                            {
                                                "figure_id": f"layout_fig_{page_num+1}_{row}_{col}",
                                                "file_path": str(figure_path),
                                                "bbox": [
                                                    cell_rect.x0,
                                                    cell_rect.y0,
                                                    cell_rect.x1,
                                                    cell_rect.y1,
                                                ],
                                                "page": page_num + 1,
                                                "confidence": 0.7,
                                                "method": "layout_analysis",
                                                "dimensions": {
                                                    "width": pix.width,
                                                    "height": pix.height,
                                                },
                                            }
                                        )

        except Exception as e:
            self.logger.warning(f"Layout analysis method failed: {e}")

        return results

    def _image_extraction_method(self, pdf_document, output_dir: str) -> Dict:
        """Method 3: Extract embedded images with smart filtering."""
        results = {"figures": [], "tables": []}

        try:
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)

                        # Smart filtering for real figures
                        if (
                            pix.n - pix.alpha < 4  # RGB or grayscale
                            and pix.width > 150  # Minimum width
                            and pix.height > 120  # Minimum height
                            and pix.width * pix.height > 30000  # Minimum area
                            and pix.width / pix.height < 10  # Not too wide
                            and pix.height / pix.width < 10
                        ):  # Not too tall

                            # Additional quality check
                            img_data = pix.tobytes("png")
                            if len(img_data) > 5000:  # At least 5KB

                                figure_dir = Path(output_dir) / "figures"
                                figure_dir.mkdir(exist_ok=True)
                                figure_path = (
                                    figure_dir
                                    / f"embedded_fig_page{page_num+1}_{img_index+1}.png"
                                )
                                pix.save(str(figure_path))

                                results["figures"].append(
                                    {
                                        "figure_id": f"embedded_fig_{page_num+1}_{img_index+1}",
                                        "file_path": str(figure_path),
                                        "bbox": [
                                            0,
                                            0,
                                            pix.width,
                                            pix.height,
                                        ],  # Approximate
                                        "page": page_num + 1,
                                        "confidence": 0.9,  # High confidence for embedded images
                                        "method": "embedded_image",
                                        "dimensions": {
                                            "width": pix.width,
                                            "height": pix.height,
                                        },
                                    }
                                )

                        pix = None

                    except Exception as e:
                        continue

        except Exception as e:
            self.logger.warning(f"Image extraction method failed: {e}")

        return results

    def _contour_detection_method(self, pdf_document, output_dir: str) -> Dict:
        """Method 4: OpenCV contour detection for figures."""
        results = {"figures": [], "tables": []}

        try:
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Render page at high resolution
                mat = fitz.Matrix(3.0, 3.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Edge detection
                    edges = cv2.Canny(gray, 50, 150)

                    # Find contours
                    contours, _ = cv2.findContours(
                        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for i, contour in enumerate(contours):
                        # Filter contours by area and aspect ratio
                        area = cv2.contourArea(contour)
                        if area > 50000:  # Minimum area for figures

                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / h

                            # Reasonable aspect ratio for figures
                            if 0.2 < aspect_ratio < 5.0 and w > 200 and h > 150:

                                # Scale back coordinates
                                x1, y1 = x / 3.0, y / 3.0
                                x2, y2 = (x + w) / 3.0, (y + h) / 3.0

                                # Extract the region
                                rect = fitz.Rect(x1, y1, x2, y2)
                                region_pix = page.get_pixmap(
                                    matrix=fitz.Matrix(4.0, 4.0), clip=rect
                                )

                                if region_pix.width > 300 and region_pix.height > 200:
                                    figure_dir = Path(output_dir) / "figures"
                                    figure_dir.mkdir(exist_ok=True)
                                    figure_path = (
                                        figure_dir
                                        / f"contour_fig_page{page_num+1}_{i+1}.png"
                                    )
                                    region_pix.save(str(figure_path))

                                    results["figures"].append(
                                        {
                                            "figure_id": f"contour_fig_{page_num+1}_{i+1}",
                                            "file_path": str(figure_path),
                                            "bbox": [
                                                float(x1),
                                                float(y1),
                                                float(x2),
                                                float(y2),
                                            ],
                                            "page": page_num + 1,
                                            "confidence": 0.6,
                                            "method": "contour_detection",
                                            "dimensions": {
                                                "width": region_pix.width,
                                                "height": region_pix.height,
                                            },
                                        }
                                    )

        except Exception as e:
            self.logger.warning(f"Contour detection method failed: {e}")

        return results

    def _filter_unique_detections(
        self, existing: List[Dict], new: List[Dict]
    ) -> List[Dict]:
        """Filter out detections that are too similar to existing ones."""
        unique_new = []

        for new_item in new:
            is_duplicate = False
            new_bbox = new_item.get("bbox", [0, 0, 0, 0])

            for existing_item in existing:
                existing_bbox = existing_item.get("bbox", [0, 0, 0, 0])

                # Check overlap
                iou = self._calculate_iou(new_bbox, existing_bbox)
                if iou > 0.3:  # 30% overlap threshold
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_new.append(new_item)

        return unique_new

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU)."""
        if len(box1) < 4 or len(box2) < 4:
            return 0.0

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


# Main integration function
def extract_figures_with_publaynet(pdf_path: str, output_dir: str = None) -> dict:
    """
    Extract figures using hybrid approach with multiple methods.
    """
    if output_dir is None:
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(pdf_dir, f"{pdf_name}_hybrid_figures")

    try:
        # Use the hybrid extractor
        extractor = HybridFigureExtractor(confidence_threshold=0.5)
        results = extractor.extract_from_pdf(pdf_path, output_dir)

        return {
            "success": True,
            "figures": results["figures"],
            "tables": results["tables"],
            "total_figures": len(results["figures"]),
            "total_tables": len(results["tables"]),
            "output_dir": output_dir,
            "method": "hybrid_extraction",
            "methods_used": results.get("methods_used", []),
            "method_results": results.get("method_results", {}),
        }

    except Exception as e:
        print(f"⚠️  Hybrid extraction failed: {e}")
        return {
            "success": False,
            "figures": [],
            "tables": [],
            "total_figures": 0,
            "total_tables": 0,
            "output_dir": output_dir,
            "method": "failed",
            "error": str(e),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Figure Extraction")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--output", "-o", default="hybrid_extraction", help="Output directory"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"🚀 Starting hybrid figure extraction for: {args.pdf_path}")

    results = extract_figures_with_publaynet(args.pdf_path, args.output)

    print(f"\n🎯 Hybrid Results:")
    print(f"✅ Success: {results['success']}")
    print(f"🖼️  Figures: {results['total_figures']}")
    print(f"📋 Tables: {results['total_tables']}")
    print(f"📁 Output: {results['output_dir']}")
    print(f"🔧 Methods used: {', '.join(results.get('methods_used', []))}")

    if results.get("method_results"):
        print(f"\n📊 Method breakdown:")
        for method, method_result in results["method_results"].items():
            figures = len(method_result.get("figures", []))
            tables = len(method_result.get("tables", []))
            print(f"   {method}: {figures} figures, {tables} tables")
