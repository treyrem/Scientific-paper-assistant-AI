# Improved Figure Extractor with Better Filtering
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


class ImprovedFigureExtractor:
    """
    Improved figure extractor with better filtering and validation.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        self.detection_methods = []
        self._setup_detection_methods()

    def _setup_detection_methods(self):
        """Setup detection methods with improved filtering."""
        self.detection_methods = [
            self._embedded_image_method,  # Most reliable first
            self._layout_analysis_method_improved,
            self._contour_detection_method_improved,
        ]
        self.logger.info(f"Initialized {len(self.detection_methods)} detection methods")

    def extract_from_pdf(self, pdf_path: str, output_dir: str = "improved_extraction") -> Dict:
        """Extract using improved methods with better filtering."""
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

        self.logger.info(f"Processing {len(pdf_document)} pages with improved methods...")

        # Try each detection method
        for method_idx, method in enumerate(self.detection_methods):
            method_name = method.__name__.replace("_", " ").title()

            try:
                self.logger.info(f"Trying method {method_idx + 1}: {method_name}")
                method_results = method(pdf_document, output_dir)

                if method_results["figures"] or method_results["tables"]:
                    all_results["methods_used"].append(method_name)
                    all_results["method_results"][method_name] = method_results

                    # Add unique results with quality filtering
                    new_figures = self._filter_and_validate_figures(
                        all_results["figures"], method_results["figures"]
                    )
                    new_tables = self._filter_and_validate_tables(
                        all_results["tables"], method_results["tables"]
                    )

                    all_results["figures"].extend(new_figures)
                    all_results["tables"].extend(new_tables)

                    self.logger.info(
                        f"✅ {method_name} found {len(new_figures)} quality figures, {len(new_tables)} tables"
                    )
                else:
                    self.logger.info(f"❌ {method_name} found nothing")

            except Exception as e:
                self.logger.warning(f"❌ {method_name} failed: {e}")
                continue

        pdf_document.close()

        # Apply final quality filtering
        all_results["figures"] = self._final_figure_filtering(all_results["figures"])
        all_results["tables"] = self._final_table_filtering(all_results["tables"])

        # Save results
        results_file = Path(output_dir) / "improved_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        total_figures = len(all_results["figures"])
        total_tables = len(all_results["tables"])

        self.logger.info(
            f"🎯 Improved extraction complete: {total_figures} quality figures, {total_tables} tables"
        )

        return all_results

    def _embedded_image_method(self, pdf_document, output_dir: str) -> Dict:
        """Extract embedded images with strict quality filtering."""
        results = {"figures": [], "tables": []}

        try:
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)

                        # Strict quality filtering for embedded images
                        if self._is_quality_image(pix):
                            figure_dir = Path(output_dir) / "figures"
                            figure_dir.mkdir(exist_ok=True)
                            figure_path = (
                                figure_dir / f"embedded_fig_page{page_num+1}_{img_index+1}.png"
                            )
                            pix.save(str(figure_path))

                            # Get image position on page if possible
                            bbox = self._estimate_image_bbox(page, pix)

                            results["figures"].append({
                                "figure_id": f"embedded_fig_{page_num+1}_{img_index+1}",
                                "file_path": str(figure_path),
                                "bbox": bbox,
                                "page": page_num + 1,
                                "confidence": 0.95,  # High confidence for embedded images
                                "method": "embedded_image",
                                "dimensions": {"width": pix.width, "height": pix.height},
                                "file_size": len(pix.tobytes("png")),
                            })

                        pix = None

                    except Exception as e:
                        continue

        except Exception as e:
            self.logger.warning(f"Embedded image method failed: {e}")

        return results

    def _layout_analysis_method_improved(self, pdf_document, output_dir: str) -> Dict:
        """Improved layout analysis with better validation."""
        results = {"figures": [], "tables": []}

        try:
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_rect = page.rect

                # Use larger grid for better detection
                grid_size = 8
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

                        # Check if this region might contain a figure
                        if self._is_potential_figure_region(page, cell_rect):
                            # Render at higher resolution for quality check
                            mat = fitz.Matrix(3.0, 3.0)
                            pix = page.get_pixmap(matrix=mat, clip=cell_rect)

                            if self._validate_figure_content(pix):
                                figure_dir = Path(output_dir) / "figures"
                                figure_dir.mkdir(exist_ok=True)
                                figure_path = (
                                    figure_dir / f"layout_fig_page{page_num+1}_r{row}c{col}.png"
                                )
                                pix.save(str(figure_path))

                                results["figures"].append({
                                    "figure_id": f"layout_fig_{page_num+1}_{row}_{col}",
                                    "file_path": str(figure_path),
                                    "bbox": [cell_rect.x0, cell_rect.y0, cell_rect.x1, cell_rect.y1],
                                    "page": page_num + 1,
                                    "confidence": 0.75,
                                    "method": "layout_analysis_improved",
                                    "dimensions": {"width": pix.width, "height": pix.height},
                                })

        except Exception as e:
            self.logger.warning(f"Improved layout analysis failed: {e}")

        return results

    def _contour_detection_method_improved(self, pdf_document, output_dir: str) -> Dict:
        """Improved contour detection with better filtering."""
        results = {"figures": [], "tables": []}

        try:
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Render page at high resolution
                mat = fitz.Matrix(4.0, 4.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is not None:
                    # Improved contour detection
                    contours = self._detect_quality_contours(image)

                    for i, contour in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Scale back coordinates
                        x1, y1 = x / 4.0, y / 4.0
                        x2, y2 = (x + w) / 4.0, (y + h) / 4.0

                        # Extract and validate the region
                        rect = fitz.Rect(x1, y1, x2, y2)
                        region_pix = page.get_pixmap(matrix=fitz.Matrix(4.0, 4.0), clip=rect)

                        if self._validate_figure_content(region_pix):
                            figure_dir = Path(output_dir) / "figures"
                            figure_dir.mkdir(exist_ok=True)
                            figure_path = (
                                figure_dir / f"contour_fig_page{page_num+1}_{i+1}.png"
                            )
                            region_pix.save(str(figure_path))

                            results["figures"].append({
                                "figure_id": f"contour_fig_{page_num+1}_{i+1}",
                                "file_path": str(figure_path),
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "page": page_num + 1,
                                "confidence": 0.8,
                                "method": "contour_detection_improved",
                                "dimensions": {"width": region_pix.width, "height": region_pix.height},
                            })

        except Exception as e:
            self.logger.warning(f"Improved contour detection failed: {e}")

        return results

    def _is_quality_image(self, pix) -> bool:
        """Check if pixmap represents a quality figure."""
        # Size requirements
        if pix.width < 250 or pix.height < 200:
            return False
        
        # Aspect ratio check
        aspect_ratio = pix.width / pix.height
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            return False
        
        # Color depth check
        if pix.n - pix.alpha >= 4:  # CMYK or more
            return False
        
        # File size check (avoid tiny images)
        img_data = pix.tobytes("png")
        if len(img_data) < 10000:  # Less than 10KB
            return False
        
        # Content variance check
        try:
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            if np.std(img_array) < 20:  # Very low variance (likely blank)
                return False
        except:
            pass
        
        return True

    def _is_potential_figure_region(self, page, rect) -> bool:
        """Check if a region might contain a figure."""
        # Check text density
        text_in_region = page.get_textbox(rect)
        text_density = len(text_in_region) / (rect.width * rect.height) if text_in_region else 0
        
        # Low text density suggests non-text content
        if text_density > 0.01:  # Too much text
            return False
        
        # Size requirements
        if rect.width < 100 or rect.height < 80:
            return False
        
        return True

    def _validate_figure_content(self, pix) -> bool:
        """Validate that pixmap contains meaningful figure content."""
        if not pix or pix.width < 200 or pix.height < 150:
            return False
        
        try:
            # Convert to numpy array for analysis
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                return False
            
            # Check content variance (avoid blank regions)
            if np.std(img) < 15:
                return False
            
            # Check for edge content (figures usually have edges)
            edges = cv2.Canny(img, 50, 150)
            edge_ratio = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
            
            if edge_ratio < 0.01:  # Too few edges
                return False
            
            # Check aspect ratio
            aspect_ratio = pix.width / pix.height
            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                return False
            
            return True
            
        except Exception:
            return False

    def _detect_quality_contours(self, image):
        """Detect high-quality contours that might represent figures."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive threshold for better edge detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        quality_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 80000:  # Minimum area threshold
                continue
            
            # Check contour properties
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Reasonable aspect ratio for figures
            if 0.3 < aspect_ratio < 3.0 and w > 300 and h > 200:
                quality_contours.append(contour)
        
        return quality_contours

    def _estimate_image_bbox(self, page, pix):
        """Estimate the position of an embedded image on the page."""
        # This is a simplified estimation - in practice, getting exact position is complex
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Assume image is centered or proportionally placed
        img_aspect = pix.width / pix.height
        page_aspect = page_width / page_height
        
        if img_aspect > page_aspect:
            # Image is wider - fit to width
            scale = min(page_width * 0.8 / pix.width, page_height * 0.8 / pix.height)
        else:
            # Image is taller - fit to height
            scale = min(page_width * 0.8 / pix.width, page_height * 0.8 / pix.height)
        
        scaled_width = pix.width * scale
        scaled_height = pix.height * scale
        
        # Center the image
        x1 = (page_width - scaled_width) / 2
        y1 = (page_height - scaled_height) / 2
        x2 = x1 + scaled_width
        y2 = y1 + scaled_height
        
        return [x1, y1, x2, y2]

    def _filter_and_validate_figures(self, existing_figures: List[Dict], new_figures: List[Dict]) -> List[Dict]:
        """Filter new figures against existing ones and validate quality."""
        validated_figures = []
        
        for new_fig in new_figures:
            # Quality validation
            if not self._meets_quality_standards(new_fig):
                continue
            
            # Check for duplicates
            is_duplicate = False
            new_bbox = new_fig.get("bbox", [0, 0, 0, 0])
            
            for existing_fig in existing_figures:
                existing_bbox = existing_fig.get("bbox", [0, 0, 0, 0])
                
                # Check overlap
                iou = self._calculate_iou(new_bbox, existing_bbox)
                if iou > 0.4:  # 40% overlap threshold
                    # Keep the higher quality/confidence one
                    if (new_fig.get("confidence", 0) > existing_fig.get("confidence", 0) or
                        new_fig.get("method") == "embedded_image"):
                        # Remove existing lower quality figure
                        if existing_fig in existing_figures:
                            existing_figures.remove(existing_fig)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                validated_figures.append(new_fig)
        
        return validated_figures

    def _filter_and_validate_tables(self, existing_tables: List[Dict], new_tables: List[Dict]) -> List[Dict]:
        """Filter tables with similar logic to figures."""
        return self._filter_and_validate_figures(existing_tables, new_tables)

    def _meets_quality_standards(self, figure: Dict) -> bool:
        """Check if figure meets quality standards."""
        dims = figure.get("dimensions", {})
        width = dims.get("width", 0)
        height = dims.get("height", 0)
        
        # Size requirements
        if width < 200 or height < 150:
            return False
        
        # Area requirement
        if width * height < 40000:
            return False
        
        # Confidence requirement
        if figure.get("confidence", 0) < 0.6:
            return False
        
        # File size check (if available)
        file_size = figure.get("file_size", 0)
        if file_size > 0 and file_size < 5000:  # Less than 5KB
            return False
        
        return True

    def _final_figure_filtering(self, figures: List[Dict]) -> List[Dict]:
        """Apply final filtering to remove remaining false positives."""
        if not figures:
            return figures
        
        # Sort by confidence and method preference
        method_priority = {"embedded_image": 3, "layout_analysis_improved": 2, "contour_detection_improved": 1}
        
        figures.sort(key=lambda x: (
            method_priority.get(x.get("method", ""), 0),
            x.get("confidence", 0),
            x.get("dimensions", {}).get("width", 0) * x.get("dimensions", {}).get("height", 0)
        ), reverse=True)
        
        # Remove very similar figures (stricter final pass)
        final_figures = []
        for fig in figures:
            is_too_similar = False
            fig_bbox = fig.get("bbox", [0, 0, 0, 0])
            
            for existing in final_figures:
                existing_bbox = existing.get("bbox", [0, 0, 0, 0])
                if self._calculate_iou(fig_bbox, existing_bbox) > 0.3:
                    is_too_similar = True
                    break
            
            if not is_too_similar:
                final_figures.append(fig)
        
        # Limit total number to reasonable amount
        max_figures_per_page = 3
        page_counts = {}
        filtered_final = []
        
        for fig in final_figures:
            page = fig.get("page", 1)
            page_counts[page] = page_counts.get(page, 0) + 1
            
            if page_counts[page] <= max_figures_per_page:
                filtered_final.append(fig)
        
        return filtered_final

    def _final_table_filtering(self, tables: List[Dict]) -> List[Dict]:
        """Apply final filtering to tables."""
        return self._final_figure_filtering(tables)  # Use same logic

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


# Main integration function - updated to use improved extractor
def extract_figures_with_publaynet(pdf_path: str, output_dir: str = None) -> dict:
    """
    Extract figures using improved hybrid approach with better filtering.
    """
    if output_dir is None:
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(pdf_dir, f"{pdf_name}_improved_figures")

    try:
        # Use the improved extractor
        extractor = ImprovedFigureExtractor(confidence_threshold=0.7)
        results = extractor.extract_from_pdf(pdf_path, output_dir)

        return {
            "success": True,
            "figures": results["figures"],
            "tables": results["tables"],
            "total_figures": len(results["figures"]),
            "total_tables": len(results["tables"]),
            "output_dir": output_dir,
            "method": "improved_hybrid_extraction",
            "methods_used": results.get("methods_used", []),
            "method_results": results.get("method_results", {}),
        }

    except Exception as e:
        print(f"⚠️  Improved extraction failed: {e}")
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

    parser = argparse.ArgumentParser(description="Improved Figure Extraction")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", default="improved_extraction", help="Output directory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"🚀 Starting improved figure extraction for: {args.pdf_path}")

    results = extract_figures_with_publaynet(args.pdf_path, args.output)

    print(f"\n🎯 Improved Results:")
    print(f"✅ Success: {results['success']}")
    print(f"🖼️  Quality Figures: {results['total_figures']}")
    print(f"📋 Tables: {results['total_tables']}")
    print(f"📁 Output: {results['output_dir']}")
    print(f"🔧 Methods used: {', '.join(results.get('methods_used', []))}")

    if results.get("method_results"):
        print(f"\n📊 Method breakdown:")
        for method, method_result in results["method_results"].items():
            figures = len(method_result.get("figures", []))
            tables = len(method_result.get("tables", []))
            print(f"   {method}: {figures} figures, {tables} tables")