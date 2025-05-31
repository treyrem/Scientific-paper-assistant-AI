# Integration updates for your paper_analyzer2.py

import sys
import os
from pathlib import Path

# Add the PubLayNet extractor to your imports
try:
    from publaynet_figure_extractor import PubLayNetFigureExtractor, integrate_with_existing_analyzer
except ImportError:
    print("Warning: PubLayNet figure extractor not available. Install dependencies:")
    print("pip install detectron2 torch torchvision opencv-python")
    PubLayNetFigureExtractor = None

class EnhancedPaperAnalyzer:
    """
    Enhanced version of your paper analyzer with PubLayNet figure extraction
    """
    
    def __init__(self, use_publaynet=True, confidence_threshold=0.7):
        """
        Initialize the enhanced paper analyzer
        
        Args:
            use_publaynet: Whether to use PubLayNet for figure extraction
            confidence_threshold: Confidence threshold for PubLayNet detections
        """
        self.use_publaynet = use_publaynet and PubLayNetFigureExtractor is not None
        self.confidence_threshold = confidence_threshold
        
        if self.use_publaynet:
            try:
                self.figure_extractor = PubLayNetFigureExtractor(
                    confidence_threshold=confidence_threshold
                )
                print("✓ PubLayNet figure extractor initialized")
            except Exception as e:
                print(f"Warning: Could not initialize PubLayNet extractor: {e}")
                self.use_publaynet = False
                self.figure_extractor = None
        else:
            self.figure_extractor = None
            print("Using fallback figure extraction method")
    
    def analyze_paper(self, pdf_path: str, output_dir: str = None) -> dict:
        """
        Enhanced paper analysis with PubLayNet figure extraction
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory for extracted content
            
        Returns:
            Dictionary containing analysis results with enhanced figure extraction
        """
        if output_dir is None:
            output_dir = f"analysis_{Path(pdf_path).stem}"
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {
            "pdf_path": pdf_path,
            "analysis_method": "enhanced_with_publaynet" if self.use_publaynet else "standard",
            "figures": [],
            "tables": [],
            "layout_analysis": None,
            "text_content": "",
            "sections": [],
            "metadata": {}
        }
        
        try:
            # Enhanced figure and layout extraction with PubLayNet
            if self.use_publaynet:
                print("🔍 Running PubLayNet-based figure extraction...")
                publaynet_results = integrate_with_existing_analyzer(
                    pdf_path, 
                    os.path.join(output_dir, "publaynet_extraction")
                )
                
                # Integrate PubLayNet results
                results.update({
                    "figures": publaynet_results["extraction_results"]["figures"],
                    "tables": publaynet_results["extraction_results"]["tables"],
                    "layout_analysis": publaynet_results["structure_analysis"],
                    "publaynet_results": publaynet_results["extraction_results"]
                })
                
                print(f"✓ Extracted {len(results['figures'])} figures and {len(results['tables'])} tables")
            
            # Continue with your existing text extraction and analysis
            # (This would be your existing paper_analyzer2.py logic)
            results.update(self._extract_text_and_sections(pdf_path))
            results.update(self._extract_metadata(pdf_path))
            
            # Enhanced section-figure mapping
            if self.use_publaynet and results["figures"]:
                results["section_figure_mapping"] = self._map_figures_to_sections(
                    results["sections"], results["figures"]
                )
            
            # Save comprehensive results
            self._save_results(results, output_dir)
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            results["error"] = str(e)
        
        return results
    
    def _extract_text_and_sections(self, pdf_path: str) -> dict:
        """
        Extract text content and identify sections
        (Your existing implementation from paper_analyzer2.py would go here)
        """
        import fitz
        
        # Placeholder for your existing text extraction logic
        doc = fitz.open(pdf_path)
        text_content = ""
        sections = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            text_content += page_text + "\n"
        
        doc.close()
        
        # Your existing section identification logic would go here
        # This is a simplified version
        sections = self._identify_sections(text_content)
        
        return {
            "text_content": text_content,
            "sections": sections
        }
    
    def _identify_sections(self, text_content: str) -> list:
        """
        Identify paper sections from text content
        (Your existing section identification logic)
        """
        # Simplified section identification
        # Replace with your existing logic from paper_analyzer2.py
        
        sections = []
        common_sections = [
            "abstract", "introduction", "methodology", "methods",
            "results", "discussion", "conclusion", "references"
        ]
        
        lines = text_content.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            if any(section in line_lower for section in common_sections):
                # Save previous section
                if current_section:
                    sections.append({
                        "title": current_section,
                        "content": "\n".join(section_content),
                        "type": "section"
                    })
                
                # Start new section
                current_section = line.strip()
                section_content = []
            else:
                if current_section:
                    section_content.append(line)
        
        # Add last section
        if current_section:
            sections.append({
                "title": current_section,
                "content": "\n".join(section_content),
                "type": "section"
            })
        
        return sections
    
    def _extract_metadata(self, pdf_path: str) -> dict:
        """
        Extract paper metadata
        (Your existing metadata extraction logic)
        """
        import fitz
        
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        # Your existing metadata extraction logic would go here
        # This is a simplified version
        
        extracted_metadata = {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", "")
        }
        
        doc.close()
        
        return {"metadata": extracted_metadata}
    
    def _map_figures_to_sections(self, sections: list, figures: list) -> dict:
        """
        Map extracted figures to paper sections based on page numbers and content
        """
        section_figure_mapping = {}
        
        for section in sections:
            section_title = section["title"]
            section_content = section["content"].lower()
            
            # Find figures mentioned in this section
            referenced_figures = []
            for figure in figures:
                figure_id = figure.get("figure_id", "")
                page_num = figure.get("page", 0)
                
                # Check if figure is referenced in section text
                if (f"figure {page_num}" in section_content or 
                    f"fig. {page_num}" in section_content or
                    f"fig {page_num}" in section_content):
                    referenced_figures.append(figure)
            
            if referenced_figures:
                section_figure_mapping[section_title] = referenced_figures
        
        return section_figure_mapping
    
    def _save_results(self, results: dict, output_dir: str):
        """Save analysis results to JSON file"""
        import json
        
        output_file = Path(output_dir) / "enhanced_analysis_results.json"
        
        # Create a serializable version of results
        serializable_results = self._make_serializable(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Analysis results saved to {output_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        import json
        
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

# Integration function to replace your existing figure extraction
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
        output_dir = f"figures_{Path(pdf_path).stem}"
    
    try:
        # Use PubLayNet for figure extraction
        if PubLayNetFigureExtractor is not None:
            extractor = PubLayNetFigureExtractor(confidence_threshold=0.7)
            results = extractor.extract_from_pdf(pdf_path, output_dir)
            
            return {
                "success": True,
                "figures": results["figures"],
                "tables": results["tables"],
                "total_figures": len(results["figures"]),
                "total_tables": len(results["tables"]),
                "output_dir": output_dir,
                "method": "publaynet"
            }
        else:
            # Fallback to basic extraction
            return extract_figures_fallback(pdf_path, output_dir)
            
    except Exception as e:
        print(f"PubLayNet extraction failed: {e}")
        return extract_figures_fallback(pdf_path, output_dir)

def extract_figures_fallback(pdf_path: str, output_dir: str) -> dict:
    """
    Fallback figure extraction method using basic PyMuPDF
    """
    import fitz
    
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
                    
                    figures.append({
                        "figure_id": f"fig_{page_num+1}_{img_index+1}",
                        "file_path": str(figure_path),
                        "page": page_num + 1,
                        "type": "figure",
                        "confidence": 1.0,  # No confidence score for fallback method
                        "method": "fallback"
                    })
                
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
        "method": "fallback"
    }

# Command line interface
def main():
    """Main function for command line usage"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Enhanced Scientific Paper Analyzer with PubLayNet"
    )
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--confidence", "-c", type=float, default=0.7,
                       help="Confidence threshold for PubLayNet")
    parser.add_argument("--no-publaynet", action="store_true",
                       help="Disable PubLayNet and use fallback method")
    parser.add_argument("--figures-only", action="store_true",
                       help="Extract figures only, skip full analysis")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return 1
    
    # Set output directory
    if args.output is None:
        args.output = f"analysis_{Path(args.pdf_path).stem}"
    
    try:
        if args.figures_only:
            # Extract figures only
            print("Extracting figures only...")
            if args.no_publaynet:
                results = extract_figures_fallback(args.pdf_path, args.output)
            else:
                results = extract_figures_with_publaynet(args.pdf_path, args.output)
            
            print(f"\n✓ Figure extraction completed!")
            print(f"Figures extracted: {results['total_figures']}")
            print(f"Tables extracted: {results['total_tables']}")
            print(f"Method used: {results['method']}")
            print(f"Output directory: {results['output_dir']}")
            
        else:
            # Full paper analysis
            print("Running full paper analysis...")
            analyzer = EnhancedPaperAnalyzer(
                use_publaynet=not args.no_publaynet,
                confidence_threshold=args.confidence
            )
            
            results = analyzer.analyze_paper(args.pdf_path, args.output)
            
            print(f"\n✓ Paper analysis completed!")
            print(f"Analysis method: {results['analysis_method']}")
            print(f"Figures extracted: {len(results['figures'])}")
            print(f"Tables extracted: {len(results['tables'])}")
            print(f"Sections identified: {len(results['sections'])}")
            print(f"Output directory: {args.output}")
            
            if "section_figure_mapping" in results:
                print(f"Section-figure mappings: {len(results['section_figure_mapping'])}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
    