import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer
import os
from collections import defaultdict, Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("layout_analyzer")

# Fix for PIL Image.LINEAR issue - add this before importing layoutparser
import PIL.Image
if not hasattr(PIL.Image, 'LINEAR'):
    # Patch PIL.Image to add LINEAR attribute
    PIL.Image.LINEAR = PIL.Image.BILINEAR
    logger.info("Added PIL.Image.LINEAR attribute")

# Now import layoutparser
import layoutparser as lp

def extract_text_fragments(pdf_path, total_pages):
    """Extract text fragments from all pages of the PDF"""
    all_fragments = []  # will hold dicts: {page, text, bbox}
    
    # Try to extract text with pdfminer
    try:
        logger.info(f"Extracting text fragments using pdfminer from {total_pages} pages")
        for p in range(total_pages):
            page_fragments = []
            for element in extract_pages(pdf_path, page_numbers=[p]):
                if isinstance(element, LTTextContainer):
                    txt = element.get_text().strip()
                    if not txt:
                        continue
                    page_fragments.append({
                        "page": p + 1,
                        "text": txt,
                        "bbox": element.bbox
                    })
            
            # If no fragments found for this page with pdfminer, try fitz (PyMuPDF) as backup
            if not page_fragments:
                logger.info(f"No fragments found with pdfminer for page {p+1}, trying PyMuPDF")
                doc = fitz.open(pdf_path)
                page = doc.load_page(p)
                text_blocks = page.get_text("blocks")
                for block in text_blocks:
                    # block[0:4] contains the bbox (x0, y0, x1, y1)
                    # block[4] contains the text
                    txt = block[4].strip()
                    if not txt:
                        continue
                    page_fragments.append({
                        "page": p + 1,
                        "text": txt,
                        "bbox": block[0:4]
                    })
                doc.close()
            
            all_fragments.extend(page_fragments)
            logger.info(f"Extracted {len(page_fragments)} fragments from page {p+1}")
    except Exception as e:
        logger.error(f"Error extracting text fragments: {str(e)}. Trying alternative method...")
        try:
            # Backup method: use extract_text from pdfminer
            for p in range(total_pages):
                text = extract_text(pdf_path, page_numbers=[p])
                if text.strip():
                    all_fragments.append({
                        "page": p + 1,
                        "text": text.strip(),
                        "bbox": (0, 0, 612, 792)  # Default letter size
                    })
        except Exception as e2:
            logger.error(f"Failed to extract text: {str(e2)}")
    
    logger.info(f"Total fragments extracted: {len(all_fragments)}")
    return all_fragments

def initialize_layout_model():
    """Initialize and return a layout detection model"""
    try:
        # Try first with the AutoLayoutModel, which is more flexible
        logger.info("Attempting to initialize AutoLayoutModel")
        model = lp.AutoLayoutModel('lp://EfficientDete/PubLayNet')
        model_type = "AutoLayoutModel"
        logger.info("Successfully initialized AutoLayoutModel")
    except Exception as e:
        logger.warning(f"Error with AutoLayoutModel: {str(e)}")
        try:
            # Fall back to Detectron2LayoutModel
            logger.info("Attempting to initialize Detectron2LayoutModel")
            model = lp.Detectron2LayoutModel(
                config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                enforce_cpu=True
            )
            model_type = "Detectron2LayoutModel"
            logger.info("Successfully initialized Detectron2LayoutModel")
        except Exception as e:
            logger.error(f"Error initializing layout detection model: {str(e)}")
            return None, "None"
    
    return model, model_type

def process_page(page_num, doc, all_fragments, model, output_dir="out"):
    """Process a single page and detect layouts"""
    blocks = []
    logger.info(f"Processing page {page_num+1}")
    
    # Rasterize
    page = doc.load_page(page_num)
    mat = fitz.Matrix(2, 2)  # 2× zoom
    pix = page.get_pixmap(matrix=mat)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = arr[..., :3]
    img = Image.fromarray(arr)
    
    # Collect page-specific fragments
    page_fragments = [frag for frag in all_fragments if frag["page"] == page_num+1]
    logger.info(f"Found {len(page_fragments)} fragments for page {page_num+1}")
    
    if model is not None:
        try:
            # Run layout detection
            layout = model.detect(np.array(img), threshold=0.5)
            logger.info(f"Detected {len(layout)} layout blocks on page {page_num+1}")
            
            for block in layout:
                # Crop & save for inspection
                x1, y1, x2, y2 = map(int, block.coordinates)
                crop = img.crop((x1, y1, x2, y2))
                os.makedirs(output_dir, exist_ok=True)
                crop_path = f"{output_dir}/p{page_num+1:02d}_{block.type}_{len(blocks)+1}.png"
                crop.save(crop_path)
                
                # Collect matching text fragments
                fragments = []
                for frag in page_fragments:
                    x0, y0, x3, y3 = frag["bbox"]
                    # containment heuristic
                    if x0 >= x1 and x3 <= x2 and y0 >= y1 and y3 <= y2:
                        fragments.append(frag["text"])
                
                blocks.append({
                    "page": page_num + 1,
                    "type": block.type,
                    "coords": block.coordinates,
                    "text": "\n".join(fragments),
                    "crop": crop_path
                })
        except Exception as e:
            logger.error(f"Error detecting layout on page {page_num+1}: {str(e)}")
            # Fall back to simple text extraction for this page
            page_text = "\n".join([frag["text"] for frag in page_fragments])
            blocks.append({
                "page": page_num + 1,
                "type": "Text (Simple)",
                "coords": [0, 0, img.width, img.height],
                "text": page_text,
                "crop": None
            })
    else:
        # Simple text extraction if no model is available
        page_text = "\n".join([frag["text"] for frag in page_fragments])
        logger.info(f"Using simple text extraction for page {page_num+1}")
        blocks.append({
            "page": page_num + 1,
            "type": "Text (Simple)",
            "coords": [0, 0, img.width, img.height],
            "text": page_text,
            "crop": None
        })
    
    logger.info(f"Created {len(blocks)} blocks for page {page_num+1}")
    return blocks

def analyze_paper(pdf_path, max_pages=5):
    """Main function to analyze a scientific paper PDF"""
    logger.info(f"Starting analysis of {pdf_path}")
    
    # Create output directory
    os.makedirs("out", exist_ok=True)
    
    # Open document
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    logger.info(f"Document has {total_pages} pages")
    
    # Document info
    doc_info = {
        "total_pages": total_pages,
        "processed_pages": min(max_pages, total_pages)
    }
    
    # Extract text fragments
    all_fragments = extract_text_fragments(pdf_path, total_pages)
    
    # Initialize layout model
    model, model_type = initialize_layout_model()
    logger.info(f"Using {model_type} for detection")
    
    # Process pages
    all_blocks = []
    for p in range(min(max_pages, total_pages)):
        logger.info(f"Processing page {p+1}...")
        page_blocks = process_page(p, doc, all_fragments, model)
        all_blocks.extend(page_blocks)
    
    # Group blocks by type
    by_type = defaultdict(list)
    for blk in all_blocks:
        by_type[blk["type"]].append(blk)
    
    # Calculate fragment summaries
    fragment_summary = calculate_fragment_summary(all_fragments, by_type)
    
    # Close the document
    doc.close()
    logger.info(f"Analysis complete, found {len(all_blocks)} blocks")
    
    return doc_info, all_fragments, all_blocks, by_type, fragment_summary

def calculate_fragment_summary(all_fragments, by_type):
    """Generate a summary of fragments"""
    summary = {
        "total_fragments": len(all_fragments),
        "fragments_per_page": defaultdict(int),
        "fragments_by_type": defaultdict(int),
        "avg_chars_per_fragment": 0,
        "word_count": 0,
        "word_frequency": Counter(),
        "fragments_with_numbers": 0,
        "fragments_with_special_chars": 0
    }
    
    # Calculate per-page fragment counts
    for frag in all_fragments:
        summary["fragments_per_page"][frag["page"]] += 1
        
        # Calculate word statistics
        words = frag["text"].split()
        summary["word_count"] += len(words)
        summary["word_frequency"].update(words)
        
        # Check for numbers and special characters
        if any(c.isdigit() for c in frag["text"]):
            summary["fragments_with_numbers"] += 1
        if any(not c.isalnum() and not c.isspace() for c in frag["text"]):
            summary["fragments_with_special_chars"] += 1
    
    # Calculate average characters per fragment
    if all_fragments:
        total_chars = sum(len(frag["text"]) for frag in all_fragments)
        summary["avg_chars_per_fragment"] = total_chars / len(all_fragments)
    
    # Count fragments by layout type
    for block_type, blocks in by_type.items():
        total_fragments_in_type = sum(1 for block in blocks if block["text"].strip())
        summary["fragments_by_type"][block_type] = total_fragments_in_type
    
    # Find most common words
    summary["most_common_words"] = summary["word_frequency"].most_common(10)
    
    return summary

if __name__ == "__main__":
    # This allows the module to be run directly for testing
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        doc_info, fragments, blocks, by_type, fragment_summary = analyze_paper(pdf_path)
        print(f"Document has {doc_info['total_pages']} pages")
        print(f"Extracted {len(fragments)} text fragments")
        print(f"Found {len(blocks)} blocks")
        for type_name, type_blocks in by_type.items():
            print(f"{type_name}: {len(type_blocks)} blocks")