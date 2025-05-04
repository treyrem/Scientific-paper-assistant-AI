# run_pipeline.py

import os
import sys
import logging
import torch
import argparse
from pathlib import Path

# Force environment variable to disable PubLayNet initially (safer)
os.environ["USE_PUBLAYNET"] = "false"

# Force CPU for PyTorch to avoid CUDA assertion errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set torch to use CPU by default
torch.cuda.is_available = lambda: False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import pipeline components from our new module
from pipeline_modules import EnhancedAnalysisPipeline


def main():
    """Run the fixed pipeline with safer defaults"""
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
            f"\nFull results saved to: {args.output_dir}/{Path(args.pdf_path).stem}_enhanced_analysis.json"
        )

    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        print(f"\nERROR: {str(e)}")
        print("\nTry running with --use-gpu=False or check logs for details")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
