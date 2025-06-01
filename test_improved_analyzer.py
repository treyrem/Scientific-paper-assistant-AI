#!/usr/bin/env python3
"""
Test script for the improved paper analyzer.
This replaces your existing paper_analyzer2.py with the improved version.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path so we can import the improved modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the improved analyzer
try:
    from fixed_paper_analyzer import ImprovedPaperProcessor, main
    from improved_figure_extractor import extract_figures_with_publaynet
    print("✅ Successfully imported improved modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure both 'fixed_paper_analyzer.py' and 'improved_figure_extractor.py' are in the same directory")
    sys.exit(1)


def test_improved_analyzer(pdf_path: str):
    """Test the improved paper analyzer with your PDF."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load OpenAI API key (same path as your original)
    openai_api_key = None
    try:
        from dotenv import load_dotenv
        env_path = Path(r"C:\LabGit\Scientific-paper-assistant-AI\api_keys\OPEN_AI_KEY.env").resolve()
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
            openai_api_key = os.getenv("OPENAI_API_KEY")
            print(f"✅ Loaded OpenAI API key from: {env_path}")
        else:
            print(f"⚠️  OpenAI key file not found at: {env_path}")
    except ImportError:
        print("⚠️  python-dotenv not installed - OpenAI synthesis will be skipped")
    
    # Create improved processor
    try:
        processor = ImprovedPaperProcessor(
            use_gpu=True,
            openai_api_key=openai_api_key,
            openai_model="gpt-3.5-turbo"
        )
        print("✅ Created improved processor")
    except Exception as e:
        print(f"❌ Error creating processor: {e}")
        return None
    
    # Process the paper
    try:
        print(f"\n🚀 Processing paper: {pdf_path}")
        analysis = processor.process_paper(
            pdf_path,
            extract_figures=True,
            figure_output_dir=None
        )
        
        if analysis:
            # Save results
            output_path = pdf_path.replace(".pdf", "_improved_analysis.json")
            analysis.save_to_file(output_path)
            
            print(f"\n✅ Analysis complete! Results saved to: {output_path}")
            print(f"\n=== IMPROVED ANALYSIS RESULTS ===")
            print(f"📄 Title: {analysis.title or 'N/A'}")
            print(f"👥 Authors: {', '.join(analysis.authors) if analysis.authors else 'N/A'}")
            print(f"📅 Year: {analysis.publication_year or 'N/A'}")
            print(f"🔗 DOI: {analysis.doi or 'N/A'}")
            
            print(f"\n📋 Sections Found ({len(analysis.sections)}):")
            for section_type, section in analysis.sections.items():
                word_count = len(section.content.split())
                print(f"  - {section_type}: {word_count} words (pages {section.page_numbers})")
            
            if analysis.full_summary:
                print(f"\n📝 Full Summary:")
                print(analysis.full_summary)
            
            if analysis.significance:
                print(f"\n🎯 Significance:")
                print(analysis.significance)
            
            if analysis.keywords:
                print(f"\n🔑 Keywords ({len(analysis.keywords)}):")
                print(", ".join(analysis.keywords[:10]))
            
            print(f"\n🖼️  Figures: {analysis.total_figures_extracted}")
            print(f"📊 Tables: {analysis.total_tables_extracted}")
            print(f"🔧 Method: {analysis.figure_extraction_method}")
            
            if analysis.figures:
                print(f"\n📸 Sample figures:")
                for i, figure in enumerate(analysis.figures[:3]):
                    print(f"  {i+1}. {figure.get('figure_id', 'Unknown')} (Page {figure.get('page', '?')})")
                    print(f"     Method: {figure.get('method', 'Unknown')}")
                    print(f"     Confidence: {figure.get('confidence', 0):.2f}")
                    print(f"     File: {figure.get('file_path', 'N/A')}")
                
                if len(analysis.figures) > 3:
                    print(f"  ... and {len(analysis.figures) - 3} more figures")
            
            return analysis
            
        else:
            print("❌ Analysis failed - no results returned")
            return None
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_original():
    """Compare results with your original analyzer."""
    print("\n" + "="*60)
    print("COMPARISON WITH ORIGINAL RESULTS:")
    print("="*60)
    print("Original Issues:")
    print("❌ No sections identified ([] empty sections)")
    print("❌ 284 figures extracted (too many false positives)")
    print("❌ No meaningful summaries generated")
    print("❌ No keywords extracted")
    print("❌ Poor title extraction ('Published as a conference paper...')")
    print()
    print("Expected Improvements:")
    print("✅ Better section detection using multiple strategies")
    print("✅ Fewer, higher-quality figures with filtering")
    print("✅ Meaningful summaries and keywords")
    print("✅ Better metadata extraction")
    print("✅ Fallback text extraction methods")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Improved Paper Analyzer")
    parser.add_argument("pdf_path", help="Path to PDF file to analyze")
    parser.add_argument("--compare", action="store_true", help="Show comparison with original")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"❌ PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    print("🔧 Testing Improved Paper Analyzer")
    print("="*50)
    
    if args.compare:
        compare_with_original()
    
    # Run the test
    result = test_improved_analyzer(args.pdf_path)
    
    if result:
        print(f"\n🎉 Test completed successfully!")
        print(f"📁 Check the output files in the same directory as your PDF")
    else:
        print(f"\n💥 Test failed - check the logs above for details")


# Alternative: run the main function directly (like your original script)
def run_as_main():
    """Run using the main function from the improved analyzer."""
    from fixed_paper_analyzer import main
    main()


if __name__ == "__main__" and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    # If called with a PDF path as argument, run our test
    pass
else:
    # Otherwise, you can uncomment this to run the main function directly
    # run_as_main()
    pass