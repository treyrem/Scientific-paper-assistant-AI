#!/usr/bin/env python3
"""
Test script for PubLayNet figure extraction integration
Run this to verify your setup is working correctly
"""

import os
import sys
import traceback
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("🔍 Testing dependencies...")
    
    dependencies = [
        ("PyMuPDF", "fitz"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("PyTorch", "torch"),
        ("Requests", "requests"),
    ]
    
    missing = []
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    # Test Detectron2 separately (more complex)
    try:
        import detectron2
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        print("✓ Detectron2")
    except ImportError:
        print("✗ Detectron2 - MISSING")
        missing.append("Detectron2")
    except Exception as e:
        print(f"✗ Detectron2 - ERROR: {e}")
        missing.append("Detectron2")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Please install them using the requirements file.")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def test_publaynet_extractor():
    """Test the PubLayNet extractor initialization"""
    print("\n🔍 Testing PubLayNet extractor...")
    
    try:
        from publaynet_figure_extractor import PubLayNetFigureExtractor
        
        # Test initialization (this will download model if needed)
        print("Initializing PubLayNet extractor...")
        extractor = PubLayNetFigureExtractor(confidence_threshold=0.7)
        
        print("✅ PubLayNet extractor initialized successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure publaynet_figure_extractor.py is in your Python path")
        return False
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print("This might be due to missing dependencies or model download issues")
        traceback.print_exc()
        return False

def test_enhanced_analyzer():
    """Test the enhanced paper analyzer"""
    print("\n🔍 Testing enhanced paper analyzer...")
    
    try:
        from paper_analyzer_integration import EnhancedPaperAnalyzer
        
        # Test initialization
        analyzer = EnhancedPaperAnalyzer(use_publaynet=True)
        print("✅ Enhanced paper analyzer initialized successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure paper_analyzer_integration.py is in your Python path")
        return False
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        traceback.print_exc()
        return False

def test_with_sample_pdf():
    """Test with a sample PDF if available"""
    print("\n🔍 Looking for sample PDF files...")
    
    # Look for PDF files in current directory
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if not pdf_files:
        print("ℹ️  No PDF files found in current directory")
        print("To test with a real PDF, add a scientific paper PDF to this directory")
        return True
    
    # Use the first PDF found
    test_pdf = pdf_files[0]
    print(f"📄 Found PDF: {test_pdf}")
    
    try:
        from paper_analyzer_integration import extract_figures_with_publaynet
        
        print("Running figure extraction test...")
        
        # Create test output directory
        test_output = "test_extraction_output"
        Path(test_output).mkdir(exist_ok=True)
        
        # Run extraction
        results = extract_figures_with_publaynet(str(test_pdf), test_output)
        
        print(f"✅ Figure extraction completed!")
        print(f"   Method used: {results['method']}")
        print(f"   Figures extracted: {results['total_figures']}")
        print(f"   Tables extracted: {results['total_tables']}")
        print(f"   Output directory: {results['output_dir']}")
        
        # Check if files were created
        if results['total_figures'] > 0:
            print(f"   Figure files created in: {results['output_dir']}")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        traceback.print_exc()
        return False

def test_gpu_availability():
    """Test GPU availability for faster processing"""
    print("\n🔍 Testing GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            
            print(f"✅ GPU available!")
            print(f"   GPU count: {gpu_count}")
            print(f"   Current GPU: {gpu_name}")
            print("   PubLayNet will use GPU for faster processing")
        else:
            print("ℹ️  GPU not available - will use CPU")
            print("   Processing will be slower but still functional")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test error: {e}")
        return False

def create_sample_test_script():
    """Create a sample script for users to test their own PDFs"""
    
    sample_script = '''#!/usr/bin/env python3
"""
Sample script to test PubLayNet figure extraction on your PDF
"""

from paper_analyzer_integration import EnhancedPaperAnalyzer, extract_figures_with_publaynet

def test_your_pdf(pdf_path):
    """Test figure extraction on your PDF"""
    
    print(f"Testing figure extraction on: {pdf_path}")
    
    # Option 1: Quick figure extraction only
    print("\\n1. Quick figure extraction...")
    figure_results = extract_figures_with_publaynet(pdf_path, "quick_test_output")
    print(f"Figures extracted: {figure_results['total_figures']}")
    print(f"Tables extracted: {figure_results['total_tables']}")
    
    # Option 2: Full enhanced analysis
    print("\\n2. Full enhanced analysis...")
    analyzer = EnhancedPaperAnalyzer(use_publaynet=True, confidence_threshold=0.7)
    full_results = analyzer.analyze_paper(pdf_path, "full_test_output")
    
    print(f"Analysis method: {full_results['analysis_method']}")
    print(f"Figures: {len(full_results['figures'])}")
    print(f"Tables: {len(full_results['tables'])}")
    print(f"Sections: {len(full_results['sections'])}")
    
    # Print figure details
    if full_results['figures']:
        print("\\nExtracted figures:")
        for i, fig in enumerate(full_results['figures'][:3]):  # Show first 3
            print(f"  {i+1}. {fig['figure_id']} (Page {fig['page']}, Confidence: {fig['confidence']:.2f})")
            if fig.get('caption'):
                print(f"     Caption: {fig['caption'][:100]}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_sample.py <path_to_your_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_your_pdf(pdf_path)
'''
    
    with open("test_sample.py", "w") as f:
        f.write(sample_script)
    
    print("\n📝 Created test_sample.py")
    print("   Use this to test your own PDFs: python test_sample.py your_paper.pdf")

def main():
    """Main test function"""
    print("🚀 PubLayNet Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("GPU Availability", test_gpu_availability), 
        ("PubLayNet Extractor", test_publaynet_extractor),
        ("Enhanced Analyzer", test_enhanced_analyzer),
        ("Sample PDF Processing", test_with_sample_pdf),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your PubLayNet integration is ready to use.")
        create_sample_test_script()
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed correctly.")
    
    print("\n📚 Next steps:")
    print("1. Copy publaynet_figure_extractor.py and paper_analyzer_integration.py to your project")
    print("2. Update your paper_analyzer2.py to use the new functions")
    print("3. Test with your scientific papers!")

if __name__ == "__main__":
    main()
