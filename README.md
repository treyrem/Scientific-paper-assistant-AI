Scientific Paper Assistant AI
A comprehensive tool for analyzing, summarizing, and interacting with scientific research papers in PDF format. This project helps researchers, students, and academics to quickly understand and extract key information from scientific papers.
Features

PDF Analysis: Extract structured information from research papers including title, authors, sections, and layout analysis
Automatic Summarization: Generate extractive and synthesized summaries of papers
Key Concept Extraction: Identify and define important terms and concepts
Interactive Chatbot: Ask questions about a paper and get contextual answers
Quiz Generation: Create multiple-choice quizzes to test paper comprehension
User-friendly Interface: Web-based Streamlit application for easy interaction

Components
The project consists of four main Python scripts:

paper_analyzer2.py: Core library for analyzing PDFs and extracting information
paper_chatbot.py: Interactive chatbot to answer questions about papers
quiz_generator.py: Tool to generate quizzes based on paper content
streamlit_app.py: Web interface integrating all components

Requirements
Core Dependencies

Python 3.7+
PyMuPDF (fitz)
nltk
numpy
tqdm
scikit-learn
transformers
torch (optional for GPU support)

OpenAI Integration (Optional but Recommended)

openai

Web Interface

streamlit

Installation

Clone the repository:
git clone https://github.com/treyrem/scientific-paper-assistant-ai.git
cd scientific-paper-assistant-ai

Install dependencies:
pip install -r requirements.txt

Set up OpenAI API key (for enhanced summaries, quiz generation, and chatbot):

Create a .env file at api_keys/OPEN_AI_KEY.env with your API key:
OPENAI_API_KEY=your_api_key_here




Usage
Command Line Tools
Paper Analysis
bash python paper_analyzer2.py path/to/paper.pdf --output output/path.json
Optional arguments:

--no-gpu: Disable GPU usage
--debug: Enable debug logging
--openai-model MODEL: Specify OpenAI model (default: "gpt-3.5-turbo")

Paper Chatbot
bashpython paper_chatbot.py path/to/analysis.json
Optional arguments:

--openai-model MODEL: Specify OpenAI model
--debug: Enable debug logging

Quiz Generator
bashpython quiz_generator.py path/to/analysis.json -o output/quiz.json -n 5
Optional arguments:

--num-questions NUM, -n NUM: Number of quiz questions to generate (default: 5)
--openai-model MODEL: Specify OpenAI model
--debug: Enable debug logging

Web Interface
bashstreamlit run streamlit_app.py
This launches a web interface with three main tabs:

Analysis & Summary: Upload and analyze papers
Quiz: Generate questions about the paper
Chatbot: Ask questions about the paper's content

How the System Works
PDF Processing

The system analyzes PDFs using layout analysis to identify sections by detecting:

Font size differences
Bold text
Section header patterns
Content organization

Key Features

Metadata Extraction: Extracts title, authors, publication year, and DOI.
Section Identification: Recognizes standard paper sections (abstract, introduction, methods, results, etc.) based on layout and text patterns.
Summarization: Uses two approaches:

Extractive summarization using TF-IDF to identify important sentences
Synthesized summarization using OpenAI to create coherent summaries


Key Concept Extraction: Identifies important terms and provides definitions.
Interactive Chat: Allows asking questions about a paper with relevant answers.
Quiz Generation: Creates multiple-choice questions to test understanding.

OpenAI Integration
The system has optional integration with OpenAI for enhanced features:

Improved summaries
Quiz generation
Chatbot functionality

It loads the API key from a .env file located at a specific path.
Implementation Details
paper_analyzer2.py

Uses PyMuPDF (fitz) for PDF text extraction
Implements layout analysis to identify sections
Extracts metadata using regex patterns
Uses TF-IDF for extractive summarization
Integrates with OpenAI for synthesized summaries

paper_chatbot.py

Takes analysis JSON as input
Prepares paper context for OpenAI
Manages conversation history
Handles API interaction

quiz_generator.py

Takes analysis JSON as input
Constructs prompts for OpenAI to generate quiz questions
Parses responses into structured question format
Saves quizzes as JSON

streamlit_app.py

Provides a web interface with tabs for different functions
Handles file uploads and temporary storage
Displays analysis results and summaries
Provides interactive quiz interface
Implements chat functionality

Project Structure
scientific-paper-assistant-ai/
├── paper_analyzer2.py         # Core PDF analysis
├── paper_chatbot.py           # Interactive chatbot
├── quiz_generator.py          # Quiz question generator
├── streamlit_app.py           # Web interface
├── api_keys/                  # API key storage
│   └── OPEN_AI_KEY.env        # OpenAI API key file
├── requirements.txt           # Project dependencies
└── README.md                  # This file
Limitations

Best results with research papers following standard academic formatting
May struggle with papers containing complex layouts or non-standard formatting
Summaries and extractions are automated and may not capture all nuances
OpenAI integration required for advanced features (synthesized summaries, quiz, chatbot)

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
[Insert appropriate license information here]
Acknowledgments
This project leverages several open-source libraries and the OpenAI API to provide its functionality.
